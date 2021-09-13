import os
import sys
import time
import warnings

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.backends.cudnn as cudnn
from torch.multiprocessing import set_start_method

from arguments import parse_arguments_train
from data.dirt_dataset import dirt_provider
from data.common import visualize
from util import Meter, MetricsLogger, DiceLoss, TrainerModes, set_parameter_requires_grad

warnings.filterwarnings('ignore')


class Trainer(object):
    '''This class takes care of training and validation of our model'''
    def __init__(self, model, checkpoint, mode, args):
        self.num_workers = 4
        self.batch_size = {"train": 4, "val": 1}
        self.accumulation_steps = 32 // self.batch_size['train']
        self.lr = 5e-4
        self.num_epochs = 1
        self.start_epoch = 0
        self.best_dice = 0
        self.best_loss = 1e6
        self.best_seg_loss = 1e6
        self.phases = ["train", "val"]
        self.mode = mode
        self.args = args
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        self.net = self.net.to(self.device)
        self.freeze_backbone_if_needed()
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.criterion_cls = torch.nn.BCEWithLogitsLoss()
        cudnn.benchmark = False
        self.dataloaders = {
            phase: dirt_provider(
                data_folder=self.args.data_root,
                phase=phase,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }
        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        self.logger = MetricsLogger()
        # self.tensorboard_writer = SummaryWriter("../logs/runs")

    def load_checkpoint(self, checkpoint):

        def try_to_load(field_to_store, checkpoint, key):
            try:
                field_to_store.load_state_dict(checkpoint[key])
            except:
                print(f"No '{key}' key in checkpoint!")

        def try_to_assign(checkpoint, key, default_val=None, verbose=True):
            try:
                if verbose:
                    print(f"Key '{key}' is {checkpoint[key]}")
                return checkpoint[key]
            except:
                print(f"No '{key}' key in checkpoint!")
                return default_val

        def load_model_state_dict(net, checkpoint, backend, key="state_dict"):
            # Check if there is a classification head
            try:
                net.load_state_dict(checkpoint[key])
            except Exception as e:
                print("Trying to load model without classifier")
                temp_model = smp.Unet(backend, encoder_weights='imagenet', classes=4, activation=None)
                temp_model.load_state_dict(checkpoint[key])
                net.encoder = temp_model.encoder
                net.decoder = temp_model.decoder
                net.segmentation_head = temp_model.segmentation_head
                print("Successfully loaded model without classifier")

        load_model_state_dict(self.net, checkpoint, self.args.backend)
        self.start_epoch = checkpoint["epoch"]
        self.best_dice = checkpoint["best_dice"]
        # self.best_loss = try_to_assign(checkpoint, "best_loss", self.best_loss)
        self.best_seg_loss = try_to_assign(checkpoint, "best_seg_loss", self.best_seg_loss)

    def freeze_backbone_if_needed(self):
        if self.mode == TrainerModes.cls:
            set_parameter_requires_grad(self.net.encoder)
            set_parameter_requires_grad(self.net.decoder)
            set_parameter_requires_grad(self.net.segmentation_head)

    @staticmethod
    def threshold_preds(masks_pred):
        masks_pred[masks_pred < 0.001] = -1.
        masks_pred[masks_pred >= 0.001] = 1.
        return masks_pred

    def forward(self, images, target_masks, target_lables):
        images = images.to(self.device)
        masks = target_masks.to(self.device)
        labels = target_lables.to(self.device)
        labels.squeeze_()
        if self.mode == TrainerModes.combine:
            masks_pred, labels_pred = self.net(images)
            loss_cls = self.criterion_cls(labels_pred, labels)
            loss_seg = self.criterion(masks_pred, masks)
            loss_dice = DiceLoss.forward(masks_pred, masks)
            # loss_dice = torch.tensor(5000)
            loss = (loss_cls, loss_seg, loss_dice)
        elif self.mode == TrainerModes.seg:
            masks_pred = self.net(images)
            masks_pred = self.threshold_preds(masks_pred)
            # modelFile = onnx.load('effnetb0_unet_golfv2_320x640.onnx')
            # inputArray = images.cpu().numpy()
            # output = caffe2.python.onnx.backend.run_model(modelFile, inputArray.astype(np.float32))

            loss = self.criterion(masks_pred, masks)
        elif self.mode == TrainerModes.cls:
            masks_pred, labels_pred = self.net(images)
            loss = self.criterion_cls(labels_pred, labels)

        return loss, masks_pred

    def postprocess_losses(self, losses, total_batches):
        res = []
        for l in losses:
            l = (l * self.accumulation_steps) / total_batches
            res.append(l)
        return res

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | : {start}")
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss, rn_loss_seg, rn_loss_cls, rn_loss_dice = 0.0, 0.0, 0.0, 0.0
        total_batches = len(dataloader)
        for itr, batch in enumerate(dataloader):  # replace `dataloader` with `tk0` for tqdm
            try:
                images, masks, labels = batch
            except ValueError:
                images, masks = batch
                labels = torch.randn(1)

            losses, outputs = self.forward(images, masks, labels)

            if self.mode == TrainerModes.combine:
                loss_cls, loss_seg, loss_dice = losses
                loss_cls /= self.accumulation_steps
                loss_seg /= self.accumulation_steps
                loss_dice /= self.accumulation_steps
                loss = loss_seg + loss_cls + 0.2 * loss_dice
            else:
                loss = losses
                loss /= self.accumulation_steps

            running_loss += loss.item()

            if self.mode == TrainerModes.combine:
                rn_loss_seg += loss_seg.item()
                rn_loss_cls += loss_cls.item()
                rn_loss_dice += loss_dice.item()
            if self.mode == TrainerModes.seg or self.mode == TrainerModes.combine:
                outputs = outputs.detach().cpu()
                meter.update(masks, outputs)

        epoch_loss_list = self.postprocess_losses((running_loss, rn_loss_seg, rn_loss_cls, rn_loss_dice), total_batches)
        epoch_loss = epoch_loss_list[0]
        dice, iou = self.logger.epoch_log(self.mode, phase, epoch, epoch_loss_list, meter)
        self.losses[phase].append(epoch_loss)
        if self.mode != TrainerModes.cls:
            self.dice_scores[phase].append(dice)
            self.iou_scores[phase].append(iou)
        torch.cuda.empty_cache()
        return epoch_loss_list, dice

    def start(self):
        with torch.no_grad():
            self.iterate(epoch=0, phase="val")


def prepare_and_visualize(image, mask):
    image = np.transpose(image, [1, 2, 0])
    mask = np.transpose(mask, [1, 2, 0])
    # print(image.shape, mask.shape)
    for i in range(4):
        visualize(image=image[:, :, 0], mask=mask[:, :, i])


def main(args):
    ckpt = None
    # model = torchvision.models.segmentation.fcn_resnet18(num_classes=4, pretrained=False, aux_loss=None, export_onnx=True)
    model = smp.Unet(args.backend, encoder_weights='imagenet', classes=args.num_classes, activation=None)
                    # aux_params={'classes': 4, 'dropout': 0.75})
    if os.path.isfile(args.model):
        ckpt = torch.load(args.model)
        # print("Loaded existing checkpoint!", f"Continue from epoch {ckpt['epoch']}", sep='\n')

    for mode in TrainerModes:
        if mode.value == args.mode:
            train_mode = mode
            break

    model_trainer = Trainer(model, ckpt, train_mode, args)
    model_trainer.start()


if __name__ == '__main__':
    set_start_method('spawn')
    main(parse_arguments_train(sys.argv[1:]))
