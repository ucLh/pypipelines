from __future__ import annotations
from abc import ABC, abstractmethod
import argparse
import os
import sys
import time

from apex import amp
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from torch.multiprocessing import set_start_method
# from torchtools.optim import RangerLars

from data import provider, visualize, shuffle_minibatch
from util import Meter, MetricsLogger, DiceLoss, TrainerModes, set_parameter_requires_grad


class Trainer(object):
    '''This class takes care of training and validation of our model'''
    def __init__(self, model, checkpoint, mode):
        self.num_workers = 6
        self.batch_size = {"train": 16, "val": 4}
        self.accumulation_steps = 32 // self.batch_size['train']
        self.lr = 1e-6
        self.num_epochs = 160
        self.start_epoch = 0
        self.best_dice = 0
        self.best_loss = 1e6
        self.best_seg_loss = 1e6
        self.phases = ["train", "val"]
        self.mode = mode
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        self.net = self.net.to(self.device)
        self.freeze_backbone_if_needed()
        self.net, self.optimizer = amp.initialize(model, optim.Adam(self.net.parameters(), lr=self.lr), opt_level="O1")
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=5, eta_min=1e-9)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.num_epochs)
        if checkpoint is not None:
            self.net.load_state_dict(checkpoint["state_dict"])
            # self.optimizer.load_state_dict(checkpoint["optimizer"])
            # self.try_to_load(self.scheduler, checkpoint, "scheduler")
            amp.load_state_dict(checkpoint["amp"])
            self.start_epoch = checkpoint["epoch"]
            self.best_dice = checkpoint["best_dice"]
            # self.best_loss = self.try_to_assign(checkpoint, "best_loss", self.best_loss)
            self.best_seg_loss = self.try_to_assign(checkpoint, "best_seg_loss", self.best_seg_loss)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.criterion_cls = torch.nn.BCEWithLogitsLoss()
        # self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=3, verbose=True, factor=0.2)
        cudnn.benchmark = False
        self.dataloaders = {
            phase: provider(
                data_folder='../data/Severstal/train_test_images/',
                df_path='../data/Severstal/train_test.csv',
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
        self.tensorboard_writer = SummaryWriter("../logs/runs")

    def freeze_backbone_if_needed(self):
        if self.mode == TrainerModes.cls:
            set_parameter_requires_grad(self.net.encoder)
            set_parameter_requires_grad(self.net.decoder)
            set_parameter_requires_grad(self.net.segmentation_head)

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
            loss = (loss_cls, loss_seg, loss_dice)
        elif self.mode == TrainerModes.seg:
            masks_pred, labels_pred = self.net(images)
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
        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader):  # replace `dataloader` with `tk0` for tqdm
            images, masks, labels = batch
            # if phase == "train":
            #     images, masks = shuffle_minibatch(images, masks)
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

            if phase == "train":
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
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
        self.tensorboard_writer.add_scalar('Loss/' + phase, epoch_loss, epoch)
        self.losses[phase].append(epoch_loss)
        if self.mode != TrainerModes.cls:
            self.tensorboard_writer.add_scalar('Dice/' + phase, dice, epoch)
            self.tensorboard_writer.add_scalar('Iou/' + phase, iou, epoch)
            self.dice_scores[phase].append(dice)
            self.iou_scores[phase].append(iou)
        print('Learning rate: ', self.optimizer.param_groups[0]['lr'])
        torch.cuda.empty_cache()
        return epoch_loss_list, dice

    def start(self):
        # self.tensorboard_writer.add_graph(self.net, self.dataloaders["train"].dataset[0])
        for epoch in range(self.start_epoch, self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_dice": self.best_dice,
                "best_loss": self.best_loss,
                "best_seg_loss": self.best_seg_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "amp": amp.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            }
            with torch.no_grad():
                losses, val_dice = self.iterate(epoch, "val")
                val_loss = losses[0]
                self.scheduler.step(val_loss)
            if val_loss <= self.best_loss:
                print("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state, "./model.pth")
            elif self.mode == "combine":
                if losses[1] <= self.best_seg_loss:
                    print("******** New suboptimal found, saving state ********")
                    state["best_seg_loss"] = self.best_seg_loss = losses[1]
                    torch.save(state, "./model_seg.pth")
            if epoch == self.num_epochs - 1:
                print("******** Saving last state ********")
                torch.save(state, "./last_model.pth")
            print()

    @staticmethod
    def try_to_load(field_to_store, checkpoint, key):
        try:
            field_to_store.load_state_dict(checkpoint[key])
        except:
            print(f"No '{key}' key in checkpoint!")

    @staticmethod
    def try_to_assign(checkpoint, key, default_val=None, verbose=True):
        try:
            if verbose:
                print(f"Key '{key}' is {checkpoint[key]}")
            return checkpoint[key]
        except:
            print(f"No '{key}' key in checkpoint!")
            return default_val


def prepare_and_visualize(image, mask):
    image = np.transpose(image, [1, 2, 0])
    mask = np.transpose(mask, [1, 2, 0])
    # print(image.shape, mask.shape)
    for i in range(4):
        visualize(image=image[:, :, 0], mask=mask[:, :, i])


def main(args):
    ckpt = None
    if os.path.isfile(args.model):
        model = smp.FPN(args.backend, encoder_weights=None, classes=4, activation=None,
                         aux_params={'classes': 4, 'dropout': 0.75})
        ckpt = torch.load(args.model)
        print("Loaded existing checkpoint!", f"Continue from epoch {ckpt['epoch']}", sep='\n')
    else:
        model = smp.FPN(args.backend, encoder_weights='imagenet', classes=4, activation=None,
                        aux_params={'classes': 4, 'dropout': 0.75})

    for mode in TrainerModes:
        if mode.value == args.mode:
            train_mode = mode
            break

    model_trainer = Trainer(model, ckpt, train_mode)
    # Visualization check
    # for i, batch in enumerate(model_trainer.dataloaders["train"]):
    #     print('pass')
    #     break
    # image_raw, mask_raw = batch
    #
    # image, mask = image_raw[0], mask_raw[0]
    # prepare_and_visualize(image, mask)
    #
    # image, mask = image_raw[1], mask_raw[1]
    # prepare_and_visualize(image, mask)
    #
    # image, mask = shuffle_minibatch(image_raw, mask_raw)
    # image, mask = image[0], mask[0]
    # prepare_and_visualize(image, mask)
    #
    # image, mask = shuffle_minibatch(image_raw, mask_raw)
    # image, mask = image[1], mask[1]
    # prepare_and_visualize(image, mask)
    # exit(1)

    model_trainer.start()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # parser.add_argument('--data_root', type=str,
    #                     help='Path to data directory which needs to be forward passed through the network',
    #                     default='../datasets/queries')
    # parser.add_argument('--df_root', type=str,
    #                     help='Path to data directory which needs to be forward passed through the network',
    #                     default='../datasets/queries')
    parser.add_argument('--mode', type=str,
                        help='Training mode. One of "seg", "cls" or "combine"',
                        default='seg')
    parser.add_argument('--model', type=str,
                        help='Path to (.pth) file',
                        default='../ckpt/effnetb0_fpn_dice_v2.pth')
    parser.add_argument('--backend', type=str,
                        help='Model backend',
                        default='efficientnet-b0')
    parser.add_argument('--log_dir', type=str,
                        help='Directory where to write event logs.',
                        default='../testing_results')
    return parser.parse_args(argv)


if __name__ == '__main__':
    set_start_method('spawn')
    main(parse_arguments(sys.argv[1:]))
