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
# from torchtools.optim import RangerLars

from data import provider, visualize, shuffle_minibatch_onehot
from models import ClassifierNew
from util import Meter, MetricsLogger, set_parameter_requires_grad


class Trainer(object):
    '''This class takes care of training and validation of our model'''
    def __init__(self, encoder, model, checkpont):
        self.num_workers = 6
        self.batch_size = {"train": 32, "val": 4}
        self.accumulation_steps = 32 // self.batch_size['train']
        self.lr = 8e-5
        self.num_epochs = 200
        self.start_epoch = 0
        self.best_dice = 0
        self.best_loss = 1e6
        self.phases = ["train", "val"]
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        self.net = self.net.to(self.device)
        self.encoder = encoder
        self.encoder = self.encoder.to(self.device)
        set_parameter_requires_grad(self.encoder)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=5, eta_min=1e-9)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.num_epochs)
        if checkpont is not None:
            self.net.load_state_dict(checkpont["state_dict"])
            # self.optimizer.load_state_dict(checkpont["optimizer"])
            try:
                self.scheduler.load_state_dict(checkpont["scheduler"])
            except:
                print("No scheduler")
                pass
            self.start_epoch = checkpont["epoch"]
            self.best_dice = checkpont["best_dice"]
            # try:
            #     self.best_loss = checkpont["best_loss"]
            #     print('***Best loss*** ', self.best_loss)
            # except:
            #     print("No best_loss")
            #     pass
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        # self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=3, verbose=True, factor=0.2)
        cudnn.benchmark = False
        self.dataloaders = {
            phase: provider(
                data_folder='../data/Severstal/train_images/',
                df_path='../data/Severstal/train.csv',
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

    def forward(self, images, targets, net, criterion):
        images = images.to(self.device)
        masks = targets.to(self.device)
        features = self.encoder(images)
        outputs = net(features[-1])
        # print(outputs[-1].view(outputs[-1].size(0), -1).shape)
        masks.squeeze_()
        loss = criterion(outputs, masks)
        return loss, outputs

    def iterate(self, epoch, phase, net):
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
        net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        #         tk0 = tqdm(dataloader, total=total_batches)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader):  # replace `dataloader` with `tk0` for tqdm
            images, targets = batch
            # if phase == "train":
            #     images, targets = shuffle_minibatch_onehot(images.cpu(), targets.cpu())
            # else:
            #     images, targets = shuffle_minibatch_onehot(images.cpu(), targets.cpu(), mixup=False)
            loss, outputs = self.forward(images.cuda(), targets.cuda(), self.net, self.criterion_cls)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
        #             tk0.set_postfix(loss=(running_loss / ((itr + 1))))
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        self.logger.cls_epoch_log(phase, epoch, epoch_loss)
        self.tensorboard_writer.add_scalar('Loss/' + phase, epoch_loss, epoch)
        self.losses[phase].append(epoch_loss)
        print('Learning rate: ', self.optimizer.param_groups[0]['lr'])
        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.iterate(epoch, "train", self.net)
            state = {
                "epoch": epoch,
                "best_dice": self.best_dice,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            }
            with torch.no_grad():
                val_loss = self.iterate(epoch, "val", self.net)
                self.scheduler.step(val_loss)
            if val_loss <= self.best_loss:
                print("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state, "./model.pth")
            elif epoch == self.num_epochs - 1:
                print("******** Saving last state ********")
                torch.save(state, "./last_model.pth")
            print()


def prepare_and_visualize(image, mask):
    image = np.transpose(image, [1, 2, 0])
    mask = np.transpose(mask, [1, 2, 0])
    for i in range(4):
        visualize(image=image[:, :, 0], mask=mask[:, :, i])


def main(args):
    model_seg = smp.Unet(args.backend, encoder_weights=None, classes=4, activation=None)
    ckpt_seg = torch.load("../ckpt/effnetb7_1024_mixup_v2.pth")
    model_seg.load_state_dict(ckpt_seg["state_dict"])
    ckpt = None
    # ckpt = torch.load(args.model)
    # print("Loaded existing checkpoint!", f"Continue from epoch {ckpt['epoch']}", sep='\n')

    model_cls = ClassifierNew()
    model_trainer = Trainer(model_seg.encoder, model_cls, ckpt)
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

    parser.add_argument('--model', type=str,
                        help='Path to (.pth) file',
                        default='model.pth')
    parser.add_argument('--backend', type=str,
                        help='Model backend',
                        default='efficientnet-b7')
    parser.add_argument('--log_dir', type=str,
                        help='Directory where to write event logs.',
                        default='../testing_results')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
