import argparse
import os
import sys
import time

import albumentations as A
from mlcomp.contrib.transform.albumentations import ChannelTranspose
from mlcomp.contrib.dataset.classify import ImageDataset
# from mlcomp.contrib.transform.rle import rle2mask, mask2rle
from mlcomp.contrib.transform.tta import TtaWrap
import torch
import torch.backends.cudnn as cudnn
from torch.jit import load
from torch.utils.data import DataLoader

from data import provider, visualize
from util import Meter, MetricsLogger


def create_transforms(additional):
    res = list(additional)
    # add necessary transformations
    res.extend([
        A.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ),
        ChannelTranspose()
    ])
    res = A.Compose(res)
    return res


class Tester(object):
    '''This class takes care of training and validation of our model'''

    def __init__(self, model):
        self.num_workers = 6
        self.batch_size = {"val": 1}
        self.accumulation_steps = 32 // self.batch_size['val']
        self.best_dice = 0
        self.phases = ["val"]
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        # self.net = self.net.to(self.device)

        self.criterion = torch.nn.BCEWithLogitsLoss()

        cudnn.benchmark = True
        self.dataloaders = {
            phase: provider(
                data_folder='../data/cropped/',
                df_path='../data/cropped.csv',
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

    def forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)
        return loss, outputs

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]

        transforms = [
            [],
            [A.HorizontalFlip(p=1)]
        ]

        transforms = [create_transforms(t) for t in transforms]
        datasets = [TtaWrap(ImageDataset(img_folder='../data/cropped/', transforms=t), tfms=t) for t in transforms]
        loaders = [DataLoader(d, num_workers=self.num_workers,
                              batch_size=self.batch_size["val"],
                              shuffle=False) for d in datasets]

        running_loss = 0.0
        total_batches = len(dataloader)
        #         tk0 = tqdm(dataloader, total=total_batches)
        for itr, batch in enumerate(dataloader):  # replace `dataloader` with `tk0` for tqdm
            # original_batch = batch[-1]
            # loaders_batch = batch[1:]
            images, targets = batch
            print(dataloader.dataset.fnames[itr])
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.update(targets, outputs)
        #             tk0.set_postfix(loss=(running_loss / ((itr + 1))))
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice, iou = self.logger.epoch_log(phase, epoch, epoch_loss, meter, start)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        torch.cuda.empty_cache()
        return epoch_loss, dice

    def start(self):
        # self.tensorboard_writer.add_graph(self.net, self.dataloaders["train"].dataset[0])
        with torch.no_grad():
            val_loss, val_dice = self.iterate(0, "val")


def main(args):
    model = load(args.model).cuda()
    model_tester = Tester(model)
    # # Visualization check
    # image, mask = model_trainer.dataloaders["train"].dataset[3]
    # image = np.transpose(image, [1, 2, 0])
    # mask = np.transpose(mask, [1, 2, 0])
    # print(image.shape, mask.shape)
    # for i in range(4):
    #     visualize(image=image[:, :, 0], mask=mask[:, :, i])
    # exit(1)

    model_tester.start()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # parser.add_argument('--data_root', type=str,
    #                     help='Path to data directory which needs to be forward passed through the network',
    #                     default='../datasets/queries')
    # parser.add_argument('--df_root', type=str,
    #                     help='Path to data directory which needs to be forward passed through the network',
    #                     default='../datasets/queries')
    parser.add_argument('--model', type=str,
                        help='Path to (.pth) file',
                        # default='../ckpt/traced_effnetb7_1024_best.pth')
                        default='../data/severstalmodels/se_resnext50_32x4d.pth')
    parser.add_argument('--log_dir', type=str,
                        help='Directory where to write event logs.',
                        default='../testing_results')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
