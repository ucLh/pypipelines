import warnings
warnings.filterwarnings('ignore')
import argparse
import enum
import os
import sys


import numpy as np
import cv2
import albumentations as A
from tqdm import tqdm
import pandas as pd
from prettytable import PrettyTable

import torch
import torch.nn as nn
import torch.onnx
from torch.utils.data import DataLoader
from torch.jit import load

from mlcomp.contrib.transform.albumentations import ChannelTranspose
from mlcomp.contrib.dataset.classify import ImageDataset
from mlcomp.contrib.transform.rle import mask2rle
from mlcomp.contrib.transform.tta import TtaWrap

from data import make_mask_custom
from util import compute_iou_batch, dice_channel_torch, dice_single_channel, MetricsLogger


class Model:
    def __init__(self, models):
        self.models = models

    def __call__(self, x):
        res = []
        x = x.cuda()
        with torch.no_grad():
            for m in self.models:
                res.append(m(x))
        res = torch.stack(res)
        return torch.mean(res, dim=0)


class ModelNames(enum.Enum):
    eff_net_v2 = 'effnetb7_mixup_v2'
    eff_net = 'effnetb7_mixup'
    se_resnext50 = 'unet_se_resnext50'
    resnet34 = 'unet_resnet34'
    mobilenet2 = 'unet_mobilenet2'


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


def calculate_dice_and_iou(model, datasets, loaders, mask_df, thresholds, min_area):
    def tta_mean(preds, labels=None):
        preds = torch.stack(preds)
        preds = torch.mean(preds, dim=0)
        if labels is not None:
            labels = torch.stack(labels)
            labels = torch.mean(labels, dim=0)
            labels = labels.detach().cpu().numpy()  # has shape (1, 4)
            labels = labels[0]

        return preds.detach().cpu().numpy(), labels

    def batch_postprocessing():
        for p, file in zip(preds, image_file):
            file = os.path.basename(file)
            # Image postprocessing
            p_img = []
            _, target = make_mask_custom(j, mask_df)
            target = torch.tensor(target[np.newaxis, :])
            for i in range(4):
                p_channel = np.zeros((256, 1600), dtype=np.uint8)
                imageid_classid = file + '_' + str(i + 1)
                if (labels is None) or (labels[i] > 0):
                    p_channel = p[i]
                    p_channel = (p_channel > thresholds[i]).astype(np.uint8)
                    if p_channel.sum() < min_area[i]:
                        p_channel = np.zeros(p_channel.shape, dtype=p_channel.dtype)
                # Save channel to obtain all channels for metrics calculation
                p_img.append(p_channel)
                res.append({
                    'ImageId_ClassId': imageid_classid,
                    'EncodedPixels': mask2rle(p_channel)
                })

            p_img = np.array(p_img)
            # Make it seem like a batch of 1
            p_img = torch.tensor(p_img[np.newaxis, :])

            dice_score_list_ch.append(dice_single_channel(p_img[0, 2], target[0, 2], 0.5).numpy())
            dice_score_list.append(dice_channel_torch(p_img, target, 0.5).numpy())
            iou_score_list.append(compute_iou_batch(p_img, target, classes=[1]))

    res, dice_score_list, dice_score_list_ch, iou_score_list = [], [], [], []
    total = len(datasets[0]) // 1  # Todo: Add batch size
    with torch.no_grad():
        for j, loaders_batch in enumerate(tqdm(zip(*loaders), total=total)):
            preds = []
            image_file = []
            for i, batch in enumerate(loaders_batch):
                features = batch['features'].cuda()
                output = model(features)
                if type(output) is tuple:
                    output, label = output
                p = torch.sigmoid(output)
                # inverse operations for TTA
                p = datasets[i].inverse(p)
                preds.append(p)
                image_file = batch['image_file']

            # TTA mean
            preds, labels = tta_mean(preds)

            # Batch post processing
            batch_postprocessing()

    dice = np.mean(dice_score_list)
    dice_ch = np.mean(dice_score_list_ch)
    iou = np.nanmean(iou_score_list)
    return dice, dice_ch, iou, res


def main(args):
    img_folder = args.img_folder
    mask_df = pd.read_csv(args.mask_df)
    batch_size = 1
    num_workers = 0
    thresholds = [0.5, 0.5, 0.5, 0.5]
    min_area = [600, 600, 1000, 2000]

    unet_se_resnext50_32x4d = \
        load('../data/severstalmodels/se_resnext50_32x4d.pth').cuda()
    unet_mobilenet2 = load('../data/severstalmodels/unet_mobilenet2.pth').cuda()
    unet_resnet34 = load('../data/severstalmodels/unet_resnet34.pth').cuda()
    eff_net_v2 = load('../ckpt/traced_effnetb7_1024_mixup_v2.pth').cuda()
    eff_net = load('../ckpt/effnetb0_final_stage/traced_effnetb0_averaged.pth').cuda()

    models_list = [eff_net_v2, eff_net, unet_se_resnext50_32x4d, unet_resnet34, unet_mobilenet2]
    models_dict = {}
    for i, model_name in enumerate(ModelNames):
        models_list[i].eval()
        models_dict[model_name.value] = models_list[i]

    # Different transforms for TTA wrapper
    transforms = [
        [],
        [A.HorizontalFlip(p=1)],
        # [A.VerticalFlip(p=1)],
    ]

    transforms = [create_transforms(t) for t in transforms]
    datasets = [TtaWrap(ImageDataset(img_folder=img_folder, transforms=t), tfms=t) for t in transforms]
    loaders = [DataLoader(d, num_workers=num_workers, batch_size=batch_size, shuffle=False) for d in datasets]

    table = PrettyTable()
    table.field_names = ['Model', 'dice', 'iou']

    logger = MetricsLogger.config_logger('metric_scores', '../logs/scores.log')

    dice_dict, iou_dict = {}, {}
    for key in models_dict.keys():
        dice, dice_ch, iou, res = calculate_dice_and_iou(models_dict[key], datasets, loaders, mask_df, thresholds, min_area)
        print(dice, iou)
        table.add_row([key, dice, iou])
        dice_dict[key] = dice
        iou_dict[key] = iou

    logger.info(table)

    df = pd.DataFrame(res)
    df = df.fillna('')
    df.to_csv('submission.csv', index=False)

    df['Image'] = df['ImageId_ClassId'].map(lambda x: x.split('_')[0])
    df['Class'] = df['ImageId_ClassId'].map(lambda x: x.split('_')[1])
    df['empty'] = df['EncodedPixels'].map(lambda x: not x)
    classes = df[df['empty'] == False]['Class'].value_counts()
    print(classes)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_folder', type=str,
                        help='Path to (.pth) file',
                        default='../data/cropped/')
    parser.add_argument('--mask_df', type=str,
                        help='Model backend',
                        default='../data/cropped.csv')
    parser.add_argument('--log_dir', type=str,
                        help='Directory where to write event logs.',
                        default='../testing_results')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))