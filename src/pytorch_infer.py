import warnings
warnings.filterwarnings('ignore')
import argparse
import enum
import os
import sys

import numpy as np
import albumentations as A
from tqdm import tqdm
import pandas as pd
from prettytable import PrettyTable

import torch
import torch.onnx
from torch.utils.data import DataLoader
from torch.jit import load

from mlcomp.contrib.transform.albumentations import ChannelTranspose
from mlcomp.contrib.dataset.classify import ImageDataset
from mlcomp.contrib.transform.rle import mask2rle
from mlcomp.contrib.transform.tta import TtaWrap

from data.common import make_mask_custom
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
    se_resnext50 = 'unet_se_resnext50'
    resnet34 = 'unet_resnet34'
    mobilenet2 = 'unet_mobilenet2'
    effnetb7 = 'effnetb7'
    effnetb0 = 'effnetb0'


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


class ModelInferenceHandler:
    def __init__(self, mask_df):
        self.mask_df = mask_df
        self.thresholds = [0.5, 0.5, 0.5, 0.5]
        self.min_area = [600, 600, 1000, 2000]
        self.res = []
        self.dice_score_list = []
        self.dice_score_list_ch = []
        self.iou_score_list = []

    @staticmethod
    def _tta_mean(preds):
        preds = torch.stack(preds)
        preds = torch.mean(preds, dim=0)
        return preds.detach().cpu().numpy()

    def _make_predicition(self, labels, file, p, p_img):
        for i in range(4):
            p_channel = np.zeros((256, 1600), dtype=np.uint8)
            imageid_classid = file + '_' + str(i + 1)
            if (labels is None) or (labels[i] > 0):
                p_channel = p[i]
                p_channel = (p_channel > self.thresholds[i]).astype(np.uint8)
                if p_channel.sum() < self.min_area[i]:
                    p_channel = np.zeros(p_channel.shape, dtype=p_channel.dtype)
            # Save channel to obtain all channels for metrics calculation
            p_img.append(p_channel)
            self.res.append({
                'ImageId_ClassId': imageid_classid,
                'EncodedPixels': mask2rle(p_channel)
            })

    def _batch_processing(self, image_file, j, labels, preds):
        for p, file in zip(preds, image_file):
            file = os.path.basename(file)
            # Image postprocessing
            p_img = []
            _, target = make_mask_custom(j, self.mask_df)
            target = torch.tensor(target[np.newaxis, :])
            self._make_predicition(labels, file, p, p_img)

            p_img = np.array(p_img)
            # Make it seem like a batch of 1
            p_img = torch.tensor(p_img[np.newaxis, :])
            self._update_metrics(p_img, target)

    def _update_metrics(self, p_img, target):
        self.dice_score_list_ch.append(dice_single_channel(p_img[0, 2], target[0, 2], 0.5).numpy())
        self.dice_score_list.append(dice_channel_torch(p_img, target, 0.5).numpy())
        self.iou_score_list.append(compute_iou_batch(p_img, target, classes=[1]))

    def _infer_net(self, datasets, loaders, model, total):
        with torch.no_grad():
            for j, loaders_batch in enumerate(tqdm(zip(*loaders), total=total)):
                preds, image_file = [], []
                labels = None
                for i, batch in enumerate(loaders_batch):
                    features = batch['features'].cuda()
                    output = model(features)
                    if type(output) is tuple:
                        output, labels = output
                    p = torch.sigmoid(output)
                    # inverse operations for TTA
                    p = datasets[i].inverse(p)
                    preds.append(p)
                    image_file = batch['image_file']

                # TTA mean
                preds = self._tta_mean(preds)
                if labels is not None:
                    # labels = self._tta_mean(labels)
                    labels = labels[0]

                # Batch post processing
                self._batch_processing(image_file, j, labels, preds)

    def calculate_dice_and_iou(self, model, datasets, loaders):
        self.res, self.dice_score_list, self.dice_score_list_ch, self.iou_score_list = [], [], [], []
        total = len(datasets[0]) // 1  # Todo: Add batch size
        self._infer_net(datasets, loaders, model, total)
        dice = np.mean(self.dice_score_list)
        dice_ch = np.mean(self.dice_score_list_ch)
        iou = np.nanmean(self.iou_score_list)
        return dice, dice_ch, iou, self.res


def write_prediction(res, imageid_classid, p_channel):
    res.append({'ImageId_ClassId': imageid_classid,
                'EncodedPixels': mask2rle(p_channel)})


def make_zeroed_prediction(res, file):
    p_img = []
    for i in range(4):
        imageid_classid = file + '_' + str(i + 1)
        p_channel = np.zeros((256, 1600), dtype=np.uint8)
        p_img.append(p_channel)
        write_prediction(res, imageid_classid, p_channel)
    p_img = np.array(p_img)
    # Make it seem like a batch of 1
    p_img = torch.tensor(p_img[np.newaxis, :])
    return p_img


def calculate_dice_and_iou_ensemble(model, cls, datasets, loaders, mask_df, thresholds, min_area):
    res, dice_score_list, dice_score_list_ch, iou_score_list = [], [], [], []
    total = len(datasets[0]) // 1
    with torch.no_grad():
        for i, batch in enumerate(tqdm(zip(*loaders), total=total)):
            batch = batch[0]
            features = batch['features'].cuda()
            pred_aux, label = cls(features)
            label = label[0].cpu().numpy()
            file = os.path.basename(batch['image_file'][0])

            _, target = make_mask_custom(i, mask_df)
            target = torch.tensor(target[np.newaxis, :])

            if not label[label > 0]:  # Check for any defect
                p_img = make_zeroed_prediction(res, file)
                dice_score_list_ch.append(dice_single_channel(p_img[0, 2], target[0, 2], 0.5).numpy())
                dice_score_list.append(dice_channel_torch(p_img, target, 0.5).numpy())
                iou_score_list.append(compute_iou_batch(p_img, target, classes=[1]))
                continue

            pred_raw, _ = model(features)  # Only now we feed data to big network
            p = torch.sigmoid(pred_raw[0]).cpu().numpy()
            p_aux = torch.sigmoid(pred_aux[0]).cpu().numpy()

            # Image postprocessing
            p_img = []
            for j in range(4):
                p_channel = np.zeros((256, 1600), dtype=np.uint8)
                imageid_classid = file + '_' + str(j + 1)

                if label[j] > 0:  # Now we are checking for speciefic defects
                    p_channel = p[j]
                    p_channel = (p_channel > thresholds[j]).astype(np.uint8)
                    if p_channel.sum() < min_area[j]:
                        p_channel = np.zeros(p_channel.shape, dtype=p_channel.dtype)
                    else:  # We use effnetb0 mask only if effnetb7 finds defect. We are basically refining effnetb7
                        p_channel = (p[j] + p_aux[j]) / 2  # Take mean
                        p_channel = (p_channel > thresholds[j]).astype(np.uint8)
                        if p_channel.sum() < min_area[j]:
                            p_channel = np.zeros(p_channel.shape, dtype=p_channel.dtype)

                p_img.append(p_channel)
                write_prediction(res, imageid_classid, p_channel)
            p_img = np.array(p_img, dtype=np.uint8)
            # Make it seem like a batch of 1
            p_img = torch.tensor(p_img[np.newaxis, :])

            dice_score_list_ch.append(dice_single_channel(p_img[0, 2], target[0, 2], 0.5).numpy())
            dice_score_list.append(dice_channel_torch(p_img, target, 0.5).numpy())
            iou_score_list.append(compute_iou_batch(p_img, target, classes=[1]))

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
    effnetb0 = load('../ckpt/effnetb0_final_stage/traced_model.pth').cuda()
    effnetb7 = load('../ckpt/traced_effnetb7_mixup_retrain.pth').cuda()
    cls = load('../ckpt/traced_effnetb0_classifier_maskdrop.pth').cuda()

    models_list = [unet_se_resnext50_32x4d, unet_resnet34, unet_mobilenet2, eff_net_v2, effnetb0]
    models_dict = {}
    for i, model_name in enumerate(ModelNames):
        models_list[i].eval()
        models_dict[model_name.value] = models_list[i]

    # Different transforms for TTA wrapper
    transforms = [
        [],
        # [A.HorizontalFlip(p=1)],
        # [A.VerticalFlip(p=1)],
    ]

    transforms = [create_transforms(t) for t in transforms]
    datasets = [TtaWrap(ImageDataset(img_folder=img_folder, transforms=t), tfms=t) for t in transforms]
    loaders = [DataLoader(d, num_workers=num_workers, batch_size=batch_size, shuffle=False) for d in datasets]

    table = PrettyTable()
    table.field_names = ['Model', 'dice', 'iou']

    logger = MetricsLogger.config_logger('metric_scores', '../logs/scores.log')
    inference_handler = ModelInferenceHandler(mask_df)

    dice_dict, iou_dict = {}, {}
    for key in models_dict.keys():
        dice, dice_ch, iou, res = \
            inference_handler.calculate_dice_and_iou(models_dict[key], datasets, loaders)
        print(dice, iou)
        table.add_row([key, dice, iou])
        dice_dict[key] = dice
        iou_dict[key] = iou

    dice, dice_ch, iou, res = calculate_dice_and_iou_ensemble(effnetb7, cls, datasets, loaders, mask_df, thresholds,
                                                              min_area)
    table.add_row(['effnetb7+effnetb0', dice, iou])
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