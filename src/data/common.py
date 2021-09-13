import os

import cv2
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import albumentations as albu
from albumentations import (HorizontalFlip, VerticalFlip, Normalize, RandomCrop, Compose,
                            RandomBrightnessContrast, Resize, MaskDropout, RandomSizedCrop,
                            CoarseDropout, OpticalDistortion)
from albumentations.pytorch import ToTensorV2


def get_transforms(phase, mean, std):
    list_transforms = list()
    # list_transforms.append(RandomSizedCrop((465, 930), 512, 1024, p=1, interpolation=0))
    # list_transforms.append(RandomCrop(448, 768, p=1))
    if phase == "train":
        list_transforms.extend(
            [
                RandomCrop(448, 800, p=1),
                CoarseDropout(max_holes=5, max_width=70, max_height=70, min_width=30, min_height=30,
                              mask_fill_value=0, p=0.5),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                OpticalDistortion(distort_limit=0.5, shift_limit=1, interpolation=cv2.INTER_NEAREST,
                                  border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1),
            ]
        )
    if phase == "val":
        list_transforms.extend(
            [
                Resize(1344, 2048, interpolation=0),
            ]
        )
    list_transforms.extend(
        [
            Normalize(mean=mean, std=std, p=1),
            ToTensorV2(),
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms


def get_file_paths(images_dir, masks_dir):

    def paste_mask_postfix(image_name):
        post = image_name[-4:]  # .png
        pre = image_name[:-4]
        return pre + '_color_mask' + post

    image_names = sorted(os.listdir(images_dir))
    mask_names = list(map(paste_mask_postfix, image_names))
    images_fps = [os.path.join(images_dir, image_name) for image_name in image_names]
    masks_fps = [os.path.join(masks_dir, mask_name) for mask_name in mask_names]
    return images_fps, masks_fps


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
