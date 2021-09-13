import os

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from albumentations import (HorizontalFlip, VerticalFlip, Normalize, RandomCrop, Compose,
                            RandomBrightnessContrast, Resize, MaskDropout, RandomSizedCrop,
                            CoarseDropout, OpticalDistortion)
from albumentations.pytorch import ToTensorV2


class DirtDataset(Dataset):
    CLASSES = ['clean', 'transparent', 'semi_transparent', 'opaque']

    def __init__(
            self,
            data_folder,
            phase,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            classes=('semi_transparent', 'opaque')
    ):
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.clean_classes = [self.CLASSES.index(cls.lower()) for cls in ['clean', 'transparent']]
        self.phase = phase
        self.images_fps, self.masks_fps = self._get_fps(data_folder)
        self.mean = mean
        self.std = std
        self.transforms = get_transforms(phase, mean, std)

    def _get_fps(self, data_folder):
        images_dir = os.path.join(data_folder, f'{self.phase}/images')
        masks_dir = os.path.join(data_folder, f'{self.phase}/indexes')
        image_names = sorted(os.listdir(images_dir))
        mask_names = sorted(os.listdir(masks_dir))
        images_fps = [os.path.join(images_dir, image_name) for image_name in image_names]
        masks_fps = [os.path.join(masks_dir, mask_name) for mask_name in mask_names]
        return images_fps, masks_fps

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i], cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        mask = cv2.imread(self.masks_fps[i], 0)
        # print(self.images_fps[i])

        # put 'transparent' and 'clean' into the same class
        clean_mask = [(mask == v) for v in self.clean_classes]
        clean_mask = np.logical_or.reduce(clean_mask)

        masks = [(mask == v) for v in self.class_values]
        # masks.append(clean_mask)
        mask = np.stack(masks, axis=-1).astype('float')

        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        mask = mask.permute(2, 0, 1)

        return image, mask

    def __len__(self):
        return len(self.images_fps)


def dirt_provider(
        data_folder,
        phase,
        mean=None,
        std=None,
        batch_size=8,
        num_workers=1,
):
    dataset = DirtDataset(data_folder, phase, mean=mean, std=std)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
    return dataloader


def get_transforms(phase, mean, std):
    list_transforms = list()
    # list_transforms.append(RandomSizedCrop((465, 930), 512, 1024, p=1, interpolation=0))
    # list_transforms.append(RandomCrop(448, 768, p=1))
    if phase == "train":
        list_transforms.extend(
            [
                RandomCrop(800, 800, p=1),
                # CoarseDropout(max_holes=5, max_width=70, max_height=70, min_width=30, min_height=30,
                #               mask_fill_value=3, p=0.5),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                # OpticalDistortion(distort_limit=0.5, shift_limit=1, interpolation=cv2.INTER_NEAREST,
                #                   border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1),
            ]
        )
    if phase == "val":
        list_transforms.extend(
            [
                Resize(1024, 1024, interpolation=0),
            ]
        )
    list_transforms.extend(
        [
            # Normalize(mean=mean, std=std, p=1),
            ToTensorV2(),
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms
