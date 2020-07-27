import os

import cv2
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from albumentations import (HorizontalFlip, VerticalFlip, Normalize, RandomCrop, Compose,
                            RandomBrightnessContrast, Resize, ImageOnlyTransform)
from albumentations.pytorch import ToTensor


def mask2rle(img):
    '''
    img: numpy array, 1 -> mask, 0 -> background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def make_mask(row_id, df):
    '''Given a row index, return image_id and mask (256, 1600, 4) from the dataframe `df`'''
    fname = df.iloc[row_id].name
    labels = df.iloc[row_id][:4]
    masks = np.zeros((256, 1600, 4), dtype=np.float32)  # float32 is V.Imp
    # 4:class 1～4 (ch:0～3)

    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(256 * 1600, dtype=np.uint8)
            for pos, le in zip(positions, length):
                mask[pos:(pos + le)] = 1
            masks[:, :, idx] = mask.reshape((256, 1600), order='F')
    return fname, masks


def make_mask_custom(row_id, df):
    '''Given a row index, return image_id and mask (256, 1600, 4) from the dataframe `df`'''
    fname = df['ImageId'][row_id]
    labels = df['EncodedPixels'][row_id]
    masks = np.zeros((4, 256, 1600), dtype=np.float32) # float32 is V.Imp
    # 4:class 1～4 (ch:0～3)
    idx = df['ClassId'][row_id] - 1

    if labels is not np.nan:
        label = labels.split(" ")
        positions = map(int, label[0::2])
        length = map(int, label[1::2])
        mask = np.zeros(256 * 1600, dtype=np.uint8)
        for pos, le in zip(positions, length):
            mask[pos:(pos + le)] = 1
        mask = mask.reshape((256, 1600), order='F')
        masks[idx, :, :] = mask

    return fname, masks


class SteelDataset(Dataset):
    def __init__(self, df, data_folder, mean, std, phase):
        self.df = df
        self.root = data_folder
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms(phase, mean, std)
        self.fnames = self.df.index.tolist()

    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        image_path = os.path.join(self.root, image_id)
        img = cv2.imread(image_path)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        # img = self._tensor_to_grayscale(img)
        mask = augmented['mask']  # 1x256x1600x4
        mask = mask[0].permute(2, 0, 1)  # 4x256x1600
        return img, mask

    def __len__(self):
        return len(self.fnames)

    @staticmethod
    def _tensor_to_grayscale(img_tensor):
        img_tensor = img_tensor[0, :, :]
        return img_tensor[np.newaxis, ...]


class SteelClassify(SteelDataset):
    def __getitem__(self, idx):
        img, mask = super(SteelDataset, self).__getitem__(idx)
        if np.count_nonzero(mask):
            label = torch.ones(1)
        else:
            label = torch.zeros(1)

        return img, label


def get_transforms(phase, mean, std):
    list_transforms = list()
    list_transforms.append(RandomCrop(256, 512, p=1))
    # list_transforms.append(Resize(256, 1024, p=1))
    if phase == "train":
        list_transforms.extend(
            [
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                RandomBrightnessContrast(p=0.5),
            ]
        )
    list_transforms.extend(
        [
            Normalize(mean=mean, std=std, p=1),
            ToTensor(),
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms


def provider(
        data_folder,
        df_path,
        phase,
        mean=None,
        std=None,
        batch_size=8,
        num_workers=4,
):
    '''Returns dataloader for the model training'''
    df = pd.read_csv(df_path)
    # https://www.kaggle.com/amanooo/defect-detection-starter-u-net
    df['ClassId'] = df['ClassId'].astype(int)
    df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
    df['defects'] = df.count(axis=1)

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["defects"], random_state=69)
    df = train_df if phase == "train" else val_df
    image_dataset = SteelDataset(df, data_folder, mean, std, phase)
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
    return dataloader


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


def shuffle_minibatch(inputs, masks, mixup=True):
    """Shuffle a minibatch and do linear interpolation between images and labels.
    Args:
        inputs: a numpy array of images with size batch_size x H x W x 3.
        targets: a numpy array of masks with size batch_size x 4 x H x W.
        mixup: a boolen as whether to do mixup or not. If mixup is True, we
            sample the weight from beta distribution using parameter alpha=1,
            beta=1. If mixup is False, we set the weight to be 1 and 0
            respectively for the randomly shuffled mini-batches.
    """
    batch_size = inputs.shape[0]
    rp1 = torch.randperm(batch_size)
    inputs1 = inputs[rp1]
    masks1 = masks[rp1]

    rp2 = torch.randperm(batch_size)
    inputs2 = inputs[rp2]
    masks2 = masks[rp2]

    if mixup is True:
        a = np.random.beta(0.4, 0.4, [batch_size, 1])
    else:
        a = np.ones((batch_size, 1))

    b = np.tile(a[..., None, None], [1, 3, 256, 512])
    c = np.tile(a[..., None, None], [1, 4, 256, 512])

    inputs1 = inputs1 * torch.from_numpy(b).float()
    inputs2 = inputs2 * torch.from_numpy(1 - b).float()

    masks1 = masks1 * torch.from_numpy(c).float()
    masks2 = masks2 * torch.from_numpy(1-c).float()

    inputs_shuffle = inputs1 + inputs2
    targets_shuffle = masks1 + masks2

    return inputs_shuffle, targets_shuffle

