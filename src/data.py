from enum import Enum
import os

import cv2
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from albumentations import (HorizontalFlip, VerticalFlip, Normalize, RandomCrop, Compose,
                            RandomBrightnessContrast, Resize, MaskDropout, RandomSizedCrop,
                            CoarseDropout, OpticalDistortion)
from albumentations.pytorch import ToTensor


class DatasetTypes(Enum):
    Seg_dataset = 'SteelDataset'
    Cls_dataset = 'ClassifyDataset'


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
        if isinstance(label, str):
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(256 * 1600, dtype=np.uint8)
            for pos, le in zip(positions, length):
                mask[pos:(pos + le)] = 1
            masks[:, :, idx] = mask.reshape((256, 1600), order='F')
    return fname, masks


def make_label(mask):
    label = np.zeros(4)
    for i in range(4):
        if np.count_nonzero(mask[i, :, :]):
            label[i] = 1

    return torch.from_numpy(label)


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

    @staticmethod
    def _mask_dropout(img, mask):
        masks = []
        md_transform = Compose([MaskDropout(max_objects=1, p=0.5), ToTensor()])
        for i in range(4):
            img = img.numpy()
            img = np.transpose(img, [1, 2, 0])
            augmented = md_transform(image=img, mask=mask[i].numpy())
            img, mask_aug = augmented['image'], augmented['mask']
            masks.append(mask_aug[0])
        masks = torch.stack(masks)
        return img, masks


class GolfDataset(Dataset):
    CLASSES = ['unlabeled', 'sky', 'sand', 'ground',
               'building', 'poo', 'ball', 'rock_stone',
               'tree_bush', 'fairway_grass', 'raw_grass', 'hole',
               'water', 'person', 'animal', 'vehicle', 'green_grass']

    def __init__(
            self,
            images_fps,
            masks_fps,
            mean,
            std,
            phase,
            transorfms_func,
            classes=('unlabeled', 'sky', 'sand', 'ground', 'tree_bush', 'raw_grass',
                     'person', 'animal', 'vehicle',)
    ):
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.grass_classes = [self.CLASSES.index(cls.lower()) for cls in ['fairway_grass', 'green_grass']]
        self.garbage_classes = [self.CLASSES.index(cls.lower()) for cls in ('building', 'poo', 'ball', 'rock_stone',
                                                                            'hole', 'water',)]
        self.images_fps = images_fps
        self.masks_fps = masks_fps
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = transorfms_func(phase, mean, std)

    @staticmethod
    def _tensor_to_grayscale(img_tensor):
        # img_tensor = img_tensor[0, :, :]
        img_tensor = (img_tensor - img_tensor.min()) / (img_tensor - img_tensor.max())
        return img_tensor[np.newaxis, ...]

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i], cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        # print(self.images_fps[i])

        # unite two types of grass in one class
        grass_masks = [(mask == v) for v in self.grass_classes]
        grass_mask = np.logical_or.reduce(grass_masks)

        # unite all auxiliary classes in one garbage class
        garbage_masks = [(mask == v) for v in self.garbage_classes]
        garbage_mask = np.logical_or.reduce(garbage_masks)

        masks = [(mask == v) for v in self.class_values]
        masks.append(grass_mask)
        masks.append(garbage_mask)
        mask = np.stack(masks, axis=-1).astype('float')

        augmented = self.transforms(image=image, mask=mask)
        image = augmented['image']
        # image = self._tensor_to_grayscale(image)
        mask = augmented['mask']
        mask = mask[0].permute(2, 0, 1)

        return image, mask

    def __len__(self):
        return len(self.images_fps)


class SteelClassify(SteelDataset):
    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        image_path = os.path.join(self.root, image_id)
        img_raw = cv2.imread(image_path)
        augmented = self.transforms(image=img_raw, mask=mask)
        img = augmented['image']
        # img = self._tensor_to_grayscale(img)
        mask = augmented['mask']  # 1x256x1600x4
        mask = mask[0].permute(2, 0, 1)  # 4x256x1600

        # img, mask = self._mask_dropout(img, mask)
        label = make_label(mask)

        return img, mask, label


def get_transforms(phase, mean, std):
    list_transforms = list()
    # list_transforms.append(RandomSizedCrop((465, 930), 512, 1024, p=1, interpolation=0))
    # list_transforms.append(RandomCrop(448, 768, p=1))
    if phase == "train":
        list_transforms.extend(
            [
                RandomCrop(448, 930, p=1),
                CoarseDropout(max_holes=5, max_width=70, max_height=70, min_width=30, min_height=30,
                              mask_fill_value=0, p=0.5),
                HorizontalFlip(p=0.5),
                # VerticalFlip(p=0.5),
                RandomBrightnessContrast(p=0.5),
                OpticalDistortion(distort_limit=0.5, shift_limit=1, interpolation=cv2.INTER_NEAREST,
                                  border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1),
            ]
        )
    if phase == "val":
        list_transforms.extend(
            [
                Resize(640, 1280, interpolation=0),
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


def non_df_provider(data_folder,
                    phase,
                    mean=None,
                    std=None,
                    batch_size=8,
                    num_workers=4,
):
    images_dir = os.path.join(data_folder, 'images')
    masks_dir = os.path.join(data_folder, 'indexes')
    images_fps, masks_fps = get_file_paths(images_dir, masks_dir)
    fps_zip = list(zip(images_fps, masks_fps))
    train_fps, val_fps = train_test_split(fps_zip, test_size=0.15, random_state=69) # TODO: Add stratification
    # val_image_fps, _ = list(zip(*val_fps))
    # print(val_image_fps)
    fps = train_fps if phase == "train" else val_fps
    images_fps, masks_fps = list(zip(*fps))  # Unzip images and masks paths
    dataset = GolfDataset(images_fps, masks_fps, mean, std, phase, get_transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
    return dataloader


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
    image_dataset = SteelClassify(df, data_folder, mean, std, phase)
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


def shuffle_minibatch_onehot(inputs, targets, mixup=True):
    """Shuffle a minibatch and do linear interpolation between images and labels.
        Args:
            inputs: a numpy array of images with size batch_size x H x W x 3.
            targets: a numpy array of labels with size batch_size x 1.
            mixup: a boolen as whether to do mixup or not. If mixup is True, we
                sample the weight from beta distribution using parameter alpha=1,
                beta=1. If mixup is False, we set the weight to be 1 and 0
                respectively for the randomly shuffled mini-batches.
        """
    batch_size = inputs.shape[0]
    rp1 = torch.randperm(batch_size)
    inputs1 = inputs[rp1]
    targets1 = targets[rp1]
    targets1_1 = targets1.unsqueeze(1)

    rp2 = torch.randperm(batch_size)
    inputs2 = inputs[rp2]
    targets2 = targets[rp2]
    targets2_1 = targets2.unsqueeze(1)

    y_onehot = torch.FloatTensor(batch_size, 2)
    y_onehot.zero_()
    targets1_1 = targets1_1[:, :, 0]
    targets1_oh = y_onehot.scatter_(1, targets1_1, 1)

    y_onehot2 = torch.FloatTensor(batch_size, 2)
    y_onehot2.zero_()
    targets2_1 = targets2_1[:, :, 0]
    targets2_oh = y_onehot2.scatter_(1, targets2_1, 1)

    if mixup is True:
        a = np.random.beta(1, 1, [batch_size, 1])
    else:
        a = np.ones((batch_size, 1))

    b = np.tile(a[..., None, None], [1, 3, 256, 512])

    inputs1 = inputs1 * torch.from_numpy(b).float()
    inputs2 = inputs2 * torch.from_numpy(1 - b).float()

    c = np.tile(a, [1, 1])
    targets1_oh = targets1_oh.float() * torch.from_numpy(c).float()
    targets2_oh = targets2_oh.float() * torch.from_numpy(1 - c).float()

    inputs_shuffle = inputs1 + inputs2
    targets_shuffle = targets1_oh + targets2_oh

def shuffle_minibatch_combined(inputs, targets, masks, mixup=True):
    """Shuffle a minibatch and do linear interpolation between images and labels.
        Args:
            inputs: a numpy array of images with size batch_size x H x W x 3.
            targets: a numpy array of labels with size batch_size x 1.
            mixup: a boolen as whether to do mixup or not. If mixup is True, we
                sample the weight from beta distribution using parameter alpha=1,
                beta=1. If mixup is False, we set the weight to be 1 and 0
                respectively for the randomly shuffled mini-batches.
        """
    batch_size = inputs.shape[0]
    rp1 = torch.randperm(batch_size)
    inputs1 = inputs[rp1]
    targets1 = targets[rp1]
    targets1_1 = targets1.unsqueeze(1)
    masks1 = masks[rp1]

    rp2 = torch.randperm(batch_size)
    inputs2 = inputs[rp2]
    targets2 = targets[rp2]
    masks2 = masks[rp2]

    y_onehot = torch.FloatTensor(batch_size, 4)
    y_onehot.zero_()
    targets1_oh = y_onehot.scatter_(1, targets1.long(), 1)

    y_onehot2 = torch.FloatTensor(batch_size, 4)
    y_onehot2.zero_()
    targets2_oh = y_onehot2.scatter_(1, targets2.long(), 1)

    if mixup is True:
        a = np.random.beta(1, 1, [batch_size, 1])
    else:
        a = np.ones((batch_size, 1))

    b = np.tile(a[..., None, None], [1, 3, 256, 512])

    inputs1 = inputs1 * torch.from_numpy(b).float()
    inputs2 = inputs2 * torch.from_numpy(1 - b).float()

    masks1 = masks1 * torch.from_numpy(b).float()
    masks2 = masks2 * torch.from_numpy(1 - b).float()

    c = np.tile(a, [1, 4])
    targets1_oh = targets1_oh.float() * torch.from_numpy(c).float()
    targets2_oh = targets2_oh.float() * torch.from_numpy(1 - c).float()

    inputs_shuffle = inputs1 + inputs2
    targets_shuffle = targets1_oh + targets2_oh
    masks_shuffle = masks1 + masks2

    return inputs_shuffle, targets_shuffle, masks_shuffle
