import warnings
warnings.filterwarnings('ignore')
import os

import numpy as np
import cv2
import albumentations as A
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.onnx
from torch.utils.data import DataLoader
from torch.jit import load

from mlcomp.contrib.transform.albumentations import ChannelTranspose
from mlcomp.contrib.dataset.classify import ImageDataset
from mlcomp.contrib.transform.rle import rle2mask, mask2rle
from mlcomp.contrib.transform.tta import TtaWrap

# unet_se_resnext50_32x4d = \
#     load('./severstalmodels/unet_se_resnext50_32x4d.pth').cuda()
# unet_mobilenet2 = load('./severstalmodels/unet_mobilenet2.pth').cuda()
# unet_resnet34 = load('./severstalmodels/unet_resnet34.pth').cuda()
eff_net = load('./traced_model.pth').cuda()

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


model = eff_net
model.eval()

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

img_folder = '../data/cropped/'
batch_size = 1
num_workers = 0

# Different transforms for TTA wrapper
transforms = [
    [],
    [A.HorizontalFlip(p=1)]
]

transforms = [create_transforms(t) for t in transforms]
datasets = [TtaWrap(ImageDataset(img_folder=img_folder, transforms=t), tfms=t) for t in transforms]
loaders = [DataLoader(d, num_workers=num_workers, batch_size=batch_size, shuffle=False) for d in datasets]

thresholds = [0.5, 0.5, 0.5, 0.5]
min_area = [600, 600, 1000, 2000]

res = []
# Iterate over all TTA loaders
total = len(datasets[0]) // batch_size
with torch.no_grad():
    for loaders_batch in tqdm(zip(*loaders), total=total):
        preds = []
        image_file = []
        for i, batch in enumerate(loaders_batch):
            features = batch['features'].cuda()
            dummy = torch.ones((1, 3, 256, 1600)).cuda()
            output = model(dummy)
            #         print(features.shape)
            p = torch.sigmoid(output)
            # inverse operations for TTA
            p = datasets[i].inverse(p)
            preds.append(p)
            image_file = batch['image_file']

        # TTA mean
        preds = torch.stack(preds)
        preds = torch.mean(preds, dim=0)
        preds = preds.detach().cpu().numpy()
        print(preds.shape)

        # Batch post processing
        for p, file in zip(preds, image_file):
            file = os.path.basename(file)
            # Image postprocessing
            for i in range(4):
                p_channel = p[i]
                imageid_classid = file + '_' + str(i + 1)
                p_channel = (p_channel > thresholds[i]).astype(np.uint8)
                if p_channel.sum() < min_area[i]:
                    p_channel = np.zeros(p_channel.shape, dtype=p_channel.dtype)

                res.append({
                    'ImageId_ClassId': imageid_classid,
                    'EncodedPixels': mask2rle(p_channel)
                })

df = pd.DataFrame(res)
df.to_csv('submission.csv', index=False)

df = pd.DataFrame(res)
df = df.fillna('')
df.to_csv('submission.csv', index=False)

df['Image'] = df['ImageId_ClassId'].map(lambda x: x.split('_')[0])
df['Class'] = df['ImageId_ClassId'].map(lambda x: x.split('_')[1])
df['empty'] = df['EncodedPixels'].map(lambda x: not x)
classes = df[df['empty'] == False]['Class'].value_counts()
print(classes)
