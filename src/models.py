import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Threshold
import segmentation_models_pytorch as smp


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ClassifierNew(nn.Module):
    def __init__(self, inp=1280, h1=2048, out=2, d=0.75):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.mp = nn.AdaptiveMaxPool2d((1, 1))
        self.fla = Flatten()
        self.bn0 = nn.BatchNorm1d(inp, eps=1e-05, momentum=0.1, affine=True)
        self.dropout0 = nn.Dropout(d)
        self.fc1 = nn.Linear(inp, h1)
        self.bn1 = nn.BatchNorm1d(h1, eps=1e-05, momentum=0.1, affine=True)
        self.dropout1 = nn.Dropout(d)
        self.fc2 = nn.Linear(h1, out)

    def forward(self, x):
        ap = self.ap(x)
        mp = self.mp(x)
        x = torch.cat((ap, mp), dim=1)
        x = self.fla(x)
        x = self.bn0(x)
        x = self.dropout0(x)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.fc2(x)

        return x


class Encoder(smp.Unet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder = nn.Module()
        self.segmentation_head = nn.Module()

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        labels = self.classification_head(features[-1])
        return labels, features


class Decoder(smp.Unet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = nn.Module()
        self.classification_head = nn.Module()

    def forward(self, *x):
        decoder_output = self.decoder(*x)
        masks = self.segmentation_head(decoder_output)
        return masks


class Argmaxer(smp.Unet):
    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        # masks[0, 1, 200:, :] = masks[0, 1, 200:, :] - 5
        # torch.sub(masks[0, 1, 200:, :], other=5, alpha=1, out=masks[0, 1, 200:, :])

        return torch.argmax(masks, dim=1)


class FloatToIntConverter(smp.Unet):
    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)
        # Everything below zero to zero, everything above zero to one
        masks = torch.clamp(masks, min=0).bool().int()

        return masks

# class SegmentationHead(nn.Module):
#     def __init__(self, encoder, decoder, segmentation_head):
#         self.de
#
#     def forward(self, x):
#         """Sequentially pass `x` trough model`s encoder, decoder and heads"""
#         features = self.encoder(x)
#         decoder_output = self.decoder(*features)
#
#         masks = self.segmentation_head(decoder_output)
#
#         if self.classification_head is not None:
#             labels = self.classification_head(features[-1])
#             return masks, labels
#
#         return masks
