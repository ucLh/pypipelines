import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ClassifierNew(nn.Module):
    def __init__(self, encoder, inp=643, h1=1024, out=2, d=0.5):
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
        self.encoder = encoder

    def forward(self, x):
        features = self.encoder(x)[-1]
        ap = self.ap(features)
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