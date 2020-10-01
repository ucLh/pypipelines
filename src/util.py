import enum
import logging
logging.basicConfig(level=logging.DEBUG, filemode='w')
import os
import re
from shutil import copyfile

import numpy as np
import torch


class TrainerModes(enum.Enum):
    seg = "seg"
    cls = "cls"
    combine = "combine"


class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.nn.functional.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


def predict(X, threshold):
    '''X is sigmoid output of the model'''
    X_p = np.copy(X)
    preds = (X_p > threshold).astype('uint8')
    return preds


def metric(probability, truth, threshold=0.5, reduction='none'):
    '''Calculates dice of positive and negative images seperately'''
    '''probability and truth must be torch tensors'''
    batch_size = len(truth)
    masks_batch = []
    for i in range(probability.shape[0]):
        masks = [(truth[i,0] == v) for v in [i for i in range(probability.shape[1])]]
        mask_1batch = np.stack(masks, axis=0).astype('float')
        masks_batch.append(mask_1batch)
    truth = torch.tensor(np.stack(masks_batch, axis=0).astype('float')).cpu()
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert(probability.shape == truth.shape)

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

#         dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
#         dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
#         dice = dice.mean().item()

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice, dice_neg, dice_pos, num_neg, num_pos


def dice_channel_torch(probability, truth, threshold):
    batch_size = truth.shape[0]
    channel_num = truth.shape[1]
    mean_dice_channel = 0.
    with torch.no_grad():
        for i in range(batch_size):
            for j in range(channel_num):
                channel_dice = dice_single_channel(probability[i, j,:,:], truth[i, j, :, :], threshold)
                mean_dice_channel += channel_dice/(batch_size * channel_num)
    return mean_dice_channel


def dice_single_channel(probability, truth, threshold, eps=1E-9):
    p = (probability.view(-1) > threshold).float()
    t = (truth.view(-1) > 0.5).float()
    dice = (2.0 * (p * t).sum() + eps)/(p.sum() + t.sum() + eps)
    return dice


class Meter:
    '''A meter to keep track of iou and dice scores throughout an epoch'''
    def __init__(self, phase, epoch):
        self.base_threshold = 0.5 # <<<<<<<<<<< here's the threshold
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []

    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)
        dice, dice_neg, dice_pos, _, _ = metric(probs, targets, self.base_threshold)
        self.base_dice_scores.extend(dice.tolist())
        self.dice_pos_scores.extend(dice_pos.tolist())
        self.dice_neg_scores.extend(dice_neg.tolist())
        preds = predict(probs, self.base_threshold)
        iou = compute_iou_batch(preds, targets, classes=[1])
        self.iou_scores.append(iou)

    def get_metrics(self):
        dice = np.nanmean(self.base_dice_scores)
        dice_neg = np.nanmean(self.dice_neg_scores)
        dice_pos = np.nanmean(self.dice_pos_scores)
        dices = [dice, dice_neg, dice_pos]
        iou = np.nanmean(self.iou_scores)
        return dices, iou


class MetricsLogger:

    def __init__(self, log_dir='../logs'):
        if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
            os.makedirs(log_dir)
        self.logger = self.config_logger('metrics_logger', '../logs/metrics.log')

    @staticmethod
    def config_logger(logger_name, file, level=logging.INFO):
        logger = logging.getLogger(logger_name)
        handler1 = logging.FileHandler(file)
        handler1.setLevel(level)
        logger.addHandler(handler1)
        return logger

    def epoch_log(self, mode, phase, epoch, epoch_loss_list, meter):
        '''logging the metrics at the end of an epoch'''
        if mode == TrainerModes.combine:
            return self.combine_epoch_log(phase, epoch, epoch_loss_list, meter)
        elif mode == TrainerModes.seg:
            return self.seg_epoch_log(phase, epoch, epoch_loss_list, meter)
        elif mode == TrainerModes.cls:
            return self.cls_epoch_log(phase, epoch, epoch_loss_list)

    def combine_epoch_log(self, phase, epoch, epoch_loss_list, meter):
        overall_loss, loss_seg, loss_cls, loss_dice = epoch_loss_list
        dices, iou = meter.get_metrics()
        dice, dice_neg, dice_pos = dices
        if phase == 'val':
            self.logger.info(f"Epoch: {epoch}| Loss: {overall_loss:.4f} | SegLoss: {loss_seg:.4f} | "
                             f"ClsLoss: {loss_cls:.4f} | DiceLoss: {loss_dice:.4f} | IoU: {iou:.4f} | "
                             f"dice: {dice:.4f} | dice_neg: {dice_neg:.4f} | "
                             f"dice_pos: {dice_pos:.4f}")
        else:
            print(f"Epoch: {epoch}| Loss: {overall_loss:.4f} | SegLoss: {loss_seg:.4f} | "
                  f"ClsLoss: {loss_cls:.4f} | DiceLoss: {loss_dice:.4f} | IoU: {iou:.4f} | dice: {dice:.4f} | "
                  f"dice_neg: {dice_neg:.4f} | dice_pos: {dice_pos:.4f}")
        return dice, iou

    def cls_epoch_log(self, phase, epoch, epoch_loss_list):
        epoch_loss = epoch_loss_list[0]
        if phase == 'val':
            self.logger.info(f"Epoch: {epoch}| Loss: {epoch_loss:.4f}")
        else:
            print(f"Epoch: {epoch}| Loss: {epoch_loss:.4f}")
        return None, None

    def seg_epoch_log(self, phase, epoch, epoch_loss_list, meter):
        overall_loss = epoch_loss_list[0]
        dices, iou = meter.get_metrics()
        dice, dice_neg, dice_pos = dices
        if phase == 'val':
            self.logger.info(f"Epoch: {epoch}| Loss: {overall_loss:.4f} | IoU: {iou:.4f} | "
                             f"dice: {dice:.4f} | dice_neg: {dice_neg:.4f} | "
                             f"dice_pos: {dice_pos:.4f}")
        else:
            print(f"Epoch: {epoch}| Loss: {overall_loss:.4f} | "
                  f"IoU: {iou:.4f} | dice: {dice:.4f} | "
                  f"dice_neg: {dice_neg:.4f} | dice_pos: {dice_pos:.4f}")
        return dice, iou


def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    '''computes iou for one ground truth mask and predicted mask'''
    pred[label == ignore_index] = 0
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]


def compute_iou_batch(outputs, labels, classes=None):
    '''computes mean iou for a batch of ground truth masks and predicted masks'''
    masks_batch = []
    for i in range(outputs.shape[0]):
        masks = [(labels[i, 0] == v) for v in [i for i in range(outputs.shape[1])]]
        mask_1batch = np.stack(masks, axis=0).astype('float')
        masks_batch.append(mask_1batch)
    labels = np.stack(masks_batch, axis=0).astype('float')
    ious = []
    preds = np.copy(outputs) # copy is imp
    # labels = np.array(labels) # tensor to np
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label, classes)))
    iou = np.nanmean(ious)
    return iou


def set_parameter_requires_grad(model, feature_extracting=True):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def divide_files(prefix='/home/luch/Programming/Python/autovision/segmentation_dataset/golf_v2_colored/'):
    names = os.listdir(prefix)
    names = list(filter(lambda x: os.path.isfile(prefix + x), names))
    print(len(names))
    for name in names:
        src_file = prefix + name
        if re.search(r'_color_mask', name):
            dest_file = prefix + 'masks/' + name
            print(dest_file)
            copyfile(src_file, dest_file)
        else:
            dest_file = prefix + 'images/' + name
            print(src_file)
            copyfile(src_file, dest_file)
