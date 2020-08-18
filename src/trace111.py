import argparse
import sys

import segmentation_models_pytorch as smp
import torch
from pytorch_seg_models import Unet as Unet2


def main(args):
    model = smp.Unet("efficientnet-b7", encoder_weights=None, classes=4, activation=None)
    # model = Unet2("se_resnext50_32x4d", encoder_weights=None, classes=4, activation='softmax')
    # model = smp.FPN("efficientnet-b0", encoder_weights=None, classes=4, activation=None,
    #                 aux_params={'classes': 4, 'dropout': 0.75})
    model.eval()
    model.encoder.set_swish(memory_efficient=False)
    ckpt = torch.load(f"../ckpt/effnetb7_mixup_retrain_on_extended_set/{args.model_name}")
    print(f"Best loss: {ckpt['best_loss']}, epoch: {ckpt['epoch']}")
    model.load_state_dict(ckpt["state_dict"])
    sample = torch.ones([1, 3, 64, 64]).to("cuda:0")
    traced = torch.jit.trace(model, torch.rand((1, 3, 256, 1600)))
    traced.save(f"../ckpt/effnetb7_mixup_retrain_on_extended_set/traced_{args.model_name}")
    print("saved")


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str,
                        help='Name of a pth file in ../ckpt dir',
                        default='effnetb0_fpn_dice.pth')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
