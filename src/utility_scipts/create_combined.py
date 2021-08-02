import argparse
import os
import sys

import segmentation_models_pytorch as smp
import torch

from models import ClassifierNew

def main(args):
    model_seg = smp.Unet(args.backend, encoder_weights=None, classes=4, activation=None)
    model_cls = ClassifierNew()
    ckpt_cls = torch.load(args.model)
    ckpt_seg = torch.load("../ckpt/effnetb7_1024_mixup_v2.pth")
    print("Loaded existing checkpoint!", f"Continue from epoch {ckpt_cls['epoch']}", sep='\n')

    model_seg.load_state_dict(ckpt_seg["state_dict"])
    model_cls.load_state_dict(ckpt_cls["state_dict"])

    model_combined = smp.Unet(args.backend, encoder_weights=None, classes=4, activation=None, aux_params={'classes': 2})
    model_combined.classification_head = model_cls
    model_combined.encoder = model_seg.encoder
    model_combined.decoder = model_seg.decoder
    model_combined.segmentation_head = model_seg.segmentation_head

    model_combined.eval()
    model_combined.encoder.set_swish(memory_efficient=False)
    traced = torch.jit.trace(model_combined, torch.rand((1, 3, 256, 1600)))

    save_path = "../ckpt/effnetb7_combined.pth"
    if os.path.exists(save_path):
        print("Such model already exists!")
        exit(1)

    traced.save(save_path)
    print("saved")


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str,
                        help='Path to (.pth) file',
                        default='cls.pth')
    parser.add_argument('--backend', type=str,
                        help='Model backend',
                        default='efficientnet-b7')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))