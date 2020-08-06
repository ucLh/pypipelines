import argparse
import os
import sys

import segmentation_models_pytorch as smp
import torch

from models import Encoder, Decoder

from models import ClassifierNew


def trace_and_save_model(model, sample, dir_path):
    traced = torch.jit.trace(model, sample)
    model1_path = os.path.join(dir_path, 'encoder.pth')
    traced.save(model1_path)


def main(args):
    model_seg = smp.Unet(args.backend, encoder_weights=None, classes=4, activation=None)
    model_cls = ClassifierNew()
    ckpt_cls = torch.load(args.classifier)
    ckpt_seg = torch.load(args.segmentation_net)
    print("Loaded existing checkpoint!", f"Continue from epoch {ckpt_cls['epoch']}", sep='\n')

    model_seg.load_state_dict(ckpt_seg["state_dict"])
    model_cls.load_state_dict(ckpt_cls["state_dict"])

    encoder = Encoder(args.backend, encoder_weights=None, classes=4, activation=None, aux_params={'classes': 2})
    decoder = Decoder(args.backend, encoder_weights=None, classes=4, activation=None)

    encoder.encoder = model_seg.encoder
    encoder.classification_head = model_cls

    decoder.decoder = model_seg.decoder
    decoder.segmentation_head = model_seg.segmentation_head

    dir_path = '../ckpt/enc_dec/effnetb7_clsv3/'
    if os.path.exists(dir_path):
        print("Such models already exists!")
        exit(1)
    os.mkdir(dir_path)

    encoder.eval()
    encoder.encoder.set_swish(memory_efficient=False)
    trace_and_save_model(encoder, torch.rand((1, 3, 256, 1600)), dir_path)
    print('Encoder saved!')

    decoder_sample = model_seg.encoder(torch.rand((1, 3, 256, 1600)))
    decoder.eval()
    trace_and_save_model(decoder, decoder_sample, dir_path)
    print('Decoder saved!')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--classifier', type=str,
                        help='Path to (.pth) file with classifier weights',
                        default='../ckpt/cls_v3.pth')
    parser.add_argument('--segmentation_net', type=str,
                        help='Path to (.pth) file with segmentation net weights',
                        default='../ckpt/effnetb7_1024_mixup_v2.pth')
    parser.add_argument('--backend', type=str,
                        help='Model backend',
                        default='efficientnet-b7')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
