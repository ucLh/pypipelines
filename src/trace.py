import argparse
import sys

import segmentation_models_pytorch as smp
import torch


def main(args):
    # model = smp.Unet("efficientnet-b7", encoder_weights=None, classes=4, activation=None,
    #                  aux_params={'classes': 4, 'dropout': 0.75})
    # model = Unet2("se_resnext50_32x4d", encoder_weights=None, classes=4, activation='softmax')
    model = smp.Unet("efficientnet-b0", encoder_weights=None, classes=4, activation=None,)
                    # aux_params={'classes': 4, 'dropout': 0.75})
    model.eval()
    model.encoder.set_swish(memory_efficient=False)
    ckpt = torch.load(f"../ckpt/{args.model_name}")
    print(f"Best loss: {ckpt['best_loss']}, epoch: {ckpt['epoch']}")
    model.load_state_dict(ckpt["state_dict"])
    traced = torch.jit.trace(model, torch.rand((1, 3, 256, 1600)))
    traced.save(f"../ckpt/traced_{args.model_name}")
    print("saved")


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str,
                        help='Name of a pth file in ../ckpt dir',
                        default='effnetb0_fpn_custom.pth')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
