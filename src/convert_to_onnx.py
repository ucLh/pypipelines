import argparse
import os
import sys

import segmentation_models_pytorch as smp
import torch

from models import Argmaxer, FloatToIntConverter


# def trace_and_save_model(model, sample, dir_path, name):
#     traced = torch.jit.trace(model, sample)
#     model1_path = os.path.join(dir_path, name)
#     traced.save(model1_path)

def set_parameter_requires_grad(model, feature_extracting=True):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def main(args):
    model_seg = FloatToIntConverter(args.backend, encoder_weights=None, classes=1, activation=None)
    device = torch.device('cpu')
    ckpt_seg = torch.load(args.segmentation_net, device)
    print("Loaded existing checkpoint!", f"Continue from epoch {ckpt_seg['epoch']}", sep='\n')

    model_seg.load_state_dict(ckpt_seg["state_dict"])

    dir_path = '../ckpt/autovis/'

    model_seg.eval()
    model_seg = model_seg.to(device)
    model_seg.encoder.set_swish(memory_efficient=False)

    arch = 'effnetb0_unet_wgisd_iou86_1344x2048.onnx'

    print(model_seg)
    print('')

    # create example image data
    input_ = torch.ones((1, 3, 1344, 2048))
    input_ = input_.to(device)
    print('input_ size:  {:d}x{:d}'.format(1344, 2048))

    # format output model path
    save_path = os.path.join(dir_path, arch)

    # export the model
    input_names = ["input_0"]
    output_names = ["output_0"]

    print('exporting model to ONNX...')
    torch.onnx.export(model_seg, input_, save_path, export_params=True, verbose=True, output_names=output_names,
                      input_names=input_names,
                      opset_version=9)
    print('model exported to:  {:s}'.format(save_path))
    # python3 -m onnxsim effnetb0_unet_gray_2grass_iou55_640x1280.onnx effnetb0_unet_gray_2grass_iou55_640x1280.onnx


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--segmentation_net', type=str,
                        help='Path to (.pth) file with segmentation net weights',
                        # default='/home/luch/Programming/Python/autovision/pytorch-segmentation/models/'
                        #         'v3/effnetb0_unet_golf_square.pth')
                        default='../ckpt/wgisd/wgisd_iou86.pth')
    parser.add_argument('--backend', type=str,
                        help='Model backend',
                        default='efficientnet-b0')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
