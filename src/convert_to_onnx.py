import argparse
import subprocess
import sys

import torch

from arguments import parse_arguments_convert_to_onnx
from models import Argmaxer, Thresholder


def set_parameter_requires_grad(model, feature_extracting=True):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def main(args):
    num_classes = args.num_classes
    if num_classes == 1:
        # We need to threshold only one mask
        model_seg = Thresholder(args.backend, encoder_weights=None, classes=num_classes, activation=None)
    else:
        # We argmax all the masks
        model_seg = Argmaxer(args.backend, encoder_weights=None, classes=num_classes, activation=None)
    device = torch.device('cpu')
    ckpt_seg = torch.load(args.model_in, device)
    print("Loaded existing checkpoint!", f"Continue from epoch {ckpt_seg['epoch']}", sep='\n')

    model_seg.load_state_dict(ckpt_seg["state_dict"])
    model_seg.eval()
    model_seg = model_seg.to(device)

    # Change swish activation's mode for onnx conversion
    if 'efficientnet' in args.backend:
        model_seg.encoder.set_swish(memory_efficient=False)

    # create example image data
    width, height = args.size
    input_ = torch.ones((1, 3, height, width))
    input_ = input_.to(device)
    print('input size: {:d}x{:d}'.format(height, width))

    # format output model path
    save_path = args.model_out

    # export the model
    input_names = ["input_0"]
    output_names = ["output_0"]

    print('exporting model to ONNX...')
    torch.onnx.export(model_seg, input_, save_path, export_params=True, verbose=True, output_names=output_names,
                      input_names=input_names,
                      opset_version=9)
    print('model exported to:  {:s}'.format(save_path))

    # Simplify the onnx network, it is needed for later TensorRT conversion
    # Some network backbones, like ResNet, should work without it, but for EfficientNet you need it
    subprocess.run(f'python3 -m onnxsim {args.model_out} {args.model_out}', shell=True)


if __name__ == '__main__':
    main(parse_arguments_convert_to_onnx(sys.argv[1:]))
