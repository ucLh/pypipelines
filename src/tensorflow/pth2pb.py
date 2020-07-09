import argparse
import sys
import warnings

import onnx
from onnx_tf.backend import prepare
import torch
import torch.onnx
from torch.jit import load

warnings.filterwarnings('ignore')


def main(args):
    # Load model
    model = load(args.model_path).cuda()
    # Pth -> onnx
    if args.do_onnx:
        features = torch.rand((1, 3, 256, 1600)).cuda()
        torch.onnx._export(model, features, args.onnx_name,
                           example_outputs=torch.rand((1, 4, 256, 1600)), export_params=True,
                           input_names=['resnet_input'], output_names=['resnet_output'])

    # Onnx -> pb
    onnx_model = onnx.load(args.onnx_name)  # load onnx model
    tf_rep = prepare(onnx_model)  # prepare tf representation
    tf_rep.export_graph(args.pb_name)  # export the model


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str,
                        help='Path to pth model', default='./data/severstalmodels/unet_resnet34.pth')
    parser.add_argument('--onnx_name', type=str,
                        help='Desired name for onnx model.', default='./data/severstalmodels/unet_resnet34.onnx')
    parser.add_argument('--pb_name', type=str,
                        help='Desired name for pb model.', default='./data/severstalmodels/unet_resnet34.pb')
    parser.add_argument('--do_onnx',
                        help='Whether to convert to onnx', action='store_true')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
