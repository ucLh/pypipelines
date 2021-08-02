import argparse


def parse_arguments_train(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str,
                        help='Path to data directory which needs to be forward passed through the network',
                        default='/home/luch/Programming/Python/autovision/segmentation_dataset/')
                        # default='/home/luch/Programming/Python/Datasets/golf/')
    parser.add_argument('--df_root', type=str,
                        help='Path to csv file with ground truth masks',
                        default='../data/Severstal/train_test.csv')
    parser.add_argument('--num_classes', type=int,
                        help='Number of semantic classes for the model',
                        default=4)
    parser.add_argument('--use_mixup',
                        help='Enables mixup augmentation', action='store_true')
    parser.add_argument('--model_name', type=str,
                        help='Name for model with best loss',
                        default='effnetb0_unet_golf_gray.pth')
    parser.add_argument('--mode', choices=['seg', 'cls', 'combine'],
                        help='Training mode. One of "seg", "cls" or "combine"',
                        default='seg')
    parser.add_argument('--model', type=str,
                        help='Path to (.pth) file',
                        # default='../ckpt/effnetb0_unet_golf_.pth')
                        # default='model.pth')
                        default='effnetb0_unet_gray2.pth')
    parser.add_argument('--backend', type=str,
                        help='Model backend',
                        default='efficientnet-b0')
    parser.add_argument('--log_dir', type=str,
                        help='Directory where to write event logs',
                        default='../testing_results')
    return parser.parse_args(argv)


def parse_arguments_color_mask(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str,
                        help='Name of a pth file in ../ckpt dir',
                        default='../ckpt/autovis/effnetb0_unet_gray_2grass_iou55.pth')
    parser.add_argument('--images_path', type=str,
                        help='Path to an image or a directory for inference',
                        default='../../autovision/segmentation_dataset/gray_images/')
    parser.add_argument('--output_dir', type=str,
                        help='Path to a directory for inference',
                        default='../../autovision/segmentation_dataset/val_preds640_2/')
    parser.add_argument('--size', nargs=2, metavar=('width', 'height'),
                        help='Width followed by the height of the image that network was configured to inference',
                        default=(1280, 640))
    parser.add_argument('--colors', type=str,
                        help='Path to a csv file with color map',
                        default='colors_grass.csv')
    parser.add_argument('--num_classes', type=int,
                        help='Number of semantic classes for the model',
                        default=11)
    return parser.parse_args(argv)


def parse_arguments_convert_to_onnx(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_in', type=str,
                        help='Path to a (.pth) file with segmentation net weights',
                        default='../ckpt/wgisd/effnetb0_unet_gray_2grass_iou55.pth')
    parser.add_argument('--model_out', type=str,
                        help='Path to the resulting (.onnx) network',
                        default='../ckpt/wgisd/effnetb0_unet_gray_2grass_iou55.onnx')
    parser.add_argument('--num_classes', type=int,
                        help='Number of semantic classes for the model',
                        default=11)
    parser.add_argument('--size', nargs=2, metavar=('width', 'height'),
                        help='Width followed by the height of the image that network will be configured to inference',
                        default=(2048, 1344))
    parser.add_argument('--backend', type=str,
                        help='Model backend',
                        default='efficientnet-b0')
    return parser.parse_args(argv)
