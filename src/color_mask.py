import argparse
import os
import sys

import cv2
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import numpy as np
import segmentation_models_pytorch as smp
import torch

COLOR_MAP = {0: (0, 0, 0),
             1: (0, 177, 247),
             2: (94, 30, 104),
             3: (191, 119, 56),
             4: (40, 140, 40),
             5: (146, 243, 146),
             6: (10, 250, 30),
             7: (250, 0, 55),
             8: (178, 20, 50),
             9: (0, 30, 130),
             10: (0, 255, 127),
             11: (243, 15, 190)
             }

"""
(0, 0, 0): 'unlabeled',
(0, 177, 247): 'sky',
(94, 30, 104): 'sand',
(191, 119, 56): 'ground',
(40, 140, 40): 'tree_bush',
(146, 243, 146): 'fairway_grass',
(10, 250, 30): 'raw_grass',
(250, 0, 55): 'person',
(178, 20, 50): 'animal',
(0, 30, 130): 'vehicle',
(0, 255, 127): 'green_grass',
"""


def load_model(model_name):
    model = smp.Unet("efficientnet-b0", encoder_weights=None, classes=12, activation=None, )
    model.eval()
    # model.encoder.set_swish(memory_efficient=False)
    # ckpt = torch.load(f"../ckpt/{model_name}")
    ckpt = torch.load(model_name)
    print(f"Best loss: {ckpt['best_loss']}, epoch: {ckpt['epoch']}")
    model.load_state_dict(ckpt["state_dict"])
    model.cuda()
    return model


def sub_mean_chw(data):
    data /= 255.
    data -= np.array((0.485, 0.456, 0.406))  # Broadcast subtract
    data /= np.array((0.229, 0.224, 0.225))
    return data


def read_and_resize_image(input_file_path, width, height):
    image = cv2.imread(input_file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height))
    return image


def preprocess_image(image):
    image = image.astype(np.float32)
    image = sub_mean_chw(image)
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image


def index2color(indexes):
    color_map = np.zeros((indexes.shape[0], indexes.shape[1], 3))
    for i in range(indexes.shape[0]):
        for j in range(indexes.shape[1]):
            color_map[i, j] = COLOR_MAP[indexes[i, j]][::-1]
    return color_map


def color_image(model, image_path, size, output_name):
    print(image_path)
    width, height = size
    img_original = read_and_resize_image(image_path, width, height)
    img = preprocess_image(img_original)
    img_tensor = torch.tensor(img).cuda()
    preds = model.predict(img_tensor).cpu().numpy()
    preds = preds.squeeze()

    # preds[0] = np.full((height, width), -0.5)
    indexes = np.argmax(preds, 0).astype('int32')
    color_map = index2color(indexes)

    # segmap = SegmentationMapsOnImage(indexes, shape=img.shape)
    # picture = segmap.draw_on_image(img_original)[0]
    cv2.imwrite(output_name, color_map)


def main(args):
    images_path = args.images_path
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = load_model(args.model_name)
    if os.path.isfile(images_path):
        output_name = os.path.join(output_dir, os.path.basename(images_path))
        color_image(model, args.images_path, args.size, output_name)
        return
    else:
        names = os.listdir(images_path)
        for name in names:
            output_name = os.path.join(output_dir, name)
            color_image(model, os.path.join(images_path, name), args.size, output_name)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str,
                        help='Name of a pth file in ../ckpt dir',
                        # default='effnetb0_unet_golf_classes_last.pth')
                        default='last_model.pth')
    parser.add_argument('--images_path', type=str,
                        help='Path to an image or a directory for inference',
                        default='../../autovision/segmentation_dataset/val_images/')
    parser.add_argument('--output_dir', type=str,
                        help='Path to a directory for inference',
                        default='../../autovision/segmentation_dataset/val_preds640_2/')
    parser.add_argument('--size', nargs=2, metavar=('newfile', 'oldfile'),
                        help='Width followed by the height of the image that network was configured to inference',
                        default=(1280, 640))
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
