import argparse
import csv
import os
import sys

import cv2
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import numpy as np
import segmentation_models_pytorch as smp
import torch

COLOR_MAP = {
    0: (0, 177, 247),
    1: (94, 30, 104),
    2: (191, 119, 56),
    3: (40, 140, 40),
    4: (146, 243, 146),
    5: (10, 250, 30),
    6: (250, 0, 55),
    7: (178, 20, 50),
    8: (0, 30, 130),
    9: (0, 255, 127),
    10: (243, 15, 190),
    11: (0, 0, 0),
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
    model = smp.Unet("efficientnet-b0", encoder_weights=None, classes=1, activation=None, )
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


def read_color_map(csv_path):
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        color_map = None
        for line in reader:
            temp = np.array(tuple(map(int, line['color'].split()))[::-1])
            if color_map is None:
                color_map = temp[np.newaxis, :]
            else:
                color_map = np.concatenate((color_map, temp[np.newaxis, :]))
    return color_map


def index2color(indexes, color_map):

    def map_color(index):
        return color_map[index]

    colored_mask = map_color(indexes)
    return colored_mask


def color_image(model, image_path, size, output_name, color_map):
    print(image_path)
    width, height = size
    img_original = read_and_resize_image(image_path, width, height)
    img = preprocess_image(img_original)
    img_tensor = torch.tensor(img).cuda()
    preds = model.predict(img_tensor).cpu().numpy()
    preds = preds.squeeze()

    # indexes = np.argmax(preds, 0).astype('int32')
    preds[preds < 0.1] = 0
    preds[preds >= 0.1] = 1
    color_map = index2color(preds.astype('int32'), color_map)
    img_original = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)
    result = img_original + color_map
    # segmap = SegmentationMapsOnImage(indexes, shape=img.shape)
    # picture = segmap.draw_on_image(img_original)[0]
    cv2.imwrite(output_name, result)


def main(args):
    color_map = read_color_map(args.colors)
    images_path = args.images_path
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = load_model(args.model_name)
    if os.path.isfile(images_path):
        output_name = os.path.join(output_dir, os.path.basename(images_path))
        color_image(model, args.images_path, args.size, output_name, color_map)
        return
    else:
        names = os.listdir(images_path)
        names = list(filter(lambda x: x.endswith('.jpg'), names))
        for name in names:
            output_name = os.path.join(output_dir, name)
            color_image(model, os.path.join(images_path, name), args.size, output_name, color_map)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str,
                        help='Name of a pth file in ../ckpt dir',
                        # default='effnetb0_unet_golf_classes_last.pth')
                        default='../ckpt/wgisd/wgisd_iou86.pth')
    parser.add_argument('--images_path', type=str,
                        help='Path to an image or a directory for inference',
                        default='../../autovision/wgisd/mask_test_data')
    parser.add_argument('--output_dir', type=str,
                        help='Path to a directory for inference',
                        default='../../autovision/wgisd/preds/thresh_0_1344x2048_another_ckpt/test_preds')
    parser.add_argument('--size', nargs=2, metavar=('newfile', 'oldfile'),
                        help='Width followed by the height of the image that network was configured to inference',
                        default=(2048, 1344))
    parser.add_argument('--colors', type=str,
                        help='Path to a csv file with color map',
                        default='../../autovision/wgisd/colors.csv')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
