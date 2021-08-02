import argparse
import csv
import os
import sys

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch


def load_model(model_name, num_classes):
    model = smp.Unet("efficientnet-b0", encoder_weights=None, classes=num_classes, activation=None, )
    model.eval()
    ckpt = torch.load(model_name, 'cuda:0')
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

    indexes = np.argmax(preds, 0).astype('int32')
    color_map = index2color(indexes, color_map)
    img_original = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)
    result = img_original + color_map
    cv2.imwrite(output_name, result)


def main(args):
    color_map = read_color_map(args.colors)
    images_path = args.images_path
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = load_model(args.model_name, args.num_classes)
    if os.path.isfile(images_path):
        output_name = os.path.join(output_dir, os.path.basename(images_path))
        color_image(model, args.images_path, args.size, output_name, color_map)
        return
    else:
        names = os.listdir(images_path)
        names = list(filter(lambda x: x.endswith('.jpg') or x.endswith('.png'), names))
        for name in names:
            output_name = os.path.join(output_dir, name)
            color_image(model, os.path.join(images_path, name), args.size, output_name, color_map)


def parse_arguments(argv):
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


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
