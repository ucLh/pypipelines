import argparse
import os
import sys

import cv2
from tqdm import tqdm


def read_and_crop_image(input_file_path, x_min, x_max, y_min, y_max):
    image = cv2.imread(input_file_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image[y_min:y_max, x_min:x_max]
    return image


def process_image(image_path, output_name):
    # print(image_path)
    img_cropped = read_and_crop_image(image_path, 331, 1755, 150, 850)
    cv2.imwrite(output_name, img_cropped)


def main(args):
    images_path = args.images_path
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if os.path.isfile(images_path):
        output_name = os.path.join(output_dir, os.path.basename(images_path))
        process_image(args.images_path, output_name)
        return
    else:
        names = os.listdir(images_path)
        for name in tqdm(names):
            output_name = os.path.join(output_dir, name)
            process_image(os.path.join(images_path, name), output_name)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=str,
                        help='Path to an image or a directory for inference',
                        default='/home/luch/Programming/Python/Datasets/golf/left_1')
    parser.add_argument('--output_dir', type=str,
                        help='Path to a directory for inference',
                        default='/home/luch/Programming/Python/Datasets/golf/cropped/left_1_2')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
