import argparse
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
             }


def load_model(model_name):
    model = smp.Unet("efficientnet-b0", encoder_weights=None, classes=11, activation=None, )
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


def main(args):
    model = load_model(args.model_name)
    width, height = args.size
    img_original = read_and_resize_image(args.image, width, height)
    img = preprocess_image(img_original)
    img_tensor = torch.tensor(img).cuda()
    preds = model.predict(img_tensor).cpu().numpy()
    preds = preds.squeeze()

    indexes = np.argmax(preds, 0).astype('int32')
    color_map = index2color(indexes)

    # segmap = SegmentationMapsOnImage(indexes, shape=img.shape)
    # picture = segmap.draw_on_image(img_original)[0]
    cv2.imwrite('4_color.png', color_map)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str,
                        help='Name of a pth file in ../ckpt dir',
                        default='effnetb0_unet_golf_classes_best.pth')
    parser.add_argument('--image', type=str,
                        help='Path to an image or a directory for inference',
                        default='../../autovision/segmentation_dataset/images/69.png')
    parser.add_argument('--size', nargs=2, metavar=('newfile', 'oldfile'),
                        help='Width followed by the height of the image that network was configured to inference',
                        default=(2400, 1600))
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
