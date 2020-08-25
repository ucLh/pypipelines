import argparse


def parse_arguments_train(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str,
                        help='Path to data directory which needs to be forward passed through the network',
                        default='../data/Severstal/train_test_images/')
    parser.add_argument('--df_root', type=str,
                        help='Path to csv file with ground truth masks',
                        default='../data/Severstal/train_test.csv')
    parser.add_argument('--use_mixup',
                        help='Enables mixup augmentation', action='store_true')
    parser.add_argument('--model_name', type=str,
                        help='Name for model with best loss',
                        default='model.pth')
    parser.add_argument('--mode', choices=['seg', 'cls', 'combine'],
                        help='Training mode. One of "seg", "cls" or "combine"',
                        default='combine')
    parser.add_argument('--model', type=str,
                        help='Path to (.pth) file',
                        default='../ckpt/effnetb0_classifier_maskdrop.pth')
                        # default='model.pth')
    parser.add_argument('--backend', type=str,
                        help='Model backend',
                        default='efficientnet-b0')
    parser.add_argument('--log_dir', type=str,
                        help='Directory where to write event logs',
                        default='../testing_results')
    return parser.parse_args(argv)


