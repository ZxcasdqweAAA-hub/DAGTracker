import argparse


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')

    parser.add_argument('--model_path', type=str,
                        default='models/YOLOX/weights/topictrack_mot17.pth.tar',
                        help='path to model')
    parser.add_argument('--dataset', type=str, default='mot17')
    parser.add_argument('--data_cfg', type=str, default='cfg/mot17.json')

    return parser