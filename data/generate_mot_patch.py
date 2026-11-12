import os
import argparse
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np


def generate_trajectories(file_path, groundTrues):
    f = open(file_path, 'r')

    lines = f.read().split('\n')
    values = []
    for l in lines:
        split = l.split(',')
        if len(split) < 2:
            break
        numbers = [float(i) for i in split]
        values.append(numbers)

    values = np.array(values, np.float_)

    if groundTrues:
        values = values[values[:, 6] == 1, :]  # Remove ignore objects
        values = values[values[:, 7] == 1, :]  # Pedestrian only
        values = values[values[:, 8] > 0.3, :]  # visibility only

    values = np.array(values)
    values[:, 4] += values[:, 2]
    values[:, 5] += values[:, 3]

    return values


def generate_trajectories_from_det(seq_path):
    """
    Read detection results stored as one txt per frame under a sequence directory.

    Expected per-line format in each txt:
        frame_id, id, x, y, h, w, conf

    Returns:
        ndarray with columns:
        [frame_id, id, x1, y1, x2, y2, conf]
    """
    values = []

    if not os.path.isdir(seq_path):
        raise ValueError(f"Detection path is not a directory: {seq_path}")

    txt_files = [f for f in os.listdir(seq_path) if f.endswith('.txt')]
    txt_files.sort()

    for txt_name in txt_files:
        txt_path = os.path.join(seq_path, txt_name)
        with open(txt_path, 'r') as f:
            lines = f.read().splitlines()

        for l in lines:
            if not l.strip():
                continue

            split = [x.strip() for x in l.split(',')]
            if len(split) < 7:
                continue

            numbers = [float(i) for i in split[:7]]
            frame_id, obj_id, x, y, w, h, conf = numbers
            values.append([frame_id, obj_id, x, y, x + w, y + h, conf])

    if len(values) == 0:
        return np.empty((0, 7), dtype=np.float_)

    values = np.array(values, dtype=np.float_)
    values = values[np.lexsort((values[:, 1], values[:, 0]))]
    return values

def make_parser():
    parser = argparse.ArgumentParser("MOTChallenge ReID dataset")

    parser.add_argument("--data_path", default="datasets/MOT17/images", help="path to MOT data")
    parser.add_argument("--save_path", default="datasets/MOT17/", help="Path to save the MOT-ReID dataset")
    parser.add_argument("--mot", default="17", help="MOTChallenge dataset number e.g. 17, 20")
    parser.add_argument("--train_test_split", default=True, help="Train test split ratio")
    parser.add_argument("--use_detection", default=True, help="use yolox detection or not")
    return parser


def main(args):

    # Create folder for outputs

    if not args.use_detection:
        save_path = os.path.join(args.save_path, 'ReID/gt')
        os.makedirs(save_path, exist_ok=True)
        if not args.train_test_split:
            train_save_path = os.path.join(save_path, 'all_bounding_box_train')
            os.makedirs(train_save_path, exist_ok=True)
        else:
            train_save_path = os.path.join(save_path, 'half_bounding_box_train')
            os.makedirs(train_save_path, exist_ok=True)
        test_save_path = os.path.join(save_path, 'half_bounding_box_test')
        os.makedirs(test_save_path, exist_ok=True)

        # Get gt data
        data_path = os.path.join(args.data_path, 'train')

        if args.mot == '17':
           #  s = os.listdir(data_path)
            seqs = [f for f in os.listdir(data_path) if 'FRCNN' in f]
        else:
            seqs = os.listdir(data_path)
    else:
        save_path = os.path.join(args.save_path, 'ReID/det')
        os.makedirs(save_path, exist_ok=True)
        if not args.train_test_split:
            train_save_path = os.path.join(save_path, 'all_bounding_box_train')
            os.makedirs(train_save_path, exist_ok=True)
        else:
            train_save_path = os.path.join(save_path, 'half_bounding_box_train')
            os.makedirs(train_save_path, exist_ok=True)
        test_save_path = os.path.join(save_path, 'half_bounding_box_test')
        os.makedirs(test_save_path, exist_ok=True)

        # Get gt data
        data_path = "datasets/MOT17/detections/assigned"
        seq_list = ['02', '04', '05', '09', '10', '11', '13']
        if args.mot == '17':
            #  s = os.listdir(data_path)
            seqs = [f for f in os.listdir(data_path) if f.split('-')[1] in seq_list]
        else:
            seqs = os.listdir(data_path)

    seqs.sort()

    id_offset = 0
    for seq in seqs:
        print(seq)
        print(id_offset)
        if args.use_detection:
            ground_truth_path = os.path.join(data_path, seq)
            gt = generate_trajectories_from_det(ground_truth_path)  # f, id, x_tl, y_tl, x_br, y_br, ...
        else:

            ground_truth_path = os.path.join(data_path, seq, 'gt/gt.txt')
            gt = generate_trajectories(ground_truth_path, groundTrues=True)  # f, id, x_tl, y_tl, x_br, y_br, ...

        images_path = os.path.join("datasets/MOT17/images/train", seq, 'img1')
        img_files = os.listdir(images_path)
        img_files.sort()

        num_frames = len(img_files)
        max_id_per_seq = 0
        for f in range(num_frames):

            img = cv2.imread(os.path.join(images_path, img_files[f]))

            if img is None:
                print("ERROR: Receive empty frame")
                continue

            H, W, _ = np.shape(img)

            det = gt[f + 1 == gt[:, 0], 1:].astype(np.int_)

            for d in range(np.size(det, 0)):
                id_ = det[d, 0]
                x1 = det[d, 1]
                y1 = det[d, 2]
                x2 = det[d, 3]
                y2 = det[d, 4]

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(x2, W)
                y2 = min(y2, H)

                # patch = cv2.cvtColor(img[y1:y2, x1:x2, :], cv2.COLOR_BGR2RGB)
                patch = img[y1:y2, x1:x2, :]

                max_id_per_seq = max(max_id_per_seq, id_)

                # plt.figure()
                # plt.imshow(patch)
                # plt.show()

                fileName = (
                    f"{str(id_ + id_offset).zfill(7)}_"
                    f"{seq}_"
                    f"{str(f + 1).zfill(7)}_"
                    f"x1_{str(x1).zfill(4)}_"
                    f"y1_{str(y1).zfill(4)}_"
                    f"x2_{str(x2).zfill(4)}_"
                    f"y2_{str(y2).zfill(4)}_"
                    f"acc_data.bmp"
                )
                if not args.train_test_split:
                    cv2.imwrite(os.path.join(train_save_path, fileName), patch)
                else:
                    if f < num_frames // 2:
                        cv2.imwrite(os.path.join(train_save_path, fileName), patch)
                    else:
                        cv2.imwrite(os.path.join(test_save_path, fileName), patch)

        id_offset += max_id_per_seq



if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)