import gzip

import numpy as np
import cv2
import torch
import os
import pickle
from pathlib import Path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)
from data.reid.embedding import EmbeddingComputer

dataset = "dance"
test_dataset = True
grid_off = True
os.environ['CUDA_VISIBLE_DEVICES'] = str(3)

DANCE_ROOT = "datasets/dancetrack"
DEPTH_ROOT = os.path.join(DANCE_ROOT, "DepthAnythingV2")
GTS_ROOT = os.path.join(DANCE_ROOT, "detections/assigned")
IMG_ROOT = os.path.join(DANCE_ROOT, "train")
SAVE_PATH = "datasets/dancetrack/reid_features_assigned_v2_no_half_25.pkl"

embedding_computer = EmbeddingComputer(dataset=dataset, test_dataset=test_dataset, grid_off=grid_off)


def xywh_to_xyxy(boxes):
    xyxy_boxes = boxes.copy()
    xyxy_boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    xyxy_boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    return xyxy_boxes


def load_gts_detections(gts_path):
    detections = []
    if not os.path.exists(gts_path):
        return np.array(detections)

    for line in open(gts_path):
        parts = line.strip("\n").split(",")
        _, obj_id, x, y, w, h, v = parts
        obj_id, x, y, w, h, v = map(float, (obj_id, x, y, w, h, v))
        detections.append([obj_id, x, y, w, h])

    return np.array(detections)


def get_all_sdp_sequences(train_root):
    sdp_seqs = [d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d))]
    return sorted(sdp_seqs)


def process_single_sequence(seq_name):
    seq_reid_features = {}

    seq_img_dir = os.path.join(IMG_ROOT, seq_name, "img1")
    seq_gts_dir = os.path.join(GTS_ROOT, seq_name)
    seq_depth_dir = os.path.join(DEPTH_ROOT, seq_name)

    img_files = sorted([f for f in os.listdir(seq_img_dir) if f.endswith(".jpg")])

    for img_file in img_files:
        frame_num = int(os.path.splitext(img_file)[0])
        frame_key = f"{seq_name}:{frame_num}"

        img_path = os.path.join(seq_img_dir, img_file)
        gts_path = os.path.join(seq_gts_dir, f"{frame_num:06d}.txt")
        depth_path = os.path.join(seq_depth_dir, f"{frame_num:08d}.gz")

        original_img = cv2.imread(img_path)
        if original_img is None:
            print(f"could not load image {img_path}")
            continue

        gts_detections = load_gts_detections(gts_path)
        if len(gts_detections) == 0:
            seq_reid_features[frame_key] = np.array([])
            continue

        det_boxes_xyxy = xywh_to_xyxy(gts_detections[:, 1:])

        with gzip.open(depth_path, "rb") as f:
            depth_data = pickle.load(f)["tensor"].numpy().astype(np.float32)

        reid_features = embedding_computer.compute_embedding(
            original_img,
            det_boxes_xyxy[:, :4],
            depth_data,
            frame_key
        )
        seq_reid_features[frame_key] = reid_features

    return seq_reid_features


def main():
    sdp_sequences = get_all_sdp_sequences(IMG_ROOT)

    all_reid_features = {}

    for seq in sdp_sequences:
        print(f"processing  sequence: {seq}")
        seq_features = process_single_sequence(seq)
        all_reid_features.update(seq_features)

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, 'wb') as f:
        pickle.dump(all_reid_features, f)


if __name__ == "__main__":
    main()