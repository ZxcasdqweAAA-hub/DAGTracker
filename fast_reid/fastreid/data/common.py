# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import gzip
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from .data_utils import read_image


class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel
        if "MOT17" in img_items[0][0]:
            self.depth_path = "datasets/MOT17/DepthAnythingV2"
        else:
            self.depth_path = "datasets/dancetrack/DepthAnythingV2"
        pid_set = set()
        cam_set = set()
        for i in img_items:
            pid_set.add(i[1])
            cam_set.add(i[2])

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))
        if relabel:
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]
        pid = img_item[1]
        camid = img_item[2]
        depth_info = img_item[3]
        seq, frame, x1, y1, x2, y2 = depth_info
        if "dancetrack" in self.depth_path:
            depth_path = os.path.join(self.depth_path, seq, '0' * (8 - len(str(frame))) + str(frame) + ".gz")
            with gzip.open(depth_path, "rb") as f:
                depth = pickle.load(f)["tensor"].numpy().astype(np.float32)
        else:
            depth_path = os.path.join(self.depth_path, seq,  '0' * (6 - len(str(frame))) + str(frame) + ".pt")
            depth = torch.load(depth_path)
        depth_crop = depth[y1:y2, x1:x2]
        valid = depth_crop > 0.01
        d_near, d_far = depth_crop[valid].min(), depth_crop[valid].max()
        depth_crop = (depth_crop - d_near) / (d_far - d_near)
        img = read_image(img_path)
        if self.transform is not None: img, depth_crop = self.transform(img, depth_crop)
        # if self.transform is not None: img= self.transform(img)
        if self.relabel:
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]
        return {
            "images": img,
            "targets": pid,
            "camids": camid,
            "img_paths": img_path,
            "depth": depth_crop,
        }

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)
