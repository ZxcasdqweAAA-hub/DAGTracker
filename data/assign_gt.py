import configparser
from typing import OrderedDict
import numpy as np
import os
import cv2
import os.path as osp
import shutil
import pandas as pd
from torch.utils.data import DataLoader
import torch
from scipy.optimize import linear_sum_assignment
from data.data_util import get_dancetrack_det_df_from_det, get_dancetrack_gt, get_mot_det_df_from_det, get_mot_gt

DET_COL_NAMES = ('frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf')
GT_COL_NAMES = ('frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'label', 'vis')
def iou(boxA, boxB):
    """
    Args:
        boxA: numpy array of bounding boxes with size (N, 4)
        boxB: numpy array of bounding boxes with size (M, 4)

    Returns:
        numpy array of size (N,M), where the (i, j) element is the IoU between the ith box in boxA and jth box in boxB.

    Note: bounding box coordinates are given in format (top, left, bottom, right)
    """
    x11, y11, x12, y12 = np.split(boxA, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxB, 4, axis=1)

    # determine the (x, y)-coordinates of the intersection rectangles
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))

    # compute the area of intersection rectangles
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)

    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)

    return iou

# ################################### SETUP ###################################
# Loader functions for the sequence type
_SEQ_TYPE_DETS_DF_LOADER = {'MOT': get_mot_det_df_from_det, 'MOT_DANCETRACK': get_dancetrack_det_df_from_det}
_SEQ_TYPE_GT_DF_LOADER = {'MOT': get_mot_gt, 'MOT_DANCETRACK': get_dancetrack_gt}


# Available sequences
_SEQ_TYPES = {}
# --------- MOT17 ---------
mot17_train_seqs = [f'MOT17-{seq_num:02}-SDP' for seq_num in (2, 4, 5, 9, 10, 11, 13)]
for seq_name in mot17_train_seqs:
    _SEQ_TYPES[seq_name] = 'MOT'

mot17_test_seqs = [f'MOT17-{seq_num:02}-SDP' for seq_num in (1, 3, 6, 7, 8, 12, 14)]
for seq_name in mot17_test_seqs:
    _SEQ_TYPES[seq_name] = 'MOT'
# ------- DanceTrack -------
dancetrack_seqs = [f'dancetrack{seq_num:04}' for seq_num in range(1,101)]
for seq_name in dancetrack_seqs:
    _SEQ_TYPES[seq_name] = 'MOT_DANCETRACK'




# ###################################  END  ###################################

class DataFrameWSeqInfo(pd.DataFrame):
    """
    Class used to store each sequences's processed detections as a DataFrame. We just add a metadata atribute to
    pandas DataFrames it so that sequence metainfo such as fps, etc. can be stored in the attribute 'seq_info_dict'.
    This attribute survives serialization.
    This solution was adopted from:
    https://pandas.pydata.org/pandas-docs/stable/development/extending.html#define-original-properties
    """
    _metadata = ['seq_info_dict']

    @property
    def _constructor(self):
        return DataFrameWSeqInfo


class MOTSeqProcessor:
    """
    Class to process detections files coming from different mot_seqs.
    Main method is process_detections. It does the following:
    - Loads a DataFrameWSeqInfo (~pd.DataFrame) from a  detections file (self.det_df) via a the 'det_df_loader' func
    corresponding to the sequence type (mapped via _SEQ_TYPES)
    - Adds Sequence Info to the df (fps, img size, moving/static camera, etc.) as an additional attribute (_get_det_df)
    - If GT is available, assigns GT identities to the detected boxes via bipartite matching (_assign_gt)
    - Stores the df on disk (_store_det_df)
    - If required, precomputes CNN embeddings for every detected box and stores them on disk (_store_embeddings)

    The stored information assumes that each MOT sequence has its own directory. Inside it all processed data is
    stored as follows:
        +-- <Sequence name>
        |   +-- processed_data
        |       +-- det
        |           +-- <dataset_params['det_file_name']>.pkl # pd.DataFrame with processed detections and metainfo
        |       +-- embeddings
        |           +-- <dataset_params['det_file_name']> # Precomputed embeddings for a set of detections
        |               +-- <CNN Name >
        |                   +-- {frame1}.jpg
        |                   ...
        |                   +-- {frameN}.jpg
    """

    def __init__(self, dataset_path, seq_name, is_tatal):
        self.seq_name = seq_name
        self.dataset_path = dataset_path
        self.seq_type = _SEQ_TYPES[seq_name]
        self.det_df_loader = _SEQ_TYPE_DETS_DF_LOADER[self.seq_type]
        self.gt_df_loader = _SEQ_TYPE_GT_DF_LOADER[self.seq_type]
        self.is_total = is_tatal

    def _sanity_check_boxes(self):
        # Sanity check that boxes do not lay completely outside
        frame_height, frame_width = self.det_df.seq_info_dict['frame_height'], self.det_df.seq_info_dict['frame_width']
        conds = (self.det_df['bb_width'] > 0) & (self.det_df['bb_height'] > 0)
        conds = conds & (self.det_df['bb_right'] > 0) & (self.det_df['bb_bot'] > 0)
        conds = conds & (self.det_df['bb_left'] < frame_width) & (self.det_df['bb_top'] < frame_height)
        if not self.det_df.equals(self.det_df[conds].copy()):
            pd.set_option('display.max_colwidth', None)
            invalid_rows = self.det_df[~conds]
            print(invalid_rows['frame_path'])
        assert self.det_df.equals(self.det_df[conds].copy()), "There are bounding boxes outside of the frame!"


    def _add_extra_det_features(self):
        """
        Create additional features for each detection. (e.g bbox centre, area etc.)
        """
        self.det_df['bb_bot'] = (self.det_df['bb_top'] + self.det_df['bb_height']).values
        self.det_df['bb_right'] = (self.det_df['bb_left'] + self.det_df['bb_width']).values
        self.det_df['feet_x'] = self.det_df['bb_left'] + 0.5 * self.det_df['bb_width']
        self.det_df['feet_y'] = self.det_df['bb_top'] + self.det_df['bb_height']

    def _assign_gt(self):
        """
        Assigns a GT identity to every detection in self.det_df, based on the ground truth boxes in self.gt_df.
        The assignment is done frame by frame via bipartite matching (Hungarian algorithm).
        Only detections with IoU >= 0.5 are considered for matching.
        """
        if self.det_df.seq_info_dict['has_gt'] and not self.det_df.seq_info_dict['is_gt']:
            print(f"Assigning ground truth identities to detections to sequence {self.seq_name}")
            for frame in self.det_df['frame'].unique():
                frame_detects = self.det_df[self.det_df.frame == frame]
                frame_gt = self.gt_df[self.gt_df.frame == frame]

                if len(frame_detects) == 0 or len(frame_gt) == 0:
                    # 当前帧无检测或无GT，所有检测设为 -1
                    self.det_df.loc[frame_detects.index, 'id'] = -1
                    continue

                # Compute IoU for each pair
                iou_matrix = iou(frame_detects[['bb_top', 'bb_left', 'bb_bot', 'bb_right']].values,
                                 frame_gt[['bb_top', 'bb_left', 'bb_bot', 'bb_right']].values)

                # 初始化所有检测 ID 为 -1
                self.det_df.loc[frame_detects.index, 'id'] = -1

                # 找出 IoU >= 0.5 的行（检测）和列（GT）
                valid_det_ixs, valid_gt_ixs = np.where(iou_matrix >= 0.5)
                if len(valid_det_ixs) == 0:
                    continue  # 无有效匹配

                # 获取唯一的检测索引和 GT 索引
                unique_det_ixs = np.unique(valid_det_ixs)
                unique_gt_ixs = np.unique(valid_gt_ixs)

                # 构建子 IoU 矩阵（只包含有效对）
                sub_iou = iou_matrix[np.ix_(unique_det_ixs, unique_gt_ixs)]
                cost_matrix = 1 - sub_iou  # 成本矩阵

                # 匈牙利算法求解最小成本匹配
                assigned_sub_det, assigned_sub_gt = linear_sum_assignment(cost_matrix)

                # 映射回原始索引
                assigned_det_ixs = unique_det_ixs[assigned_sub_det]
                assigned_gt_ixs = unique_gt_ixs[assigned_sub_gt]

                # 获取匹配的检测在 DataFrame 中的索引
                assigned_det_indices = frame_detects.iloc[assigned_det_ixs].index
                # 获取对应的 GT ID
                assigned_gt_ids = frame_gt.iloc[assigned_gt_ixs]['id'].values

                # 赋值 ID
                self.det_df.loc[assigned_det_indices, 'id'] = assigned_gt_ids

    def _get_dfs(self):
        """
        Load a pd.Dataframe with each entry corresponding to a detection. Same for the ground truth file.
        """
        # Read the dfs
        self.det_df, seq_info_dict = self.det_df_loader(self.seq_name, self.dataset_path, self.is_total)
        if seq_info_dict['has_gt']:
            self.gt_df = self.gt_df_loader(self.seq_name, self.dataset_path)

        # Copy the dataframe into our class
        self.det_df = DataFrameWSeqInfo(self.det_df)
        self.det_df.seq_info_dict = seq_info_dict

        self._add_extra_det_features()

        # Sanity check that bboxes are within the frame
#         self._sanity_check_boxes()

        # Sort the detections and assign unique detection ids
        self.det_df.sort_values(by='frame', inplace=True)
        self.det_df['detection_id'] = np.arange(self.det_df.shape[0])



    def _store_dfs(self):
        """
        Save detection dataframe under processed data as a txt file (GT is not saved).
        """
        if self.seq_type == "MOT":
            det_txt_path = "datasets/MOT17/detections/assigned"
        elif self.seq_type == "MOT_DANCETRACK":
            det_txt_path = "datasets/dancetrack/detections/assigned"
        processed_dets_path = osp.join(det_txt_path, self.seq_name)
        if not os.path.exists(processed_dets_path):
            os.makedirs(processed_dets_path)

        cols = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf']

        for frame_num, frame_df in self.det_df.groupby('frame'):
            frame_df_sorted = frame_df.sort_index()
            frame_data = frame_df_sorted[cols]
            file_name = f"{int(frame_num):06d}.txt"
            file_path = osp.join(processed_dets_path, file_name)
            frame_data.to_csv(file_path, index=False, header=False, sep=',')


    def process_detections(self, use_gt):
        """
        Main processing function.
        Load the dataframe > Assign gt > Store df > Store embeddings
        """
        self._get_dfs()  # Read the detection and ground truth files
        if use_gt:
            self._assign_gt()  # Assign ground truth ids
        self._store_dfs()  # Store the detection and gt dframes

        # return self.det_df

def gen_label(dataset, mode):
    if dataset == "MOT17":
        if mode == 'train':
            for seq in mot17_train_seqs:
                seqprocessor = MOTSeqProcessor(dataset_path="datasets/MOT17/images/train",
                                               seq_name=seq, is_tatal=False)
                seqprocessor.process_detections(use_gt=True)
        if mode == 'test':
            for seq in mot17_test_seqs:
                seqprocessor = MOTSeqProcessor(dataset_path="datasets/MOT17/images/test",
                                               seq_name=seq, is_tatal=True)
                seqprocessor.process_detections(use_gt=False)
    elif dataset == "dancetrack":
        if mode == 'train':
            path = "datasets/dancetrack/train"
            seqs = next(os.walk(path))[1]
            for seq in seqs:
                seqprocessor = MOTSeqProcessor(dataset_path="datasets/dancetrack/train",
                                               seq_name=seq, is_tatal=False)
                seqprocessor.process_detections(use_gt=True)
        if mode == 'test':
            path = "datasets/dancetrack/test"
            seqs = next(os.walk(path))[1]
            for seq in seqs:
                seqprocessor = MOTSeqProcessor(dataset_path="datasets/dancetrack/test",
                                               seq_name=seq, is_tatal=True)
                seqprocessor.process_detections(use_gt=False)




def visualize_frame(image_path, txt_path, show_fp=False, fp_color=(128,128,128)):

    img = cv2.imread(image_path)


    if not os.path.exists(txt_path):
        print(f"file not found {txt_path}")
        return img

    data = pd.read_csv(txt_path, header=None,
                       names=['frame', 'id', 'left', 'top', 'width', 'height', 'conf'])

    for _, row in data.iterrows():
        det_id = int(row['id'])
        if det_id == -1 and not show_fp:
            continue

        x1 = int(row['left'])
        y1 = int(row['top'])
        x2 = int(x1 + row['width'])
        y2 = int(y1 + row['height'])

        if det_id == -1:
            color = fp_color
        else:
            color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"ID:{det_id}" if det_id != -1 else "FP"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        cv2.rectangle(img,
                      (x1, y1 - text_h - 5),
                      (x1 + text_w, y1),
                      color, -1)

        cv2.putText(img, label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

    return img



if __name__ == '__main__':

    gen_label(dataset="dancetrack", mode='test')


