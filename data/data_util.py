
import configparser
import os.path as osp
import pandas as pd
DET_COL_NAMES = ('frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf')
GT_COL_NAMES = ('frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'label', 'vis')
dancetrack_seq_fps = {
    "dancetrack0001": 20,
    "dancetrack0002": 20,
    "dancetrack0006": 20,
    "dancetrack0008": 20,
    "dancetrack0012": 20,
    "dancetrack0015": 20,
    "dancetrack0016": 20,
    "dancetrack0020": 20,
    "dancetrack0023": 20,
    "dancetrack0024": 20,
    "dancetrack0027": 20,
    "dancetrack0029": 20,
    "dancetrack0032": 20,
    "dancetrack0033": 20,
    "dancetrack0037": 20,
    "dancetrack0039": 20,
    "dancetrack0044": 20,
    "dancetrack0045": 20,
    "dancetrack0049": 20,
    "dancetrack0051": 20,
    "dancetrack0052": 20,
    "dancetrack0053": 20,
    "dancetrack0055": 20,
    "dancetrack0057": 20,
    "dancetrack0061": 20,
    "dancetrack0062": 20,
    "dancetrack0066": 20,
    "dancetrack0068": 20,
    "dancetrack0069": 20,
    "dancetrack0072": 20,
    "dancetrack0074": 20,
    "dancetrack0075": 20,
    "dancetrack0080": 20,
    "dancetrack0082": 20,
    "dancetrack0083": 20,
    "dancetrack0086": 20,
    "dancetrack0087": 20,
    "dancetrack0096": 20,
    "dancetrack0098": 20,
    "dancetrack0099": 20,
}
dancetrack_seq_size = {
    "dancetrack0001": (1280, 720),
    "dancetrack0002": (1920, 1080),
    "dancetrack0006": (1920, 1080),
    "dancetrack0008": (1920, 1080),
    "dancetrack0012": (1920, 1080),
    "dancetrack0015": (1280, 720),
    "dancetrack0016": (1920, 1080),
    "dancetrack0020": (1920, 1080),
    "dancetrack0023": (1920, 1080),
    "dancetrack0024": (1280, 720),
    "dancetrack0027": (1280, 720),
    "dancetrack0029": (1920, 1080),
    "dancetrack0032": (1920, 1080),
    "dancetrack0033": (1920, 1080),
    "dancetrack0037": (1920, 1080),
    "dancetrack0039": (1920, 1080),
    "dancetrack0044": (1920, 1080),
    "dancetrack0045": (1280, 720),
    "dancetrack0049": (1920, 1080),
    "dancetrack0051": (1920, 1080),
    "dancetrack0052": (1920, 1080),
    "dancetrack0053": (1280, 720),
    "dancetrack0055": (1280, 720),
    "dancetrack0057": (1920, 1080),
    "dancetrack0061": (1920, 1080),
    "dancetrack0062": (1920, 1080),
    "dancetrack0066": (1920, 1080),
    "dancetrack0068": (1920, 1080),
    "dancetrack0069": (1280, 720),
    "dancetrack0072": (1280, 720),
    "dancetrack0074": (1920, 1080),
    "dancetrack0075": (1280, 720),
    "dancetrack0080": (1920, 1080),
    "dancetrack0082": (1920, 1080),
    "dancetrack0083": (1920, 988),
    "dancetrack0086": (1920, 1080),
    "dancetrack0087": (1280, 720),
    "dancetrack0096": (1920, 1080),
    "dancetrack0098": (1280, 720),
    "dancetrack0099": (1920, 1080),
}
def get_mot_det_df_from_det(seq_name, data_root_path, is_total):


    seq_path = osp.join(data_root_path, seq_name)  # Sequence path
    det_file_path = osp.join("datasets/MOT17/detections", f"{seq_name}.txt")  # Detection file

    det_df = pd.read_csv(det_file_path, header=None)
    det_df = det_df[det_df.columns[:len(DET_COL_NAMES)]]
    det_df.columns = DET_COL_NAMES

    if not is_total:
        det_df = det_df[det_df['conf'].ge(0.60)].copy()

    det_df['bb_left'] -= 1
    det_df['bb_top'] -= 1

    if len(det_df['id'].unique()) > 1:
        det_df['preprocessor_id'] = det_df['id']
    det_df['id'] = -1  # Erase the id column

    det_df['frame_path'] = det_df['frame'].apply(lambda frame_num: osp.join(seq_path, f'img1/{int(frame_num):06}.jpg'))

    assert osp.exists(det_df['frame_path'].iloc[0])  # Sanity check

    # Build scene info dictionary
    info_file_path = osp.join(seq_path, 'seqinfo.ini')
    cp = configparser.ConfigParser()
    cp.read(info_file_path)
    seq_info_dict = {'seq': seq_name,
                     'seq_path': seq_path,
                     'det_file_name': "ByteTrack",
                     'frame_height': int(cp.get('Sequence', 'imHeight')),
                     'frame_width': int(cp.get('Sequence', 'imWidth')),
                     'seq_len': int(cp.get('Sequence', 'seqLength')),
                     'fps': int(cp.get('Sequence', 'frameRate')),
                     'has_gt': osp.exists(osp.join(seq_path, 'gt')),
                     'is_gt': False}

    return det_df, seq_info_dict

def get_dancetrack_det_df_from_det(seq_name, data_root_path, is_total):
    """
    Load MOT detection file using detection files
    """

    seq_path = osp.join(data_root_path, seq_name)  # Sequence path
    det_file_path = osp.join("datasets/dancetrack/detections/bytetrack", seq_name, "byte065.txt")  # Detection file

    # Number and order of columns is always assumed to be the same
    det_df = pd.read_csv(det_file_path, header=None)
    det_df = det_df[det_df.columns[:len(DET_COL_NAMES)]]
    det_df.columns = DET_COL_NAMES

    if not is_total:
        det_df = det_df[det_df['conf'].ge(0.60)].copy()

    # Coordinates are 1 based
    det_df['bb_left'] -= 1
    det_df['bb_top'] -= 1

    # If id already contains an ID assignment (e.g. using tracktor output), keep it
    if len(det_df['id'].unique()) > 1:
        det_df['preprocessor_id'] = det_df['id']
    det_df['id'] = -1  # Erase the id column

    # Include frame paths into the dataframe
    det_df['frame_path'] = det_df['frame'].apply(lambda frame_num: osp.join(seq_path, f'img1/{int(frame_num):08}.jpg'))

    assert osp.exists(det_df['frame_path'].iloc[0])  # Sanity check

    # Build scene info dictionary
    info_file_path = osp.join(seq_path, 'seqinfo.ini')
    cp = configparser.ConfigParser()
    cp.read(info_file_path)
    seq_info_dict = {'seq': seq_name,
                     'seq_path': seq_path,
                     'det_file_name': "ByteTrack",
                     'frame_height': int(cp.get('Sequence', 'imHeight')),
                     'frame_width': int(cp.get('Sequence', 'imWidth')),
                     'seq_len': int(cp.get('Sequence', 'seqLength')),
                     'fps': int(cp.get('Sequence', 'frameRate')),
                     'has_gt': osp.exists(osp.join(seq_path, 'gt')),
                     'is_gt': False}

    return det_df, seq_info_dict

def get_mot_gt(seq_name, data_root_path):
    """
    Load MOT ground truth file
    """
    seq_path = osp.join(data_root_path, seq_name)  # Sequence path
    gt_file_path = osp.join(seq_path, "gt/gt.txt")  # Ground truth file

    # Read the gt file and assign the column names
    gt_df = pd.read_csv(gt_file_path, header=None)
    gt_df = gt_df[gt_df.columns[:len(GT_COL_NAMES)]]
    gt_df.columns = GT_COL_NAMES

    # Coordinates are 1 based
    gt_df['bb_left'] -= 1
    gt_df['bb_top'] -= 1

    # Clean out unnecessary classes
    gt_df = gt_df[gt_df['label'].isin([1, 2, 7, 8, 12])].copy()  # Classes 7, 8, 12 are 'ambiguous' and are not penalized. Let's keep them for now.

    # Extra bbox values that will be used for id matching
    gt_df['bb_bot'] = (gt_df['bb_top'] + gt_df['bb_height']).values
    gt_df['bb_right'] = (gt_df['bb_left'] + gt_df['bb_width']).values

    return gt_df

def get_dancetrack_gt(seq_name, data_root_path):
    """
    Load MOT ground truth file
    """
    seq_path = osp.join(data_root_path, seq_name)  # Sequence path
    gt_file_path = osp.join(seq_path, "gt/gt.txt")  # Ground truth file

    # Read the gt file and assign the column names
    gt_df = pd.read_csv(gt_file_path, header=None)
    gt_df = gt_df[gt_df.columns[:len(GT_COL_NAMES)]]
    gt_df.columns = GT_COL_NAMES

    # Coordinates are 1 based
    gt_df['bb_left'] -= 1
    gt_df['bb_top'] -= 1

    # Clean out unnecessary classes
    gt_df = gt_df[gt_df['label'].isin([1, 2, 7, 8, 12])].copy()  # Classes 7, 8, 12 are 'ambiguous' and are not penalized. Let's keep them for now.

    # Extra bbox values that will be used for id matching
    gt_df['bb_bot'] = (gt_df['bb_top'] + gt_df['bb_height']).values
    gt_df['bb_right'] = (gt_df['bb_left'] + gt_df['bb_width']).values

    return gt_df