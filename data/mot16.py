import gzip
import os
import pickle
import random

import numpy as np
import torch

from collections import defaultdict
import torch.nn.functional as F
from torch_geometric.data import Data


def compute_edge_features(
        edge_ixs,
        node_frames,
        node_bboxes,
        node_feets,
        past_reids,
        now_reids,
        fps,
        speed_ref_bboxes=None,
        speed_directions=None,
        edge_feats_to_use=(
                'secs_time_dists', 'norm_feet_x_dists', 'norm_feet_y_dists',
                'bb_height_dists', 'bb_width_dists', 'emb_dists',
                'speed_dir_consistency'
        )
):
    if edge_ixs.numel() > 0:
        row, col = edge_ixs
        device = edge_ixs.device
        epsilon = 1e-8

        past_frames, now_frames = node_frames
        past_bboxes, now_bboxes = node_bboxes
        past_feets, now_feets = node_feets

        past_times = (past_frames / fps).float().reshape(-1)
        now_times = (now_frames / fps).float().reshape(-1)
        time_dists_sec = now_times[col] - past_times[row]

        past_bb_width = (past_bboxes[:, 2] - past_bboxes[:, 0]).clamp(min=epsilon)
        past_bb_height = (past_bboxes[:, 3] - past_bboxes[:, 1]).clamp(min=epsilon)
        now_bb_width = (now_bboxes[:, 2] - now_bboxes[:, 0]).clamp(min=epsilon)
        now_bb_height = (now_bboxes[:, 3] - now_bboxes[:, 1]).clamp(min=epsilon)

        past_feet_x = past_feets[:, 0]
        past_feet_y = past_feets[:, 1]
        now_feet_x = now_feets[:, 0]
        now_feet_y = now_feets[:, 1]
        feet_dx = now_feet_x[col] - past_feet_x[row]
        feet_dy = now_feet_y[col] - past_feet_y[row]
        mean_bb_heights = (past_bb_height[row] + now_bb_height[col]) / 2
        norm_feet_x_dists = feet_dx / mean_bb_heights
        norm_feet_y_dists = feet_dy / mean_bb_heights

        height_ratio = (now_bb_height[col] / past_bb_height[row]).clamp(min=epsilon)
        width_ratio = (now_bb_width[col] / past_bb_width[row]).clamp(min=epsilon)
        bb_height_dists = torch.log(height_ratio)
        bb_width_dists = torch.log(width_ratio)

        def reid_cosine_dist(feat1, feat2):
            feat1_norm = F.normalize(feat1, p=2, dim=1)
            feat2_norm = F.normalize(feat2, p=2, dim=1)
            similarity = (feat1_norm * feat2_norm).sum(dim=1, keepdim=True)
            return similarity

        emb_dists = []
        batch_size = 50000
        for i in range(0, edge_ixs.shape[1], batch_size):
            batch_row = edge_ixs[0][i:i + batch_size]
            batch_col = edge_ixs[1][i:i + batch_size]

            past_feat = past_reids[batch_row]
            now_feat = now_reids[batch_col]
            dist = reid_cosine_dist(past_feat, now_feat)
            emb_dists.append(dist.view(-1))
        emb_dists = torch.cat(emb_dists, dim=0) if emb_dists else torch.tensor([], device=device)

        speed_dir_consistency = torch.zeros(edge_ixs.shape[1], device=device)
        if speed_ref_bboxes is not None and speed_directions is not None:
            past_ref_bboxes = speed_ref_bboxes
            past_speed_dirs = speed_directions

            past_ref_cx = (past_ref_bboxes[:, 0] + past_ref_bboxes[:, 2]) / 2.0
            past_ref_cy = (past_ref_bboxes[:, 1] + past_ref_bboxes[:, 3]) / 2.0
            now_cx = (now_bboxes[:, 0] + now_bboxes[:, 2]) / 2.0
            now_cy = (now_bboxes[:, 1] + now_bboxes[:, 3]) / 2.0
            dy = now_cy[col] - past_ref_cy[row]
            dx = now_cx[col] - past_ref_cx[row]

            new_dir_norm = torch.sqrt(dx**2 + dy**2) + epsilon
            Y = dy / new_dir_norm
            X = dx / new_dir_norm
            Y = Y.unsqueeze(1)
            X = X.unsqueeze(1)

            inertia_Y = past_speed_dirs[row, 0].unsqueeze(1)
            inertia_X = past_speed_dirs[row, 1].unsqueeze(1)
            diff_angle_cos = (inertia_X * X + inertia_Y * Y).squeeze(1)
            diff_angle_cos = torch.clamp(diff_angle_cos, min=-1.0, max=1.0)
            diff_angle = torch.arccos(diff_angle_cos)
            speed_dir_consistency = (np.pi / 2.0 - torch.abs(diff_angle)) / np.pi

        edge_feats_dict = {
            'secs_time_dists': time_dists_sec,
            'norm_feet_x_dists': norm_feet_x_dists,
            'norm_feet_y_dists': norm_feet_y_dists,
            'bb_height_dists': bb_height_dists,
            'bb_width_dists': bb_width_dists,
            'emb_dists': emb_dists,
            'speed_dir_consistency': speed_dir_consistency
        }


        edge_feats = []
        for feat_name in edge_feats_to_use:
            edge_feats.append(edge_feats_dict[feat_name])
        edge_feats = torch.stack(edge_feats, dim=1)

    else:
        dtype = past_reids.dtype if past_reids.numel() > 0 else torch.float32
        edge_feats = torch.zeros((0, len(edge_feats_to_use)), dtype=dtype, device=edge_ixs.device)

    return edge_feats


class MOT16(torch.utils.data.Dataset):
    def __init__(self, split, args):
        super(MOT16, self).__init__()

        assert split == "train", f"Split {split} is NOT supported."
        self.args = args
        self.det_thresh = 0.6
        depth_out_dim = 10
        self.mot16_seqs_dir = os.path.join("datasets", "MOT16", split)
        if self.args.use_detection:
            self.mot16_gts_dir = "datasets/MOT16/detections/assigned"
            separator = ","
            pkl_path = "datasets/MOT16/reid_features_assigned_v3.pkl"
        else:
            self.mot16_gts_dir = os.path.join("datasets", "MOT16", "gts_new", split)
            separator = " "
            pkl_path = "/MOT16/reid_features.pkl"
        self.seq_fps = {
            "MOT16-02": 30,
            "MOT16-04": 30,
            "MOT16-05": 14,
            "MOT16-09": 30,
            "MOT16-10": 30,
            "MOT16-11": 30,
            "MOT16-13": 25,
        }
        depth_pkl_path = f"/datasets/MOT16/depth_features_assigned_v2_dim{depth_out_dim}.pkl"
        self.seq_size = {
            "MOT16-05": (640, 480),
            "MOT16-02": (1980, 1080),
            "MOT16-04": (1980, 1080),
            "MOT16-09": (1980, 1080),
            "MOT16-10": (1980, 1080),
            "MOT16-11": (1980, 1080),
            "MOT16-13": (1980, 1080),
        }

        self.history_range = (30, 60)
        self.frame_drop_range = (0.05, 0.20)

        self.mot16_gts = defaultdict(lambda: defaultdict(list))
        self.mot16_seq_names = [seq for seq in os.listdir(self.mot16_seqs_dir)]

        for vid in self.mot16_seq_names:
            if not self.args.use_detection:
                mot16_gts_dir = os.path.join(self.mot16_gts_dir, vid, "img1")
            else:
                mot16_gts_dir = os.path.join(self.mot16_gts_dir, vid)
            mot16_gt_paths = [os.path.join(mot16_gts_dir, filename) for filename in os.listdir(mot16_gts_dir)]
            for mot16_gt_path in mot16_gt_paths:
                for line in open(mot16_gt_path):
                    _, i, x, y, w, h, v = line.strip("\n").split(separator)
                    i, x, y, w, h, v = map(float, (i, x, y, w, h, v))
                    i, x, y, w, h = map(float, (i, x, y, w, h))
                    t = int(mot16_gt_path.split("/")[-1].split(".")[0])
                    self.mot16_gts[vid][t].append([i, x, y, w, h, v])
        self.sample_incoming_frame_paths = []
        for vid in self.mot16_gts.keys():
            t_min = min(self.mot16_gts[vid].keys())
            t_max = max(self.mot16_gts[vid].keys())

            for t in range(t_min + 1, t_max + 1):
                self.sample_incoming_frame_paths.append(
                    os.path.join(
                        self.mot16_seqs_dir,
                        vid,
                        "img1",
                        str(t).zfill(6) + ".jpg"
                    )
                )

        with open(pkl_path, 'rb') as f:
            self.reid_features = pickle.load(f)

        with open(depth_pkl_path, "rb") as f:
            self.depth_features = pickle.load(f)

    def __len__(self):
        assert self.sample_incoming_frame_paths is not None, "No data."
        return len(self.sample_incoming_frame_paths)

    def _extract_node_feature(self, node, is_past: bool):
        if is_past:
            reid_feat = node["ema_reid_feat"]
        else:
            reid_feat = node["reid_feat"]
        return reid_feat

    def __getitem__(self, item):
        frame_path = self.sample_incoming_frame_paths[item]
        parts = frame_path.split("/")
        vid = parts[-3]
        fps = self.seq_fps.get(vid, 30)

        raw_past_tracks = self.sample_past_tracks(frame_path=frame_path)
        V_now_raw = self._extract_V_now(frame_path=frame_path)
        V_past = self._aggregate_V_past(raw_past_tracks, int(parts[-1].split('.')[0]))

        V_now = []
        W, H = self.seq_size[vid]

        for obj_id, track_list in V_now_raw.items():
            track_item = track_list[0]
            bbox = torch.as_tensor(track_item["bbox"], dtype=torch.float32)
            reid_feat = torch.as_tensor(track_item["reid"], dtype=torch.float32).squeeze(0)
            if self.args.l2_norm_reid:
                reid_feat = F.normalize(reid_feat, p=2, dim=0)
            frame_id = track_item["frame_id"]
            depth_vector = track_item["depth_vector"]

            feet_x = (bbox[0] + bbox[2]) / 2.0
            feet_y = bbox[3]
            bb_height = bbox[3] - bbox[1]
            bb_width = bbox[2] - bbox[0]

            V_now.append({
                "obj_id": obj_id,
                "frame_id": frame_id,
                "bbox": bbox,
                "reid_feat": reid_feat,
                "depth_vector": depth_vector,
                "feet_x": feet_x,
                "feet_y": feet_y,
                "bb_height": bb_height,
                "bb_width": bb_width,
                "img_H": H,
                "img_W": W
            })

        edges_meta, edge_ixs_local = self._build_bipartite_edges(V_past, V_now)
        edge_feats = self.build_edge_features_from_nodes(V_past, V_now, edge_ixs_local, fps=fps)

        edge_labels = []
        for edge_meta in edges_meta:
            past_obj_id = edge_meta["past_obj_id"]
            now_obj_id = edge_meta["now_obj_id"]
            label = 1 if past_obj_id == now_obj_id else 0
            edge_labels.append(label)
        if edge_labels:
            edge_label = torch.tensor(edge_labels, dtype=torch.float32)
        else:
            edge_label = torch.zeros((0,), dtype=torch.float32)

        node_feats = []
        node_depth_feats = []
        node_frame_ids = []
        for past_node in V_past:
            past_feat = self._extract_node_feature(past_node, is_past=True)
            node_feats.append(past_feat)
            node_depth_feats.append(past_node["depth_vector"].squeeze())
            node_frame_ids.append(past_node["frame_ids"][-1])
        for now_node in V_now:
            now_feat = self._extract_node_feature(now_node, is_past=False)
            node_feats.append(now_feat)
            node_depth_feats.append(now_node["depth_vector"].squeeze())
            node_frame_ids.append(now_node["frame_id"])
        if len(node_feats) > 0:
            x = torch.stack(node_feats, dim=0)
        else:
            x = torch.zeros((0, 1024), dtype=torch.float32)
        if len(node_depth_feats) > 0:
            x_depth = torch.stack(node_depth_feats, dim=0)
        else:
            x_depth = torch.zeros((0, 10), dtype=torch.float32)

        frame_id = torch.tensor(node_frame_ids, dtype=torch.long) if len(node_frame_ids) > 0 else torch.zeros((0,),
                                                                                                       dtype=torch.long)
        if frame_id.numel() > 0:
            frame_id = frame_id - frame_id.min()
        num_past = len(V_past)
        edge_ixs_global = edge_ixs_local.clone()
        if edge_ixs_global.numel() > 0:
            edge_ixs_global[1, :] += num_past

        data = Data(
            x=x,
            edge_index=edge_ixs_global,
            edge_attr=edge_feats,
            edge_label=edge_label,
            x_depth=x_depth,
            frame_id=frame_id,
            num_prev_nodes=torch.tensor(num_past, dtype=torch.long),
        )

        return data

    def linear_interp_1d(self, xs, fp):
        x0 = torch.floor(xs).long()
        x1 = torch.clamp(x0 + 1, max=len(fp) - 1)
        wa = x1.float() - xs
        wb = xs - x0.float()
        return fp[x0] * wa + fp[x1] * wb

    def extract_dim_torch(self, depth_map, boxes, out_dim=10, trim_ratio=0.1):
        H, W = depth_map.shape
        N = boxes.shape[0]
        depth_map = depth_map.clone()
        mean_depths = torch.full((N,), -float('inf'), dtype=depth_map.dtype, device=boxes.device)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.int().tolist()
            x1 = max(0, min(x1, W - 1))
            x2 = max(0, min(x2, W - 1))
            y1 = max(0, min(y1, H - 1))
            y2 = max(0, min(y2, H - 1))
            if x2 > x1 and y2 > y1:
                mean_depths[i] = depth_map[y1:y2, x1:x2].mean()

        order = torch.argsort(mean_depths, descending=True)
        boxes_sorted = boxes[order]

        depth_features = torch.zeros((N, out_dim), dtype=depth_map.dtype, device=depth_map.device)

        for idx_in_order, box in enumerate(boxes_sorted):
            x1, y1, x2, y2 = box.int()

            x1 = max(0, min(x1.item(), W - 1))
            x2 = max(0, min(x2.item(), W - 1))
            y1 = max(0, min(y1.item(), H - 1))
            y2 = max(0, min(y2.item(), H - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            row_depths = []
            for y in range(y1, y2):
                row = depth_map[y, x1:x2]
                if row.numel() > 0:
                    row_depths.append(torch.max(row).item())

            if len(row_depths) == 0:
                continue

            row_depths = torch.tensor(row_depths, device=depth_map.device, dtype=torch.float32)

            if trim_ratio > 0:
                lo = row_depths.quantile(trim_ratio)
                hi = row_depths.quantile(1 - trim_ratio)
                mask = (row_depths >= lo) & (row_depths <= hi)
                row_depths = row_depths[mask]

            L = len(row_depths)
            if L < 2:
                continue

            if L < out_dim:
                sampled = F.interpolate(
                    row_depths.view(1, 1, -1),
                    size=out_dim,
                    mode='linear',
                    align_corners=True
                ).view(-1)
            else:
                xs = torch.linspace(0, L - 1, out_dim, device=depth_map.device)
                sampled = self.linear_interp_1d(xs, row_depths)

            original_idx = order[idx_in_order]
            depth_features[original_idx] = sampled

            depth_map[y1:y2, x1:x2] = 0.0

        return depth_features

    def compute_giou_matrix(self, bboxes1, bboxes2):
        N = bboxes1.shape[0]
        M = bboxes2.shape[0]

        bboxes1_expand = bboxes1.unsqueeze(1).expand(N, M, 4)
        bboxes2_expand = bboxes2.unsqueeze(0).expand(N, M, 4)

        inter_x1 = torch.max(bboxes1_expand[..., 0], bboxes2_expand[..., 0])
        inter_y1 = torch.max(bboxes1_expand[..., 1], bboxes2_expand[..., 1])
        inter_x2 = torch.min(bboxes1_expand[..., 2], bboxes2_expand[..., 2])
        inter_y2 = torch.min(bboxes1_expand[..., 3], bboxes2_expand[..., 3])
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        area1 = (bboxes1_expand[..., 2] - bboxes1_expand[..., 0]) * (bboxes1_expand[..., 3] - bboxes1_expand[..., 1])
        area2 = (bboxes2_expand[..., 2] - bboxes2_expand[..., 0]) * (bboxes2_expand[..., 3] - bboxes2_expand[..., 1])
        union_area = area1 + area2 - inter_area + 1e-6  # 避免除0

        iou = inter_area / union_area
        enclose_x1 = torch.min(bboxes1_expand[..., 0], bboxes2_expand[..., 0])
        enclose_y1 = torch.min(bboxes1_expand[..., 1], bboxes2_expand[..., 1])
        enclose_x2 = torch.max(bboxes1_expand[..., 2], bboxes2_expand[..., 2])
        enclose_y2 = torch.max(bboxes1_expand[..., 3], bboxes2_expand[..., 3])
        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1) + 1e-6
        giou = iou - (enclose_area - union_area) / enclose_area

        return giou

    def sample_past_tracks(self, frame_path):
        past_tracks = defaultdict(list)

        try:
            parts = frame_path.split("/")
            if len(parts) < 4:
                return dict(past_tracks)
            vid = parts[-3]

            t_star = int(parts[-1].split(".")[0])
        except (IndexError, ValueError):
            return dict(past_tracks)

        if vid not in self.mot16_gts or len(self.mot16_gts[vid]) == 0:
            return dict(past_tracks)

        t_min = min(self.mot16_gts[vid].keys())

        sampled_history = random.randint(*self.history_range)
        k = max(t_min, t_star - sampled_history)
        candidate_frames = [
            t for t in range(k, t_star) if t in self.mot16_gts[vid]
        ]
        drop_ratio = random.uniform(*self.frame_drop_range)

        num_frames = len(candidate_frames)
        num_drop = int(drop_ratio * num_frames)

        if num_drop > 0:
            drop_frames = set(random.sample(candidate_frames, num_drop))
        else:
            drop_frames = set()

        for t in candidate_frames:
            if t in drop_frames:
                continue
            det = []
            ids = []
            conf = []
            for obj_id, x, y, w, h, v in self.mot16_gts[vid][t]:
                x1, y1 = float(x), float(y)
                x2, y2 = float(x + w), float(y + h)
                det.append([x1, y1, x2, y2])
                ids.append(obj_id)
                conf.append(v)

            det_np = np.array(det, dtype=np.float32)

            frame_key = f"{vid}:{t}"

            dets_embs = self.reid_features[frame_key]

            depth_vector = torch.as_tensor(self.depth_features[frame_key], dtype=torch.float32)

            for i, obj_id in enumerate(ids):
                past_tracks[obj_id].append({
                    "frame_id": t,
                    "bbox": det[i],
                    "reid": dets_embs[i:i + 1, :],
                    "depth_vector": depth_vector[i:i + 1, :],
                    "conf": conf[i]
                })

        past_tracks = {obj_id: track_list for obj_id, track_list in past_tracks.items()
                       if len(track_list) > 0}

        return past_tracks

    def _aggregate_V_past(self, raw_past_tracks, t_now):
        V_past = []

        for obj_id, track_list in raw_past_tracks.items():
            track_list_sorted = sorted(track_list, key=lambda x: x["frame_id"])
            track_frame_map = {item["frame_id"]: item for item in track_list_sorted}
            track_length = len(track_list_sorted)
            frame_ids = [item["frame_id"] for item in track_list_sorted]
            latest_track = track_list_sorted[-1]
            t_latest = latest_track["frame_id"]

            latest_bbox = torch.as_tensor(latest_track["bbox"], dtype=torch.float32)
            latest_cx = (latest_bbox[0] + latest_bbox[2]) / 2.0
            latest_cy = (latest_bbox[1] + latest_bbox[3]) / 2.0
            latest_center = torch.tensor([latest_cx, latest_cy], dtype=torch.float32)
            latest_w = latest_bbox[2] - latest_bbox[0]
            latest_h = latest_bbox[3] - latest_bbox[1]
            latest_area = latest_w * latest_h

            last_feet_x = (latest_bbox[0] + latest_bbox[2]) / 2
            last_feet_y = latest_bbox[3]
            last_feet = torch.tensor([last_feet_x, last_feet_y], dtype=torch.float32)

            speed_direction_vec = torch.zeros(2, dtype=torch.float32)
            candidate_frames = [t_latest - 3, t_latest - 2, t_latest - 1]
            paired_frame = None
            for t_candidate in candidate_frames:
                if t_candidate in track_frame_map:
                    paired_frame = t_candidate
                    break

            if paired_frame is None and track_length >= 2:
                paired_frame = track_list_sorted[-2]["frame_id"]

            if paired_frame is not None:
                paired_track = track_frame_map[paired_frame]
                paired_bbox = torch.as_tensor(paired_track["bbox"], dtype=torch.float32)
                paired_cx = (paired_bbox[0] + paired_bbox[2]) / 2.0
                paired_cy = (paired_bbox[1] + paired_bbox[3]) / 2.0

                dy = latest_cy - paired_cy
                dx = latest_cx - paired_cx
                norm = np.sqrt(dy ** 2 + dx ** 2) + 1e-6
                dir_y = dy / norm
                dir_x = dx / norm
                speed_direction_vec = torch.tensor([dir_y, dir_x], dtype=torch.float32)

            speed_ref_frame = None
            for t in [t_now - 3, t_now - 2, t_now - 1]:
                if t in track_frame_map:
                    speed_ref_frame = t
                    break
            if speed_ref_frame is None:
                speed_ref_frame = t_latest
            speed_ref_bbox = torch.as_tensor(track_frame_map[speed_ref_frame]["bbox"], dtype=torch.float32)

            ema_reid_feat = None
            for track_item in track_list_sorted:
                curr_reid = torch.as_tensor(track_item["reid"], dtype=torch.float32).squeeze(0)
                if self.args.l2_norm_reid :
                    curr_reid = F.normalize(curr_reid, p=2, dim=0)
                if self.args.emb_alpha < 0.1:
                    ema_reid_feat = curr_reid
                else:
                    if ema_reid_feat is None:
                        ema_reid_feat = curr_reid
                    else:
                        conf = track_item["conf"]
                        trust = (conf - self.det_thresh) / (1 - self.det_thresh)
                        af = self.args.emb_alpha
                        ema_alpha = af + (1 - af) * (1 - trust)
                        ema_reid_feat = ema_alpha * ema_reid_feat + (1 - ema_alpha) * curr_reid
                        if self.args.l2_norm_reid:
                            ema_reid_feat = F.normalize(ema_reid_feat, p=2, dim=0)


            if ema_reid_feat is None:
                ema_reid_feat = torch.as_tensor(latest_track["reid"], dtype=torch.float32).squeeze(0)
                if self.args.l2_norm_reid:
                    ema_reid_feat = F.normalize(ema_reid_feat, p=2, dim=0)

            depth_vector = torch.as_tensor(latest_track["depth_vector"], dtype=torch.float32)

            V_past.append({
                "obj_id": obj_id,
                "frame_ids": frame_ids,
                "latest_bbox": latest_bbox,
                "latest_center": latest_center,
                "latest_area": latest_area,
                "last_feet_x": last_feet_x,
                "last_feet_y": last_feet_y,
                "last_feet": last_feet,
                "speed_direction": speed_direction_vec,
                "ema_reid_feat": ema_reid_feat,
                "depth_vector": depth_vector,
                "speed_ref_bbox": speed_ref_bbox
            })

        return V_past

    def _extract_V_now(self, frame_path):
        now_node = defaultdict(list)
        try:
            parts = frame_path.split("/")
            vid = parts[-3]
            t = int(parts[-1].split(".")[0])

            det = []
            ids = []
            if vid in self.mot16_gts and t in self.mot16_gts[vid]:
                for obj_id, x, y, w, h, _ in self.mot16_gts[vid][t]:
                    x1, y1 = float(x), float(y)
                    x2, y2 = float(x + w), float(y + h)
                    det.append([x1, y1, x2, y2])
                    ids.append(obj_id)

            if len(det) == 0:
                return now_node
            det_np = np.array(det, dtype=np.float32)

            frame_key = f"{vid}:{t}"

            dets_embs = self.reid_features[frame_key]

            depth_vector = torch.as_tensor(self.depth_features[frame_key], dtype=torch.float32)

            for i, obj_id in enumerate(ids):
                now_node[obj_id].append({
                    "frame_id": t,
                    "bbox": det[i],
                    "reid": dets_embs[i:i + 1, :],
                    "depth_vector": depth_vector[i:i + 1, :]
                })
            now_node = {k: v for k, v in now_node.items() if len(v) > 0}
        except Exception as e:
            print(f"Extract V_now error: {e}")
        return now_node

    def _build_bipartite_edges(self, V_past, V_now):

        if len(V_past) == 0 or len(V_now) == 0:
            return [], torch.zeros((2, 0), dtype=torch.long)

        past_bboxes = torch.stack([node["latest_bbox"] for node in V_past])
        past_reids = torch.stack([node["ema_reid_feat"] for node in V_past])

        now_bboxes = torch.stack([node["bbox"] for node in V_now])
        now_reids = torch.stack([node["reid_feat"] for node in V_now])

        giou_matrix = self.compute_giou_matrix(past_bboxes, now_bboxes)

        past_reids_norm = F.normalize(past_reids, p=2, dim=1)
        now_reids_norm = F.normalize(now_reids, p=2, dim=1)
        reid_sim = torch.mm(past_reids_norm, now_reids_norm.T)
        combined_sim = 0.05 * giou_matrix + 0.95 * reid_sim

        edge_ixs_list = []
        edges = []
        top_k = 10
        for past_idx in range(len(V_past)):
            sims = combined_sim[past_idx]
            top_k_ixs = torch.topk(sims, k=min(top_k, len(sims)), largest=True).indices
            for now_idx in top_k_ixs:
                edge_ixs_list.append([past_idx, now_idx])
                edges.append({
                    "past_obj_id": V_past[past_idx]["obj_id"],
                    "now_obj_id": V_now[now_idx]["obj_id"],
                    "past_idx": past_idx,
                    "now_idx": now_idx
                })

        edge_ixs = torch.tensor(edge_ixs_list, dtype=torch.long).T if edge_ixs_list else torch.zeros((2, 0),
                                                                                                     dtype=torch.long)
        return edges, edge_ixs

    def build_edge_features_from_nodes(self, V_past, V_now, edge_ixs, fps):
        if len(V_past) == 0 or len(V_now) == 0 or edge_ixs.numel() == 0:
            return torch.zeros((0, 7), dtype=torch.float32, device=edge_ixs.device if edge_ixs.numel() > 0 else 'cpu')

        past_frames = torch.tensor([node["frame_ids"][-1] for node in V_past], dtype=torch.float32)
        past_bboxes = torch.stack([node["latest_bbox"] for node in V_past])
        past_feets = torch.stack([torch.tensor([node["last_feet_x"], node["last_feet_y"]]) for node in V_past])
        past_reids = torch.stack([node["ema_reid_feat"] for node in V_past])
        speed_ref_bboxes = torch.stack([node["speed_ref_bbox"] for node in V_past])
        speed_directions = torch.stack([node["speed_direction"] for node in V_past])

        now_frames = torch.tensor([node["frame_id"] for node in V_now], dtype=torch.float32)
        now_bboxes = torch.stack([node["bbox"] for node in V_now])
        now_feets = torch.stack([torch.tensor([node["feet_x"], node["feet_y"]]) for node in V_now])
        now_reids = torch.stack([node["reid_feat"] for node in V_now])

        node_frames = (past_frames, now_frames)
        node_bboxes = (past_bboxes, now_bboxes)
        node_feets = (past_feets, now_feets)

        edge_feats = compute_edge_features(
            edge_ixs,
            node_frames,
            node_bboxes,
            node_feets,
            past_reids=past_reids,
            now_reids=now_reids,
            fps=fps,
            speed_ref_bboxes=speed_ref_bboxes,
            speed_directions=speed_directions
        )

        return edge_feats
    def build_depth_features_pkl(
            self,
            out_dim=10,
            trim_ratio=0.1,
            out_dir="/home/chenjiazhou/datasets/MOT16",
            overwrite=False,
            save_as_numpy=False,
    ):

        os.makedirs(out_dir, exist_ok=True)

        if self.args.use_detection:
            save_name = f"depth_features_assigned_v2_dim{out_dim}.pkl"
        else:
            save_name = f"depth_features_dim{out_dim}.pkl"

        save_path = os.path.join(out_dir, save_name)

        if os.path.exists(save_path) and not overwrite:
            print(f"[Skip] depth pkl already exists: {save_path}")
            return save_path

        depth_features = {}

        total_frames = 0
        saved_frames = 0
        skipped_no_reid = 0
        skipped_no_det = 0
        skipped_bad_depth = 0

        for vid in sorted(self.mot16_gts.keys()):
            frame_ids = sorted(self.mot16_gts[vid].keys())

            for t in frame_ids:
                total_frames += 1
                frame_key = f"{vid}:{t}"


                det = []
                for obj_id, x, y, w, h, _ in self.mot16_gts[vid][t]:
                    x1, y1 = float(x), float(y)
                    x2, y2 = float(x + w), float(y + h)
                    det.append([x1, y1, x2, y2])

                if len(det) == 0:
                    skipped_no_det += 1
                    continue

                det_np = np.asarray(det, dtype=np.float32)
                depth_path = os.path.join(
                    "datasets/MOT17/DepthAnythingV2",
                    vid.replace("MOT16", "MOT17") + "-SDP",
                    str(t).zfill(6) + ".pt"
                )


                depth_data = torch.load(depth_path)

                depth = torch.from_numpy(depth_data).float()

                valid_mask = depth > 0.01
                if valid_mask.sum() == 0:
                    skipped_bad_depth += 1
                    depth_vector = torch.zeros((det_np.shape[0], out_dim), dtype=torch.float32)
                else:
                    d_near = depth[valid_mask].min().item()
                    d_far = depth[valid_mask].max().item()

                    if abs(d_far - d_near) < 1e-6:
                        skipped_bad_depth += 1
                        depth_vector = torch.zeros((det_np.shape[0], out_dim), dtype=torch.float32)
                    else:
                        depth = (depth - d_near) / (d_far - d_near)
                        depth_vector = self.extract_dim_torch(
                            depth,
                            torch.from_numpy(det_np),
                            out_dim=out_dim,
                            trim_ratio=trim_ratio
                        )

                if save_as_numpy:
                    depth_features[frame_key] = depth_vector.cpu().numpy().astype(np.float32)
                else:
                    depth_features[frame_key] = depth_vector.cpu()

                saved_frames += 1

                if saved_frames % 500 == 0:
                    print(f"[Depth PKL] saved {saved_frames} frames... latest={frame_key}")

        with open(save_path, "wb") as f:
            pickle.dump(depth_features, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("=" * 80)
        print(f"[Done] depth pkl saved to: {save_path}")
        print(f"total_frames      : {total_frames}")
        print(f"saved_frames      : {saved_frames}")
        print(f"skipped_no_reid   : {skipped_no_reid}")
        print(f"skipped_no_det    : {skipped_no_det}")
        print(f"skipped_bad_depth : {skipped_bad_depth}")
        print("=" * 80)

        return save_path
