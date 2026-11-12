from __future__ import print_function

import os
import math
import numpy as np
import torch
import torch.nn.functional as F

from data.reid.embedding import EmbeddingComputer
from models.monotracker.model.new_MPN_subgraph import MOTMPNet
from models.monotracker.model.model_util import read_yaml_to_dict

def linear_assignment(cost_matrix):
    try:
        import lap

        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment

        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def iou_batch(bboxes1, bboxes2):
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        - wh
    )
    return o


def linear_interp_1d(xs, fp):
    x0 = torch.floor(xs).long()
    x1 = torch.clamp(x0 + 1, max=len(fp) - 1)
    wa = x1.float() - xs
    wb = xs - x0.float()
    return fp[x0] * wa + fp[x1] * wb


def normalize_depth_map(depth):
    if not isinstance(depth, torch.Tensor):
        depth = torch.as_tensor(depth)
    depth = depth.float()
    valid = depth > 0.01
    if valid.sum() == 0:
        return torch.zeros_like(depth)
    d_min = depth[valid].min()
    d_max = depth[valid].max()
    if torch.abs(d_max - d_min) < 1e-6:
        return torch.zeros_like(depth)
    return (depth - d_min) / (d_max - d_min)


def extract_dim_torch(depth_map, boxes, out_dim=10, trim_ratio=0.1):
    if not isinstance(depth_map, torch.Tensor):
        depth_map = torch.as_tensor(depth_map)
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.as_tensor(boxes)

    depth_map = depth_map.float().clone()
    boxes = boxes.float()
    H, W = depth_map.shape[-2], depth_map.shape[-1]
    N = boxes.shape[0]

    depth_features = torch.zeros((N, out_dim), dtype=torch.float32, device=depth_map.device)
    if N == 0:
        return depth_features.cpu()

    order = torch.argsort(boxes[:, 3], descending=True)
    boxes_sorted = boxes[order]

    for idx_in_order, box in enumerate(boxes_sorted):
        x1, y1, x2, y2 = box.int()
        x1 = max(0, min(int(x1.item()), W - 1))
        x2 = max(0, min(int(x2.item()), W))
        y1 = max(0, min(int(y1.item()), H - 1))
        y2 = max(0, min(int(y2.item()), H))
        if x2 <= x1 or y2 <= y1:
            continue

        row_depths = []
        for y in range(y1, y2):
            row = depth_map[y, x1:x2]
            if row.numel() > 0:
                # Keep the same rule as the training depth-vector code.
                row_depths.append(torch.max(row).item())

        if len(row_depths) < 2:
            continue

        row_depths = torch.tensor(row_depths, device=depth_map.device, dtype=torch.float32)
        if trim_ratio > 0 and row_depths.numel() >= 4:
            lo = row_depths.quantile(trim_ratio)
            hi = row_depths.quantile(1 - trim_ratio)
            row_depths = row_depths[(row_depths >= lo) & (row_depths <= hi)]

        L = row_depths.numel()
        if L < 2:
            continue
        if L < out_dim:
            sampled = F.interpolate(
                row_depths.view(1, 1, -1),
                size=out_dim,
                mode="linear",
                align_corners=True,
            ).view(-1)
        else:
            xs = torch.linspace(0, L - 1, out_dim, device=depth_map.device)
            sampled = linear_interp_1d(xs, row_depths)

        original_idx = order[idx_in_order]
        depth_features[original_idx] = sampled
        depth_map[y1:y2, x1:x2] = 0.0

    return depth_features.cpu()


def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return np.array([-1, -1, -1, -1, -1], dtype=np.float32)
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1], dtype=np.float32)
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h + 1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_bbox_to_z_new(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    return np.array([x, y, w, h]).reshape((4, 1))


def convert_x_to_bbox_new(x):
    x, y, w, h = x.reshape(-1)[:4]
    return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2]).reshape(1, 4)


def convert_x_to_bbox(x, score=None):
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
    return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))


def new_kf_process_noise(w, h, p=1 / 20, v=1 / 160):
    return np.diag(((p * w) ** 2, (p * h) ** 2, (p * w) ** 2, (p * h) ** 2,
                    (v * w) ** 2, (v * h) ** 2, (v * w) ** 2, (v * h) ** 2))


def new_kf_measurement_noise(w, h, m=1 / 20):
    return np.diag(((m * w) ** 2, (m * h) ** 2, (m * w) ** 2, (m * h) ** 2))


def compute_giou_matrix(bboxes1, bboxes2):
    if bboxes1.numel() == 0 or bboxes2.numel() == 0:
        return torch.zeros((bboxes1.shape[0], bboxes2.shape[0]), dtype=torch.float32)

    b1 = bboxes1[:, :4].float()
    b2 = bboxes2[:, :4].float()
    N, M = b1.shape[0], b2.shape[0]
    b1e = b1.unsqueeze(1).expand(N, M, 4)
    b2e = b2.unsqueeze(0).expand(N, M, 4)

    ix1 = torch.max(b1e[..., 0], b2e[..., 0])
    iy1 = torch.max(b1e[..., 1], b2e[..., 1])
    ix2 = torch.min(b1e[..., 2], b2e[..., 2])
    iy2 = torch.min(b1e[..., 3], b2e[..., 3])
    inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)

    area1 = torch.clamp(b1e[..., 2] - b1e[..., 0], min=0) * torch.clamp(b1e[..., 3] - b1e[..., 1], min=0)
    area2 = torch.clamp(b2e[..., 2] - b2e[..., 0], min=0) * torch.clamp(b2e[..., 3] - b2e[..., 1], min=0)
    union = area1 + area2 - inter + 1e-6
    iou = inter / union

    ex1 = torch.min(b1e[..., 0], b2e[..., 0])
    ey1 = torch.min(b1e[..., 1], b2e[..., 1])
    ex2 = torch.max(b1e[..., 2], b2e[..., 2])
    ey2 = torch.max(b1e[..., 3], b2e[..., 3])
    enclose = torch.clamp(ex2 - ex1, min=0) * torch.clamp(ey2 - ey1, min=0) + 1e-6
    return iou - (enclose - union) / enclose


def compute_edge_features(edge_ixs, node_frames, node_bboxes, node_feets,
                          past_reids, now_reids, fps,
                          speed_ref_bboxes=None, speed_directions=None):
    E = edge_ixs.shape[1]
    if E == 0:
        return torch.zeros((0, 7), dtype=torch.float32)

    row, col = edge_ixs.long()
    eps = 1e-8
    past_frames, now_frames = node_frames
    past_bboxes, now_bboxes = node_bboxes
    past_feets, now_feets = node_feets

    past_times = past_frames.float().view(-1) / float(fps)
    now_times = now_frames.float().view(-1) / float(fps)
    time_dists_sec = now_times[col] - past_times[row]

    past_w = (past_bboxes[:, 2] - past_bboxes[:, 0]).clamp(min=eps)
    past_h = (past_bboxes[:, 3] - past_bboxes[:, 1]).clamp(min=eps)
    now_w = (now_bboxes[:, 2] - now_bboxes[:, 0]).clamp(min=eps)
    now_h = (now_bboxes[:, 3] - now_bboxes[:, 1]).clamp(min=eps)

    feet_dx = now_feets[col, 0] - past_feets[row, 0]
    feet_dy = now_feets[col, 1] - past_feets[row, 1]
    mean_h = (past_h[row] + now_h[col]) / 2.0
    norm_feet_x_dists = feet_dx / mean_h
    norm_feet_y_dists = feet_dy / mean_h

    bb_height_dists = torch.log((now_h[col] / past_h[row]).clamp(min=eps))
    bb_width_dists = torch.log((now_w[col] / past_w[row]).clamp(min=eps))

    past_feat = F.normalize(past_reids[row].float(), p=2, dim=1)
    now_feat = F.normalize(now_reids[col].float(), p=2, dim=1)
    emb_dists = (past_feat * now_feat).sum(dim=1)

    speed_dir_consistency = torch.zeros(E, dtype=torch.float32)
    if speed_ref_bboxes is not None and speed_directions is not None:
        past_ref = speed_ref_bboxes.float()
        past_dirs = speed_directions.float()
        ref_cx = (past_ref[:, 0] + past_ref[:, 2]) / 2.0
        ref_cy = (past_ref[:, 1] + past_ref[:, 3]) / 2.0
        now_cx = (now_bboxes[:, 0] + now_bboxes[:, 2]) / 2.0
        now_cy = (now_bboxes[:, 1] + now_bboxes[:, 3]) / 2.0
        dy = now_cy[col] - ref_cy[row]
        dx = now_cx[col] - ref_cx[row]
        norm = torch.sqrt(dx ** 2 + dy ** 2) + eps
        Y = dy / norm
        X = dx / norm
        inertia_Y = past_dirs[row, 0]
        inertia_X = past_dirs[row, 1]
        cos = torch.clamp(inertia_X * X + inertia_Y * Y, min=-1.0, max=1.0)
        angle = torch.arccos(cos)
        speed_dir_consistency = (math.pi / 2.0 - torch.abs(angle)) / math.pi

    return torch.stack([
        time_dists_sec,
        norm_feet_x_dists,
        norm_feet_y_dists,
        bb_height_dists,
        bb_width_dists,
        emb_dists,
        speed_dir_consistency,
    ], dim=1).float()

def speed_direction_batch(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = (dets[:, 0] + dets[:, 2]) / 2.0, (dets[:, 1] + dets[:, 3]) / 2.0
    CX2, CY2 = (tracks[:, 0] + tracks[:, 2]) / 2.0, (tracks[:, 1] + tracks[:, 3]) / 2.0
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx

class KalmanBoxTracker(object):
    count = 0

    def __init__(self, bbox, frame, depth, delta_t=3, orig=False, emb=None, alpha=0, new_kf=False):
        if not orig:
            from .kalmanfilter import KalmanFilterNew as KalmanFilter
        else:
            from filterpy.kalman import KalmanFilter

        self.new_kf = new_kf
        if new_kf:
            self.kf = KalmanFilter(dim_x=8, dim_z=4)
            self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                                  [0, 1, 0, 0, 0, 1, 0, 0],
                                  [0, 0, 1, 0, 0, 0, 1, 0],
                                  [0, 0, 0, 1, 0, 0, 0, 1],
                                  [0, 0, 0, 0, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 1, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 1]])
            self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 1, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 1, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 0, 0, 0]])
            _, _, w, h = convert_bbox_to_z_new(bbox).reshape(-1)
            self.kf.P = new_kf_process_noise(w, h)
            self.kf.P[:4, :4] *= 4
            self.kf.P[4:, 4:] *= 100
            self.bbox_to_z_func = convert_bbox_to_z_new
            self.x_to_bbox_func = convert_x_to_bbox_new
        else:
            self.kf = KalmanFilter(dim_x=7, dim_z=4)
            self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                                  [0, 1, 0, 0, 0, 1, 0],
                                  [0, 0, 1, 0, 0, 0, 1],
                                  [0, 0, 0, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0, 0],
                                  [0, 0, 0, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 0, 0, 1]])
            self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                                  [0, 1, 0, 0, 0, 0, 0],
                                  [0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 0, 0]])
            self.kf.R[2:, 2:] *= 10.0
            self.kf.P[4:, 4:] *= 1000.0
            self.kf.P *= 10.0
            self.kf.Q[-1, -1] *= 0.01
            self.kf.Q[4:, 4:] *= 0.01
            self.bbox_to_z_func = convert_bbox_to_z
            self.x_to_bbox_func = convert_x_to_bbox

        self.kf.x[:4] = self.bbox_to_z_func(bbox)
        self.conf = float(bbox[4]) if len(bbox) >= 5 else 1.0
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.last_observation = np.array([-1, -1, -1, -1, -1], dtype=np.float32)
        self.history_observations = []
        self.observations = dict()
        self.velocity = None
        self.delta_t = delta_t
        self.emb = emb
        self.last_emb = emb.copy()
        self.emb_bbox = bbox.copy()
        self.last_depth = torch.as_tensor(depth, dtype=torch.float32).view(1, -1) if depth is not None else None
        self.last_frame = int(frame) if frame is not None else 0
        self.frozen = False
        self.frame_observations = dict()

    def update(self, bbox, depth=None, frame=None):
        if bbox is not None:
            bbox = np.asarray(bbox, dtype=np.float32)
            self.frozen = False
            if self.last_observation.sum() >= 0:
                previous_box = None
                for dt in range(self.delta_t, 0, -1):
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                self.velocity = speed_direction(previous_box, bbox)

            self.last_observation = bbox.copy()
            self.observations[self.age] = bbox.copy()
            self.history_observations.append(bbox.copy())
            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            self.frame_observations[int(frame)] = bbox.copy()

            if self.new_kf:
                R = new_kf_measurement_noise(self.kf.x[2, 0], self.kf.x[3, 0])
                self.kf.update(self.bbox_to_z_func(bbox), R=R, new_kf=True)
            else:
                self.kf.update(self.bbox_to_z_func(bbox))

            self.emb_bbox = bbox.copy()
            if frame is not None:
                self.last_frame = int(frame)
            if depth is not None:
                self.last_depth = torch.as_tensor(depth, dtype=torch.float32).view(1, -1)
        else:
            self.kf.update(bbox, new_kf=self.new_kf)
            self.frozen = True

    def update_emb(self, emb, alpha=0.9):
        self.emb = alpha * self.emb + (1 - alpha) * emb
        self.emb /= np.linalg.norm(self.emb)
        self.last_emb=emb.copy()

    def get_emb(self):
        return self.emb

    def get_last_emb(self):
        return self.last_emb

    def apply_affine_correction(self, affine):
        m = affine[:, :2]
        t = affine[:, 2].reshape(2, 1)
        if self.last_observation.sum() > 0:
            ps = self.last_observation[:4].reshape(2, 2).T
            ps = m @ ps + t
            self.last_observation[:4] = ps.T.reshape(-1)
        for dt in range(self.delta_t, -1, -1):
            if self.age - dt in self.observations:
                ps = self.observations[self.age - dt][:4].reshape(2, 2).T
                ps = m @ ps + t
                self.observations[self.age - dt][:4] = ps.T.reshape(-1)
        self.kf.apply_affine_correction(m, t, self.new_kf)

    def predict(self):
        if self.new_kf:
            if self.kf.x[2] + self.kf.x[6] <= 0:
                self.kf.x[6] = 0
            if self.kf.x[3] + self.kf.x[7] <= 0:
                self.kf.x[7] = 0
            if self.frozen:
                self.kf.x[6] = self.kf.x[7] = 0
            Q = new_kf_process_noise(self.kf.x[2, 0], self.kf.x[3, 0])
        else:
            if (self.kf.x[6] + self.kf.x[2]) <= 0:
                self.kf.x[6] *= 0.0
            Q = None
        self.kf.predict(Q=Q)
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.x_to_bbox_func(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return self.x_to_bbox_func(self.kf.x)

    def mahalanobis(self, bbox):
        return self.kf.md_for_measurement(self.bbox_to_z_func(bbox))


ASSO_FUNCS = {
    "iou": iou_batch,
}


class OCSort(object):
    def __init__(self, det_thresh,depth_length=10, max_age=30, min_hits=3, iou_threshold=0.3, delta_t=3,
                 asso_func="iou", inertia=0.2, w_association_emb=0.75,
                 alpha_fixed_emb=0.95, aw_param=0.5, embedding_off=False,
                 cmc_off=True, aw_off=False, new_kf_off=True, grid_off=True,
                 gnn_score_thresh=0.1, top_k=10, model_yaml=None, model_ckpt=None,
                 device=None, dataset_name="mot17", **kwargs):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.depth_length = depth_length
        self.w_association_emb = w_association_emb
        self.alpha_fixed_emb = alpha_fixed_emb
        self.aw_param = aw_param
        self.embedding_off = embedding_off
        self.cmc_off = cmc_off
        self.aw_off = aw_off
        self.new_kf_off = new_kf_off
        self.grid_off = grid_off
        self.top_k = int(top_k)
        self.gnn_score_thresh = float(gnn_score_thresh)
        self.last_gnn_matrix = None
        self.last_edge_index_local = None
        KalmanBoxTracker.count = 0

        args = kwargs.get("args", None)
        test_dataset = getattr(args, "test_dataset", True) if args is not None else True
        self.embedder = EmbeddingComputer(dataset=dataset_name, test_dataset=test_dataset, grid_off=grid_off)

        if model_yaml is None and args is not None:
            model_yaml = getattr(args, "model_yaml", None)
        if model_yaml is None:
            model_yaml = "/cfg/mot17_model.yaml"

        if model_ckpt is None and args is not None:
            model_ckpt = getattr(args, "model_ckpt", None)
        if model_ckpt is None:
            model_ckpt = "results/MPNnet_11_train/10.pth"

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        cfg = read_yaml_to_dict(model_yaml)

        self.model = MOTMPNet(cfg["graph_model_params"]).to(self.device)

        if not os.path.exists(model_ckpt):
            raise FileNotFoundError(f"GNN checkpoint not found: {model_ckpt}")

        ckpt = torch.load(model_ckpt, map_location=self.device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
            print(f"[GNN] Loaded checkpoint: {model_ckpt}, epoch={ckpt.get('epoch', 'unknown')}")
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
            print(f"[GNN] Loaded checkpoint: {model_ckpt}")
        else:
            state = ckpt
            print(f"[GNN] Loaded raw state_dict: {model_ckpt}")

        clean_state = {k.replace("module.", ""): v for k, v in state.items()}
        self.model.load_state_dict(clean_state, strict=True)
        self.model.eval()

    @staticmethod
    def _as_flat_reid(feat):
        feat = np.asarray(feat, dtype=np.float32)
        return feat.reshape(-1)

    def _build_speed_matrix(self, dets, velocities, previous_obs):
        num_det = len(dets)
        num_trk = len(previous_obs)

        if num_det == 0 or num_trk == 0:
            return np.zeros((num_trk, num_det), dtype=np.float32)

        Y, X = speed_direction_batch(dets, previous_obs)  # shape: num_trk x num_det

        inertia_Y = velocities[:, 0]
        inertia_X = velocities[:, 1]

        inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
        inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)

        diff_angle_cos = inertia_X * X + inertia_Y * Y
        diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)

        diff_angle = np.arccos(diff_angle_cos)
        diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

        valid_mask = np.ones(previous_obs.shape[0], dtype=np.float32)
        valid_mask[np.where(previous_obs[:, 4] < 0)] = 0
        valid_mask = np.repeat(valid_mask[:, np.newaxis], Y.shape[1], axis=1)

        speed_matrix = valid_mask * diff_angle
        speed_matrix = speed_matrix * self.inertia

        det_scores = dets[:, -1]
        speed_matrix = speed_matrix * det_scores[np.newaxis, :]

        return speed_matrix.astype(np.float32)

    def _build_bipartite_edges(self, V_past, V_now):
        N, M = len(V_past), len(V_now)
        if N == 0 or M == 0:
            return torch.zeros((2, 0), dtype=torch.long)

        past_bboxes = torch.stack([n["latest_bbox"][:4] for n in V_past]).float()
        now_bboxes = torch.stack([n["bbox"][:4] for n in V_now]).float()
        past_reids = torch.stack([n["ema_reid_feat"] for n in V_past]).float()
        now_reids = torch.stack([n["reid_feat"] for n in V_now]).float()

        giou = compute_giou_matrix(past_bboxes, now_bboxes)
        reid_sim = F.normalize(past_reids, p=2, dim=1) @ F.normalize(now_reids, p=2, dim=1).t()
        combined_sim = 0.05 * giou + 0.95 * reid_sim

        edge_pairs = []
        for past_idx in range(N):
            k = min(self.top_k, M)
            top_idx = torch.topk(combined_sim[past_idx], k=k, largest=True).indices
            for now_idx in top_idx.tolist():
                edge_pairs.append([past_idx, now_idx])
        if len(edge_pairs) == 0:
            return torch.zeros((2, 0), dtype=torch.long)
        return torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()

    def _build_edge_features_from_nodes(self, V_past, V_now, edge_ixs_local, fps):
        if len(V_past) == 0 or len(V_now) == 0 or edge_ixs_local.numel() == 0:
            return torch.zeros((0, 7), dtype=torch.float32)

        past_frames = torch.tensor([n["frame_id"] for n in V_past], dtype=torch.float32)
        now_frames = torch.tensor([n["frame_id"] for n in V_now], dtype=torch.float32)
        past_bboxes = torch.stack([n["latest_bbox"][:4] for n in V_past]).float()
        now_bboxes = torch.stack([n["bbox"][:4] for n in V_now]).float()
        past_feets = torch.stack([n["last_feet"] for n in V_past]).float()
        now_feets = torch.stack([n["feet"] for n in V_now]).float()
        past_reids = torch.stack([n["ema_reid_feat"] for n in V_past]).float()
        now_reids = torch.stack([n["reid_feat"] for n in V_now]).float()
        speed_ref_bboxes = torch.stack([n["speed_ref_bbox"] for n in V_past]).float()
        speed_directions = torch.stack([n["speed_direction"] for n in V_past]).float()

        return compute_edge_features(
            edge_ixs_local,
            node_frames=(past_frames, now_frames),
            node_bboxes=(past_bboxes, now_bboxes),
            node_feets=(past_feets, now_feets),
            past_reids=past_reids,
            now_reids=now_reids,
            fps=fps,
            speed_ref_bboxes=speed_ref_bboxes,
            speed_directions=speed_directions,
        )

    def _tracker_bbox_for_graph(self, trk):
        if trk.last_observation is not None and np.asarray(trk.last_observation).sum() >= 0:
            return np.asarray(trk.last_observation, dtype=np.float32)
        return np.asarray(trk.emb_bbox, dtype=np.float32)

    def _speed_ref_bbox_for_graph(self, trk, frame):
        ref = np.asarray(k_previous_obs(trk.frame_observations, frame, self.delta_t), dtype=np.float32)
        if ref.shape[0] < 4 or ref[:4].sum() < 0:
            ref = np.asarray(trk.emb_bbox, dtype=np.float32)
        return ref[:4]

    def _build_graph_inputs(self, dets, dets_embs, dets_depth, frame, fps):
        V_past, V_now = [], []

        for trk in self.trackers:
            bbox_np = self._tracker_bbox_for_graph(trk)
            bbox = torch.as_tensor(bbox_np[:4], dtype=torch.float32)
            emb = torch.as_tensor(self._as_flat_reid(trk.get_emb()), dtype=torch.float32)
            depth_vec = torch.as_tensor(trk.last_depth, dtype=torch.float32).view(-1)
            feet = torch.tensor([(bbox[0] + bbox[2]) / 2.0, bbox[3]], dtype=torch.float32)
            speed_vec = torch.as_tensor(trk.velocity, dtype=torch.float32) if trk.velocity is not None else torch.zeros(2)
            speed_ref = torch.as_tensor(self._speed_ref_bbox_for_graph(trk, frame), dtype=torch.float32)
            V_past.append({
                "frame_id": int(trk.last_frame),
                "latest_bbox": bbox,
                "ema_reid_feat": emb,
                "depth_vector": depth_vec,
                "last_feet": feet,
                "speed_direction": speed_vec,
                "speed_ref_bbox": speed_ref,
            })

        for i in range(dets.shape[0]):
            bbox = torch.as_tensor(dets[i, :4], dtype=torch.float32)
            emb = torch.as_tensor(self._as_flat_reid(dets_embs[i]), dtype=torch.float32)
            depth_vec = torch.as_tensor(dets_depth[i], dtype=torch.float32).view(-1)
            feet = torch.tensor([(bbox[0] + bbox[2]) / 2.0, bbox[3]], dtype=torch.float32)
            V_now.append({
                "frame_id": int(frame),
                "bbox": bbox,
                "reid_feat": emb,
                "depth_vector": depth_vec,
                "feet": feet,
            })

        edge_ixs_local = self._build_bipartite_edges(V_past, V_now)
        edge_attr = self._build_edge_features_from_nodes(V_past, V_now, edge_ixs_local, fps=fps)

        node_feats = [n["ema_reid_feat"] for n in V_past] + [n["reid_feat"] for n in V_now]
        node_depth = [n["depth_vector"] for n in V_past] + [n["depth_vector"] for n in V_now]
        node_frames = [n["frame_id"] for n in V_past] + [n["frame_id"] for n in V_now]

        if len(node_feats) > 0:
            x = torch.stack(node_feats, dim=0).float()
            x_depth = torch.stack(node_depth, dim=0).float()
            frame_id = torch.tensor(node_frames, dtype=torch.long)
            frame_id = frame_id - frame_id.min()
        else:
            x = torch.zeros((0, 1024), dtype=torch.float32)
            x_depth = torch.zeros((0, 10), dtype=torch.float32)
            frame_id = torch.zeros((0,), dtype=torch.long)

        num_past = len(V_past)
        edge_ixs_global = edge_ixs_local.clone()
        if edge_ixs_global.numel() > 0:
            edge_ixs_global[1] += num_past

        return x, edge_ixs_global, edge_attr, x_depth, frame_id, torch.tensor(num_past, dtype=torch.long), edge_ixs_local

    def _run_gnn_association(self, dets, dets_embs, dets_depth, frame, fps, iou_matrix, speed_matrix, app_matrix):
        num_trk = len(self.trackers)
        num_det = len(dets)
        score_matrix = np.full((num_trk, num_det), -1e6, dtype=np.float32)
        score_matrix_test = np.full((num_trk, num_det), -1e6, dtype=np.float32)

        if num_trk == 0 or num_det == 0:
            self.last_gnn_matrix = score_matrix
            self.last_edge_index_local = torch.zeros((2, 0), dtype=torch.long)
            return np.empty((0, 2), dtype=int), np.arange(num_det), np.arange(num_trk), score_matrix

        x, edge_index, edge_attr, x_depth, frame_id, num_prev_nodes, edge_ixs_local = self._build_graph_inputs(
            dets=dets,
            dets_embs=dets_embs,
            dets_depth=dets_depth,
            frame=frame,
            fps=fps,
        )
        self.last_edge_index_local = edge_ixs_local

        if edge_ixs_local.numel() == 0:
            self.last_gnn_matrix = score_matrix
            return np.empty((0, 2), dtype=int), np.arange(num_det), np.arange(num_trk), score_matrix

        with torch.no_grad():
            outputs = self.model(
                x=x.to(self.device),
                edge_index=edge_index.to(self.device),
                edge_attr=edge_attr.to(self.device),
                x_depth=x_depth.to(self.device),
                frame_id=frame_id.to(self.device),
                num_prev_nodes=num_prev_nodes.to(self.device),
            )
            edge_start = outputs["classified_edges"][0].view(-1)
            edge_logits = outputs["classified_edges"][-1].view(-1)
            edge_scores = torch.sigmoid(edge_logits).detach().cpu()

        row = edge_ixs_local[0].long()
        col = edge_ixs_local[1].long()
        score_matrix[row.numpy(), col.numpy()] = edge_scores.numpy().astype(np.float32)
        score_matrix_test[row.numpy(), col.numpy()] =  torch.sigmoid(edge_start).detach().cpu().numpy().astype(np.float32)
        self.last_gnn_matrix = score_matrix
        final_matrix = score_matrix_test * self.w_association_emb + iou_matrix + speed_matrix



        assign = linear_assignment(-final_matrix)
        matched = []  # matched rows are [det_idx, trk_idx], same as original OCSort update logic.
        used_dets, used_trks = set(), set()
        for trk_idx, det_idx in assign:
            if trk_idx >= num_trk or det_idx >= num_det:
                continue
            if iou_matrix[trk_idx, det_idx] < self.iou_threshold:
                continue
            matched.append([det_idx, trk_idx])
            used_dets.add(det_idx)
            used_trks.add(trk_idx)

        unmatched_dets = np.array([i for i in range(num_det) if i not in used_dets], dtype=int)
        unmatched_trks = np.array([i for i in range(num_trk) if i not in used_trks], dtype=int)
        return np.asarray(matched, dtype=int), unmatched_dets, unmatched_trks, score_matrix

    def update(self, output_results, img_numpy, depth, frame, fps, r=None):
        self.frame_count += 1
        ret = []

        if output_results is None:
            dets = np.empty((0, 5), dtype=np.float32)
        else:
            if not isinstance(output_results, np.ndarray):
                output_results = output_results.detach().cpu().numpy()
            if output_results.shape[1] == 5:
                scores = output_results[:, 4]
                bboxes = output_results[:, :4]
            else:
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]
            dets = np.concatenate((bboxes, scores[:, None]), axis=1).astype(np.float32)
            dets = dets[scores > self.det_thresh]
        if r is not None:
            dets[:, :4] /= r
        depth_emb = depth.copy()
        depth_norm = normalize_depth_map(depth)
        dets_depth = extract_dim_torch(depth_norm, torch.as_tensor(dets[:, :4], dtype=torch.float32), out_dim=self.depth_length)

        if self.embedding_off:
            raise RuntimeError("GNN association requires ReID embeddings; do not enable embedding_off.")
        if dets.shape[0] > 0:
            dets_embs = self.embedder.compute_embedding(img_numpy, dets[:, :4], depth_emb, tag=None)
            dets_embs = np.asarray(dets_embs, dtype=np.float32)
        else:
            dets_embs = np.empty((0, 1024), dtype=np.float32)

        trust = (dets[:, 4] - self.det_thresh) / (1 - self.det_thresh + 1e-12) if dets.shape[0] > 0 else np.empty((0,))
        dets_alpha = self.alpha_fixed_emb + (1 - self.alpha_fixed_emb) * (1 - trust)

        trks = np.zeros((len(self.trackers), 5), dtype=np.float32)
        trk_embs = []
        to_del = []
        for t in range(len(self.trackers)):
            pos = self.trackers[t].predict()[0]
            trks[t] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
            else:
                trk_embs.append(self.trackers[t].get_emb())
        for t in reversed(to_del):
            self.trackers.pop(t)
        if len(to_del) > 0:
            trks = np.delete(trks, to_del, axis=0)
        trk_embs = np.array(trk_embs)

        velocities = np.array([
            trk.velocity if trk.velocity is not None else np.array((0, 0), dtype=np.float32)
            for trk in self.trackers
        ], dtype=np.float32)

        previous_obs = np.array([
            k_previous_obs(trk.observations, trk.age, self.delta_t)
            for trk in self.trackers
        ], dtype=np.float32)

        iou_matrix = iou_batch(trks, dets)

        if trk_embs.shape[0] == 0 or dets_embs.shape[0] == 0:
            app_matrix = np.zeros((len(self.trackers), len(dets)), dtype=np.float32)
        else:
            app_matrix = trk_embs @ dets_embs.T

        speed_matrix = self._build_speed_matrix(
            dets=dets,
            velocities=velocities,
            previous_obs=previous_obs,
        )

        matched, unmatched_dets, unmatched_trks, _ = self._run_gnn_association(
            dets=dets,
            dets_embs=dets_embs,
            dets_depth=dets_depth,
            frame=frame,
            fps=fps,
            iou_matrix=iou_matrix,
            speed_matrix=speed_matrix,
            app_matrix=app_matrix
        )

        for det_idx, trk_idx in matched:
            self.trackers[trk_idx].update(dets[det_idx], depth=dets_depth[det_idx:det_idx + 1], frame=frame)
            self.trackers[trk_idx].update_emb(dets_embs[det_idx], alpha=dets_alpha[det_idx])
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = np.array([self._tracker_bbox_for_graph(self.trackers[i]) for i in unmatched_trks], dtype=np.float32)
            iou_left = np.asarray(self.asso_func(left_dets, left_trks))
            if iou_left.size > 0 and iou_left.max() > self.iou_threshold - 0.1:
                rematched = linear_assignment(-iou_left)
                rm_det, rm_trk = [], []
                for li, lj in rematched:
                    if iou_left[li, lj] < self.iou_threshold:
                        continue
                    det_idx = int(unmatched_dets[li])
                    trk_idx = int(unmatched_trks[lj])
                    self.trackers[trk_idx].update(dets[det_idx], depth=dets_depth[det_idx:det_idx + 1], frame=frame)
                    self.trackers[trk_idx].update_emb(dets_embs[det_idx], alpha=dets_alpha[det_idx])
                    rm_det.append(det_idx)
                    rm_trk.append(trk_idx)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.asarray(rm_det, dtype=int))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.asarray(rm_trk, dtype=int))

        for trk_idx in unmatched_trks:
            self.trackers[int(trk_idx)].update(None, depth=None, frame=None)

        for det_idx in unmatched_dets:
            trk = KalmanBoxTracker(
                dets[int(det_idx)],
                frame=frame,
                depth=dets_depth[int(det_idx):int(det_idx) + 1],
                delta_t=self.delta_t,
                emb=dets_embs[int(det_idx)],
                alpha=0,
                new_kf=not self.new_kf_off,
            )
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
