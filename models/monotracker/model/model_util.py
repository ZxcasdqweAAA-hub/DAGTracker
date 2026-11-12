import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import yaml
from pathlib import Path


def combine_tensors(tensor1, tensor2):
    tensor1_expanded = tensor1.unsqueeze(1)
    tensor2_expanded = tensor2.unsqueeze(0)
    tensor1_expanded = tensor1_expanded.expand(-1, tensor2.size(0), -1)
    tensor2_expanded = tensor2_expanded.expand(tensor1.size(0), -1, -1)
    combined = torch.cat((tensor1_expanded, tensor2_expanded), dim=2)
    return combined


def xywh_to_x1y1x2y2(boxes):
    cx, cy, w, h = boxes.unbind(dim=1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack((x1, y1, x2, y2), dim=1)


def pairwise_grid_encoding(bboxes1, bboxes2, v=7, num_bands=1, base=2.0):

    N, _ = bboxes1.shape
    M, _ = bboxes2.shape

    x1, y1, x2, y2 = bboxes1[:, 0:1], bboxes1[:, 1:2], bboxes1[:, 2:3], bboxes1[:, 3:4]
    x1_, y1_, x2_, y2_ = bboxes2[:, 0:1], bboxes2[:, 1:2], bboxes2[:, 2:3], bboxes2[:, 3:4]

    w = x2 - x1  # [N, 1]
    h = y2 - y1
    w_ = x2_ - x1_  # [M, 1]
    h_ = y2_ - y1_

    grid_coords = torch.linspace(0, 1, v, device=bboxes1.device)
    grid_y, grid_x = torch.meshgrid(grid_coords, grid_coords, indexing='ij')  # [v, v]

    grid_x = grid_x.view(1, 1, v, v, 1)  # [1, 1, v, v, 1]
    grid_y = grid_y.view(1, 1, v, v, 1)  # [1, 1, v, v, 1]

    x1 = x1.view(N, 1, 1, 1, 1)  # [N, 1, 1, 1, 1]
    y1 = y1.view(N, 1, 1, 1, 1)
    x2 = x2.view(N, 1, 1, 1, 1)
    y2 = y2.view(N, 1, 1, 1, 1)
    w = w.view(N, 1, 1, 1, 1)
    h = h.view(N, 1, 1, 1, 1)

    x1_ = x1_.view(1, M, 1, 1, 1)  # [1, M, 1, 1, 1]
    y1_ = y1_.view(1, M, 1, 1, 1)
    x2_ = x2_.view(1, M, 1, 1, 1)
    y2_ = y2_.view(1, M, 1, 1, 1)
    w_ = w_.view(1, M, 1, 1, 1)
    h_ = h_.view(1, M, 1, 1, 1)

    cell_x1 = x1 + grid_x * w  # [N, M, v, v, 1]
    cell_y1 = y1 + grid_y * h
    cell_x2 = x1 + (grid_x + 1 / v) * w
    cell_y2 = y1 + (grid_y + 1 / v) * h

    cell_x1_ = x1_ + grid_x * w_
    cell_y1_ = y1_ + grid_y * h_
    cell_x2_ = x1_ + (grid_x + 1 / v) * w_
    cell_y2_ = y1_ + (grid_y + 1 / v) * h_

    cx = (cell_x1 + cell_x2) / 2  # [N, M, v, v, 1]
    cy = (cell_y1 + cell_y2) / 2
    cx_ = (cell_x1_ + cell_x2_) / 2
    cy_ = (cell_y1_ + cell_y2_) / 2

    cell_w = cell_x2 - cell_x1  # [N, M, v, v, 1]
    cell_h = cell_y2 - cell_y1
    cell_w_ = cell_x2_ - cell_x1_
    cell_h_ = cell_y2_ - cell_y1_

    dx = (cx_ - cx) / (cell_w + 1e-8)  # [N, M, v, v, 1]
    dy = (cy_ - cy) / (cell_h + 1e-8)
    dw = torch.log((cell_w_ + 1e-8) / (cell_w + 1e-8))
    dh = torch.log((cell_h_ + 1e-8) / (cell_h + 1e-8))

    cell_area = cell_w * cell_h  # [N, M, v, v, 1]
    cell_area_ = cell_w_ * cell_h_

    area_ratio = torch.log((cell_area_ + 1e-8) / (cell_area + 1e-8))

    center_dist = torch.sqrt((cx_ - cx) ** 2 + (cy_ - cy) ** 2)

    diag1 = torch.sqrt(cell_w ** 2 + cell_h ** 2)

    norm_dist = center_dist / (diag1 + 1e-8)

    top_rel = (cell_y1_ - cell_y1) / (cell_h + 1e-8)
    bottom_rel = (cell_y2_ - cell_y2) / (cell_h + 1e-8)
    left_rel = (cell_x1_ - cell_x1) / (cell_w + 1e-8)
    right_rel = (cell_x2_ - cell_x2) / (cell_w + 1e-8)

    rel_features = torch.cat([dx, dy, dw, dh, area_ratio, norm_dist,
                              top_rel, bottom_rel, left_rel, right_rel], dim=-1)  # [N, M, v, v, 10]

    def fourier_encode(x, num_bands=num_bands, base=base):
        frequencies = base ** torch.linspace(0, num_bands - 1, num_bands, device=x.device)  # [num_bands]

        x_expanded = x.unsqueeze(-1)  # [N, M, v, v, 10, 1]
        frequencies = frequencies.view(1, 1, 1, 1, 1, -1)  # [1, 1, 1, 1, 1, num_bands]

        angular_speeds = 2 * math.pi * x_expanded * frequencies  # [N, M, v, v, 10, num_bands]

        sin_encoding = torch.sin(angular_speeds)
        cos_encoding = torch.cos(angular_speeds)

        encoded = torch.cat([sin_encoding, cos_encoding], dim=-1)  # [N, M, v, v, 10, 2*num_bands]

        encoded = encoded.view(N, M, v, v, -1)  # [N, M, v, v, 10*2*num_bands]

        return encoded

    encoded_features = fourier_encode(rel_features)  # [N, M, v, v, 10*2*num_bands]

    inter_x1 = torch.max(cell_x1, cell_x1_)  # [N, M, v, v, 1]
    inter_y1 = torch.max(cell_y1, cell_y1_)
    inter_x2 = torch.min(cell_x2, cell_x2_)
    inter_y2 = torch.min(cell_y2, cell_y2_)

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)

    inter_area = inter_w * inter_h
    union_area = cell_area + cell_area_ - inter_area
    iou = inter_area / (union_area + 1e-8)  # [N, M, v, v, 1]

    final_features = torch.cat([encoded_features, iou], dim=-1)  # [N, M, v, v, D]

    final_features = final_features.permute(0, 1, 4, 2, 3)

    return final_features


class Embedder_Fourier(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
        super(Embedder_Fourier, self).__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(
                2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self,
                input: torch.Tensor,
                rescale: float = 1.0):
        assert (input.shape[-1] == self.input_dim)
        out = []
        if self.include_input:
            out.append(input/rescale)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))
        out = torch.cat(out, dim=-1)

        assert (out.shape[-1] == self.out_dim)
        return out


def bilinear_sample2d(im, x, y, return_inbounds=False):

    if len(im.shape) == 5:
        B, N, C, H, W = list(im.shape)
    else:
        B, C, H, W = list(im.shape)
    N = list(x.shape)[1]

    x = x.float()
    y = y.float()
    H_f = torch.tensor(H, dtype=torch.float32)
    W_f = torch.tensor(W, dtype=torch.float32)

    # inbound_mask = (x>-0.5).float()*(y>-0.5).float()*(x<W_f+0.5).float()*(y<H_f+0.5).float()

    max_y = (H_f - 1).int()
    max_x = (W_f - 1).int()

    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1

    x0_clip = torch.clamp(x0, 0, max_x)
    x1_clip = torch.clamp(x1, 0, max_x)
    y0_clip = torch.clamp(y0, 0, max_y)
    y1_clip = torch.clamp(y1, 0, max_y)
    dim2 = W
    dim1 = W * H

    base = torch.arange(0, B, dtype=torch.int64, device=x.device) * dim1
    base = torch.reshape(base, [B, 1]).repeat([1, N])

    base_y0 = base + y0_clip * dim2
    base_y1 = base + y1_clip * dim2

    idx_y0_x0 = base_y0 + x0_clip
    idx_y0_x1 = base_y0 + x1_clip
    idx_y1_x0 = base_y1 + x0_clip
    idx_y1_x1 = base_y1 + x1_clip

    # use the indices to lookup pixels in the flat image
    # im is B x C x H x W
    # move C out to last dim
    if len(im.shape) == 5:
        im_flat = (im.permute(0, 3, 4, 1, 2)).reshape(B * H * W, N, C)
        i_y0_x0 = torch.diagonal(im_flat[idx_y0_x0.long()], dim1=1, dim2=2).permute(
            0, 2, 1
        )
        i_y0_x1 = torch.diagonal(im_flat[idx_y0_x1.long()], dim1=1, dim2=2).permute(
            0, 2, 1
        )
        i_y1_x0 = torch.diagonal(im_flat[idx_y1_x0.long()], dim1=1, dim2=2).permute(
            0, 2, 1
        )
        i_y1_x1 = torch.diagonal(im_flat[idx_y1_x1.long()], dim1=1, dim2=2).permute(
            0, 2, 1
        )
    else:
        im_flat = (im.permute(0, 2, 3, 1)).reshape(B * H * W, C)
        i_y0_x0 = im_flat[idx_y0_x0.long()]
        i_y0_x1 = im_flat[idx_y0_x1.long()]
        i_y1_x0 = im_flat[idx_y1_x0.long()]
        i_y1_x1 = im_flat[idx_y1_x1.long()]

    # Finally calculate interpolated values.
    x0_f = x0.float()
    x1_f = x1.float()
    y0_f = y0.float()
    y1_f = y1.float()

    w_y0_x0 = ((x1_f - x) * (y1_f - y)).unsqueeze(2)
    w_y0_x1 = ((x - x0_f) * (y1_f - y)).unsqueeze(2)
    w_y1_x0 = ((x1_f - x) * (y - y0_f)).unsqueeze(2)
    w_y1_x1 = ((x - x0_f) * (y - y0_f)).unsqueeze(2)

    output = (
        w_y0_x0 * i_y0_x0 + w_y0_x1 * i_y0_x1 + w_y1_x0 * i_y1_x0 + w_y1_x1 * i_y1_x1
    )
    # output is B*N x C
    output = output.view(B, -1, C)
    output = output.permute(0, 2, 1)
    # output is B x C x N

    if return_inbounds:
        x_valid = (x > -0.5).byte() & (x < float(W_f - 0.5)).byte()
        y_valid = (y > -0.5).byte() & (y < float(H_f - 0.5)).byte()
        inbounds = (x_valid & y_valid).float()
        inbounds = inbounds.reshape(
            B, N
        )  # something seems wrong here for B>1; i'm getting an error here (or downstream if i put -1)
        return output, inbounds

    return output  # B, C, N


def extract_bbox_depths(depth_map, bboxes):
    depth_map_expanded = depth_map.unsqueeze(0)

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    sample_points_x = torch.stack([x1, x2], dim=0).unsqueeze(0)
    sample_points_y = torch.stack([y1, y2], dim=0).unsqueeze(0)

    B, _, N = sample_points_x.shape
    sample_points_x_flat = sample_points_x.reshape(B, -1)
    sample_points_y_flat = sample_points_y.reshape(B, -1)

    sampled_depths = bilinear_sample2d(depth_map_expanded, sample_points_x_flat, sample_points_y_flat)


    sampled_depths = sampled_depths.reshape(B, 1, 2, N)
    result = sampled_depths.squeeze(0).squeeze(0).permute(1, 0)
    return result


def generate_xz_yz_bboxes(bboxes, depths):

    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    z1, z2 = depths[:, 0], depths[:, 1]


    xz_top = torch.min(z1, z2)
    xz_bottom = torch.max(z1, z2)

    xz_bboxes = torch.stack([x1, xz_top, x2, xz_bottom], dim=1)

    yz_left = torch.min(z1, z2)
    yz_right = torch.max(z1, z2)

    yz_bboxes = torch.stack([yz_left, y1, yz_right, y2], dim=1)

    return xz_bboxes, yz_bboxes


def read_yaml_to_dict(yaml_file_path: str) -> dict:
    file_path = Path(yaml_file_path)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            yaml_dict = yaml.safe_load(f)
        return yaml_dict
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML resolve false: {e}")
    except Exception as e:
        raise Exception(f"error : {e}")
