import argparse
import gzip
import pickle

import cv2
import glob
import matplotlib
import numpy as np
import os
import time
import torch
import gc
from depth_anything_v2.dpt import DepthAnythingV2
def sort_bboxes_by_strip_blocks(depth_map, normalized_bboxes, strip_ratio=0.5, trim_ratio=0.05,
                                num_blocks=5, mean_weight=1.0, median_weight=0.0):
    H = depth_map.shape[0]
    W = depth_map.shape[1]
    N = normalized_bboxes.shape[0]
    device = depth_map.device
    dtype = torch.float32

    depth_scores = torch.zeros(N, device=device, dtype=dtype)

    for i in range(N):
        x1_norm, y1_norm, x2_norm, y2_norm = normalized_bboxes[i]
        x1 = torch.round(x1_norm * (W - 1)).long()
        y1 = torch.round(y1_norm * (H - 1)).long()
        x2 = torch.round(x2_norm * (W - 1)).long()
        y2 = torch.round(y2_norm * (H - 1)).long()

        x1 = torch.clamp(x1, 0, W - 1)
        y1 = torch.clamp(y1, 0, H - 1)
        x2 = torch.clamp(x2, x1 + 1, W - 1)
        y2 = torch.clamp(y2, y1 + 1, H - 1)

        box_w = (x2 - x1).item()
        center_x = x1 + (x2 - x1) // 2
        strip_w = max(1, int(box_w * strip_ratio))
        strip_x1 = int(torch.clamp(center_x - strip_w // 2, x1, x2).item())
        strip_x2 = int(torch.clamp(center_x + strip_w // 2, x1, x2).item())
        middle_strip = depth_map[y1:y2 + 1, strip_x1:strip_x2 + 1]  # [strip_h, strip_w]
        strip_h, strip_w = middle_strip.shape

        if strip_h == 0 or strip_w == 0:
            depth_scores[i] = float('inf')
            continue
        flat = middle_strip.flatten()
        num_pixels = flat.numel()
        if num_pixels <= 2:
            truncated = middle_strip.clone()
        else:
            lower_q = torch.quantile(flat, trim_ratio)
            upper_q = torch.quantile(flat, 1.0 - trim_ratio)
            truncated = torch.clamp(middle_strip, min=lower_q, max=upper_q)

        block_h = max(1, strip_h // num_blocks)
        block_maxes = []
        for b in range(num_blocks):
            block_y1 = b * block_h
            if b == num_blocks - 1:
                block_y2 = truncated.shape[0]
            else:
                block_y2 = min((b + 1) * block_h, truncated.shape[0])
            if block_y1 >= block_y2:
                continue
            block = truncated[block_y1:block_y2, :]
            try:
                bmax = torch.nanmax(block)
            except Exception:
                bmax = torch.max(block)
            if torch.isnan(bmax):
                bmax = torch.tensor(float('inf'), device=device, dtype=dtype)
            block_maxes.append(bmax)

        if len(block_maxes) == 0:
            depth_scores[i] = float('inf')
            continue

        while len(block_maxes) < num_blocks:
            block_maxes.append(block_maxes[-1].clone())

        block_maxes = torch.stack(block_maxes).to(device=device, dtype=dtype)

        mean_max = torch.mean(block_maxes)
        median_max = torch.median(block_maxes)
        if torch.isnan(mean_max):
            mean_max = torch.tensor(float('inf'), device=device, dtype=dtype)
        if torch.isnan(median_max):
            median_max = torch.tensor(float('inf'), device=device, dtype=dtype)

        combined_score = mean_max * mean_weight + median_max * median_weight
        depth_scores[i] = combined_score

    sorted_indices = torch.argsort(depth_scores, descending=True)
    sorted_norm_bboxes = normalized_bboxes[sorted_indices]
    sorted_scores = depth_scores[sorted_indices]

    return sorted_norm_bboxes, sorted_scores, sorted_indices
def normalize_bboxes(bboxes, height=800, width=1440):
    bboxes = bboxes[:, :4]
    if not isinstance(bboxes, torch.Tensor):
        bboxes = torch.from_numpy(bboxes)

    scale = torch.tensor([1.0 / width, 1.0 / height, 1.0 / width, 1.0 / height], dtype=bboxes.dtype,
                         device=bboxes.device)

    normalized_bboxes = bboxes * scale

    return normalized_bboxes

def printer(img_visual, outputs_np, remarked=None):
    if remarked is None:
        remarked = []
    if outputs_np[0] is not None:
        bboxes = outputs_np[:, 0:4]
        # scores = outputs_np[:, 4]
        id = 0.

        for i in range(bboxes.shape[0]):
            x1, y1, x2, y2 = bboxes[i]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            color = (0, 255, 0)
            color2 = (255, 0, 0)

            if id in remarked:
                cv2.rectangle(img_visual, (x1, y1), (x2, y2), color2, 2)
            else:
                cv2.rectangle(img_visual, (x1, y1), (x2, y2), color, 2)


            #label = f"{id:.2f}" + " " + f"{conf:.2f}"
            label = f"{id}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)

            cv2.rectangle(
                img_visual,
                (x1, y1 - text_height - 3),
                (x1 + text_width, y1),
                color,
                -1
            )

            cv2.putText(
                img_visual,
                label,
                (x1, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                1
            )
            id = id + 1
        return img_visual

def save_tensor_as_compressed_state_dict(tensor, save_path):
    state_dict = {"tensor": tensor}
    with gzip.open(save_path, "wb") as f:
        pickle.dump(state_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_compressed_state_dict_tensor(load_path, device=None):
    with gzip.open(load_path, "rb") as f:
        state_dict = pickle.load(f)

    tensor = state_dict["tensor"]


    if device is not None:
        tensor = tensor.to(device)

    return tensor

def find_txt_files(folder_path, recursive=True):




    if recursive:
        pattern = os.path.join(folder_path, "**", "*.txt")
        txt_files = glob.glob(pattern, recursive=True)
    else:
        pattern = os.path.join(folder_path, "*.txt")
        txt_files = glob.glob(pattern)

    return [os.path.abspath(file) for file in txt_files]
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')


    parser.add_argument('--model', type=str, default='bytetrack')
    parser.add_argument('--bytetrack_path', type=str,
                        default='models/YOLOX/weights/bytetrack_x_mot17.pth.tar')
    parser.add_argument('--topictrack_path', type=str,
                        default='models/YOLOX/weights/bytetrack_x_mot17.pth.tar')
    parser.add_argument('--dataset', type=str, default='mot17')
    parser.add_argument('--use_yolo', type=bool, default=True)
    args = parser.parse_args()


    args.img_path = "datasets/dancetrack/image_names_for_test"
    args.outdir = "models/monoD/Depth-Anything-V2"
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.cuda().eval()

    img_txts = find_txt_files(args.img_path)
    img_txts.sort()
    img_txts = img_txts[10:]

    for img_txt in img_txts:


        if os.path.isfile(img_txt):
            if img_txt.endswith('txt'):
                with open(img_txt, 'r') as f:
                    filenames = f.read().splitlines()
            else:
                filenames = [img_txt]
        else:
            filenames = glob.glob(os.path.join(img_txt, '**/*'), recursive=True)

        os.makedirs(args.outdir, exist_ok=True)
        cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        root_path = "datasets/dancetrack/DepthAnythingV2"

        if not os.path.exists(os.path.join(root_path, img_txt.split('/')[-1].split('.')[0])):
            os.makedirs(os.path.join(root_path, img_txt.split('/')[-1].split('.')[0]))

        for k, filename in enumerate(filenames):
            print(f'Progress {k+1}/{len(filenames)}: {filename}')

            raw_image = cv2.imread(filename)
            #  = os.path.join(root_path, args.img_path.split('/')[-1].split('.')[0], filename.split('/')[-1].split('.')[0] + ".gz")
            depth_dir = os.path.join(root_path, img_txt.split('/')[-1].split('.')[0], filename.split('/')[-1].split('.')[0] + ".gz")
            with torch.no_grad():
                depth = depth_anything.infer_image(raw_image, args.input_size)



            depth = torch.from_numpy(depth)
            min = depth.min()
            max = depth.max()
            depth = depth.to(torch.float16)

            save_tensor_as_compressed_state_dict(depth, depth_dir)

            del depth
            gc.collect()
            torch.cuda.empty_cache()

