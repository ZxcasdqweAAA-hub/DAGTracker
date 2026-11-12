import shutil
import time

import cv2
import os
import glob
import torch
import PIL.Image as Image
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from mytracker import args as mytracker_args
from mytracker import gnn_sort

seq_fps = {
        "MOT17-01-SDP": 30,
        "MOT17-02-SDP": 30,
        "MOT17-03-SDP": 30,
        "MOT17-04-SDP": 30,
        "MOT17-05-SDP": 14,
        "MOT17-06-SDP": 14,
        "MOT17-07-SDP": 30,
        "MOT17-09-SDP": 30,
        "MOT17-10-SDP": 30,
        "MOT17-11-SDP": 30,
        "MOT17-08-SDP": 30,
        "MOT17-13-SDP": 25,
        "MOT17-12-SDP": 30,
        "MOT17-14-SDP": 25,
    }

gt_path = ""
video_path = ""
def sort_output(txt_path):
    with open(txt_path, 'r') as f:
        list = []
        for line in f:
            list.append(line.strip())

    with open(txt_path, "w") as f:
        for item in sorted(list, key=lambda x: int(str(x).split(',')[0])):
            f.writelines(item)
            f.writelines('\n')
        f.close()
def draw_dataset(video_id, gt_path, dataset, save_dir, fps):

    txt_name = gt_path + '/' + video_id + '.txt'
    file_path_img = "MOT17/images/test/" + video_id + "/img1"
    frame_width = 1920
    frame_height = 1080
    exp_name = gt_path.split('/')[-2] if gt_path.split('/')[-1] == "data" else gt_path.split('/')[-1]
    video_name = os.path.join(save_dir, video_id + "_0_" + exp_name + '.mp4')
    output = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(
        *'mp4v'), fps, (frame_width, frame_height))
    save_file_path = os.path.join(save_dir, video_id)
    sort_output(txt_name)

    source_file = open(txt_name)
    records = source_file.readlines()

    i = 0
    img_names = []
    for line in records:
        staff = line.split(',')
        img_name = staff[0]
        img_names.append(img_name)
        if i == 0:
            num = int(float(staff[1])) - 1
        i += 1

    name_dict = {}
    for i in img_names:
        if img_names.count(i):
            name_dict[i] = img_names.count(i)

    l = 0
    for idx in name_dict:
        img = cv2.imread(os.path.join(
            file_path_img, str(idx).rjust(6, '0') + '.jpg'))
        for i in range(name_dict[idx]):

            line = records[l]
            l += 1
            staff = line.split(',')

            id = staff[1]
            frame = staff[0]
            cls = staff[7]
            box = staff[2:6]
            cv2.rectangle(img, (int(float(box[0])), int(float(box[1]))),
                          (int(float(box[0])) + int(float(box[2])),
                           int(float(box[1])) + int(float(box[3]))),
                          (255, 255, 0), 2, cv2.LINE_4)
            cv2.rectangle(img, (int(float(box[0])), int(float(box[1]))),
                          ((int(float(box[0])) + 20),
                           (int(float(box[1])) + 15)),
                          (255, 255, 0), thickness=-1)
            cv2.putText(img, str(int(float(id))), (int(float(box[0])), int(float(box[1]))+12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        txt = f'{dataset}  ' + 'frame: ' + str(frame) + '  num: ' + str(i + 1)
        cv2.putText(img, txt, (0, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        img = cv2.resize(img, dsize=(1920, 1080))

        output.write(img)

    output.release()
    source_file.close()

def draw_dataset_color(video_id, gt_path, dataset, save_dir, fps):
    txt_name = gt_path + '/' + video_id + '.txt'
    file_path_img = "MOT17/images/test/" + video_id + "/img1"
    frame_width = 1920
    frame_height = 1080
    exp_name = gt_path.split('/')[-2] if gt_path.split('/')[-1] == "data" else gt_path.split('/')[-1]
    video_name = os.path.join(save_dir, video_id + exp_name + '.mp4')
    output = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    save_file_path = os.path.join(save_dir, video_id)
    sort_output(txt_name)

    source_file = open(txt_name)
    records = source_file.readlines()

    color_palette = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 128, 255),
        (0, 255, 128),
        (128, 0, 255),
        (255, 0, 128),
        (128, 255, 0),
        (255, 128, 0),
        (128, 0, 128),
        (0, 128, 128),
        (128, 128, 0),
        (0, 0, 128),
        (128, 0, 0),
        (0, 128, 0),
        (255, 128, 128),
        (128, 255, 128),
        (128, 128, 255),
        (255, 255, 128),
        (128, 255, 255),
    ]

    id_color_map = {}
    next_color_idx = 0

    def get_next_color():
        nonlocal next_color_idx
        c = color_palette[next_color_idx % len(color_palette)]
        next_color_idx += 1
        return c

    i = 0
    img_names = []
    for line in records:
        staff = line.split(',')
        img_name = staff[0]
        img_names.append(img_name)
        if i == 0:
            num = int(float(staff[1])) - 1
        i += 1

    name_dict = {}
    for img_name in img_names:
        name_dict[img_name] = name_dict.get(img_name, 0) + 1

    l = 0
    id_list = [129, 166, 188]
    for idx in name_dict:
        img = cv2.imread(os.path.join(file_path_img, str(idx).rjust(6, '0') + '.jpg'))
        for _ in range(name_dict[idx]):
            line = records[l]
            l += 1
            staff = line.split(',')
            obj_id = staff[1]

            frame = staff[0]
            cls = staff[7]
            box = staff[2:6]

            if obj_id not in id_color_map:
                id_color_map[obj_id] = get_next_color()
            color = id_color_map[obj_id]
            if int(float(obj_id)) not in id_list:
                continue
            x1 = int(float(box[0]))
            y1 = int(float(box[1]))
            w  = int(float(box[2]))
            h  = int(float(box[3]))

            cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), color, 2, cv2.LINE_4)
            id_text = str(int(float(obj_id)))

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2

            padding_x = 0
            padding_y = 0

            (text_width, text_height), baseline = cv2.getTextSize(
                id_text,
                font,
                font_scale,
                font_thickness
            )

            label_width = text_width + padding_x * 2
            label_height = text_height + baseline + padding_y * 2

            label_x1 = x1
            label_y1 = y1
            label_x2 = x1 + label_width
            label_y2 = y1 + label_height

            label_x2 = min(label_x2, img.shape[1] - 1)
            label_y2 = min(label_y2, img.shape[0] - 1)

            cv2.rectangle(
                img,
                (label_x1, label_y1),
                (label_x2, label_y2),
                color,
                thickness=-1
            )

            text_x = label_x1 + padding_x
            text_y = label_y1 + padding_y + text_height
            text_x = label_x1
            text_y = label_y1  + text_height

            cv2.putText(
                img,
                id_text,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA
            )


        txt = f'{dataset}  frame: {idx}  num: {name_dict[idx]}'
        cv2.putText(img, txt, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        img = cv2.resize(img, dsize=(frame_width, frame_height))
        output.write(img)

    output.release()
    source_file.close()
def get_main_args():


    parser = mytracker_args.make_parser()
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument('--video_id', type=str, default='')
    parser.add_argument('--input', '-i',
                        default="datasets/MOT17/images/test/MOT17-08-SDP/img1")
    parser.add_argument('--output', '-o',
                        default="results/video")
    parser.add_argument("--dataset", type=str, default="mot17")

    parser.add_argument('--bytetrack_path', type=str,
                        default='weights/bytetrack_x_mot17.pth.tar')
    parser.add_argument("--result_folder", type=str, default="results/trackers/")
    parser.add_argument("--test_dataset", action="store_true")

    parser.add_argument("--min_box_area", type=float, default=10, help="filter out tiny boxes")
    parser.add_argument(
        "--aspect_ratio_thresh",
        type=float,
        default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value.",
    )
    parser.add_argument(
        "--post",
        action="store_true",
        help="run post-processing linear interpolation.",
    )
    parser.add_argument("--w_assoc_emb", type=float, default=0.75, help="Combine weight for emb cost")
    parser.add_argument(
        "--alpha_fixed_emb",
        type=float,
        default=0.95,
        help="Alpha fixed for EMA embedding",
    )
    parser.add_argument("--emb_off", action="store_true")
    parser.add_argument("--cmc_off", action="store_true")
    parser.add_argument("--aw_off", action="store_true")
    parser.add_argument("--aw_param", type=float, default=0.5)
    parser.add_argument("--new_kf_off", action="store_true")
    parser.add_argument("--grid_off", action="store_true")
    parser.add_argument(
        "--timing_warmup",
        type=int,
        default=50,
        help="number of initial frames excluded from tracker.update timing",
    )
    args = parser.parse_args()
    args.input = "MOT17/images/test/" + args.video_id +  "/img1"
    if args.dataset == "mot17":
        args.result_folder = os.path.join(args.result_folder, "MOT17-val")
    elif args.dataset == "mot20":
        args.result_folder = os.path.join(args.result_folder, "MOT20-val")
    elif args.dataset == "dance":
        args.result_folder = os.path.join(args.result_folder, "DANCE-val")
    if args.test_dataset:
        args.result_folder.replace("-val", "-test")
    return args

def filter_targets(online_targets, aspect_ratio_thresh, min_box_area):
    """Removes targets not meeting threshold criteria.

    Returns (list of tlwh, list of ids).
    """
    online_tlwhs = []
    online_ids = []
    for t in online_targets:
        tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
        tid = t[4]
        vertical = tlwh[2] / tlwh[3] > aspect_ratio_thresh
        if tlwh[2] * tlwh[3] > min_box_area and not vertical:
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
    return online_tlwhs, online_ids
def write_results_no_score(filename, results):
    """Writes results in MOT style to filename."""
    save_format = "{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n"
    with open(filename, "w") as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(
                    frame=frame_id,
                    id=track_id,
                    x1=round(x1, 1),
                    y1=round(y1, 1),
                    w=round(w, 1),
                    h=round(h, 1),
                )
                f.write(line)
def dti(txt_path, save_path, n_min=30, n_dti=20):
    def dti_write_results(filename, results):
        save_format = "{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n"
        with open(filename, "w") as f:
            for i in range(results.shape[0]):
                frame_data = results[i]
                frame_id = int(frame_data[0])
                track_id = int(frame_data[1])
                x1, y1, w, h = frame_data[2:6]
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, w=w, h=h, s=-1)
                f.write(line)

    seq_txts = sorted(glob.glob(os.path.join(txt_path, "*.txt")))
    # breakpoint()
    for seq_txt in seq_txts:
        seq_name = seq_txt.replace("\\", "/").split("/")[-1]  ## To better play along with windows paths
        seq_data = np.loadtxt(seq_txt, dtype=np.float64, delimiter=",")
        min_id = int(np.min(seq_data[:, 1]))
        max_id = int(np.max(seq_data[:, 1]))
        seq_results = np.zeros((1, 10), dtype=np.float64)
        for track_id in range(min_id, max_id + 1):
            index = seq_data[:, 1] == track_id
            tracklet = seq_data[index]
            tracklet_dti = tracklet
            if tracklet.shape[0] == 0:
                continue
            n_frame = tracklet.shape[0]
            n_conf = np.sum(tracklet[:, 6] > 0.5)
            if n_frame > n_min:
                frames = tracklet[:, 0]
                frames_dti = {}
                for i in range(0, n_frame):
                    right_frame = frames[i]
                    if i > 0:
                        left_frame = frames[i - 1]
                    else:
                        left_frame = frames[i]
                    # disconnected track interpolation
                    if 1 < right_frame - left_frame < n_dti:
                        num_bi = int(right_frame - left_frame - 1)
                        right_bbox = tracklet[i, 2:6]
                        left_bbox = tracklet[i - 1, 2:6]
                        for j in range(1, num_bi + 1):
                            curr_frame = j + left_frame
                            curr_bbox = (curr_frame - left_frame) * (right_bbox - left_bbox) / (
                                right_frame - left_frame
                            ) + left_bbox
                            frames_dti[curr_frame] = curr_bbox
                num_dti = len(frames_dti.keys())
                if num_dti > 0:
                    data_dti = np.zeros((num_dti, 10), dtype=np.float64)
                    for n in range(num_dti):
                        data_dti[n, 0] = list(frames_dti.keys())[n]
                        data_dti[n, 1] = track_id
                        data_dti[n, 2:6] = frames_dti[list(frames_dti.keys())[n]]
                        data_dti[n, 6:] = [1, -1, -1, -1]
                    tracklet_dti = np.vstack((tracklet, data_dti))
            seq_results = np.vstack((seq_results, tracklet_dti))
        save_seq_txt = os.path.join(save_path, seq_name)
        seq_results = seq_results[1:]
        seq_results = seq_results[seq_results[:, 0].argsort()]
        dti_write_results(save_seq_txt, seq_results)
def main(args):
    # np.set_printoptions(suppress=True, precision=5)
    # Set dataset and detector
    # args = get_main_args()
    # det = get_model(args.bytetrack_path, args.dataset)

    if args.track_thresh != 0.6:
        args.exp_name = args.exp_name + "_thresh_" + str(args.track_thresh)

    np.set_printoptions(suppress=True, precision=5)
    tracker = gnn_sort.OCSort(det_thresh=0.6,
                              depth_length=10,
                              model_ckpt="results/exp_name.pth",
                              model_yaml="cfg/mot17_model.yaml")
    results = []
    image_extensions = ('png', 'jpg', 'jpeg')
    image_files = sorted(
        [f for f in os.listdir(args.input)
         if f.lower().endswith(image_extensions)],
        key=lambda x: int(''.join(filter(str.isdigit, x))) if any(char.isdigit() for char in x) else x
    )

    index = 0
    total_update_time = 0.0
    timed_frames = 0
    warmup_frames = max(0, int(getattr(args, "timing_warmup", 50)))
    depth_root = "MOT17/DepthAnythingV2"
    det_root = "MOT17/detections/assigned"
    seq_name = args.input.split('/')[-2]
    fps = seq_fps.get(seq_name)
    for i in range(len(image_files)):
        img_path = os.path.join(args.input, image_files[index])
        img = cv2.imread(img_path)

        depth_path = os.path.join(depth_root, seq_name, f"{(index + 1):06d}.pt")
        depth = torch.load(depth_path)

        det_path = os.path.join(det_root, seq_name, f"{(index + 1):06d}.txt")
        detection = []
        if not os.path.exists(det_path):
            detection = None
        else:
            for line in open(det_path):
                _, i, x, y, w, h, conf = line.strip("\n").split(",")
                x, y, w, h, conf = map(float, (x, y, w, h, conf))
                x, y, w, h = map(float, (x, y, w, h))
                detection.append([x, y, w, h, conf])
            if len(detection) == 0:
                detection = None
            else:
                detection = np.asarray(detection, dtype=np.float32)
                detection[:, 2] = detection[:, 0] + detection[:, 2]
                detection[:, 3] = detection[:, 1] + detection[:, 3]
                detection[:, :4] = detection[:, :4] + 1
        # tracker.update includes DGSE-ReID, depth-cue processing, graph
        # association, assignment, and track management. Image/depth/detection
        # loading above and result writing below are intentionally excluded.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        update_start = time.perf_counter()
        targets = tracker.update(detection, img, depth, index + 1, fps)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        update_elapsed = time.perf_counter() - update_start

        if index >= warmup_frames:
            total_update_time += update_elapsed
            timed_frames += 1
        tlwhs, ids = filter_targets(targets, args.aspect_ratio_thresh, args.min_box_area)
        results.append((index + 1, tlwhs, ids))
        index = index + 1


    folder = os.path.join(args.output, args.exp_name)
    os.makedirs(folder, exist_ok=True)
    result_filename = os.path.join(folder, f"{seq_name}.txt")
    write_results_no_score(result_filename, results)

    if args.post:
        post_folder = os.path.join(args.output, args.exp_name + "_post")
        pre_folder = os.path.join(args.output, args.exp_name)
        if os.path.exists(post_folder):
            print(f"Overwriting previous results in {post_folder}")
            shutil.rmtree(post_folder)
        shutil.copytree(pre_folder, post_folder)
        post_folder_data = os.path.join(post_folder, "data")
        dti(post_folder_data, post_folder_data)
        print(f"Linear interpolation post-processing applied, saved to {post_folder_data}.")

    if timed_frames == 0:
        raise RuntimeError(
            f"No frames were timed for {seq_name}; reduce --timing_warmup "
            f"(current value: {warmup_frames})."
        )
    tracking_fps = timed_frames / total_update_time
    latency_ms = total_update_time * 1000.0 / timed_frames
    timing = {
        "sequence": seq_name,
        "total_frames": len(image_files),
        "warmup_frames": min(warmup_frames, len(image_files)),
        "timed_frames": timed_frames,
        "update_time_seconds": total_update_time,
        "latency_ms_per_frame": latency_ms,
        "tracking_fps": tracking_fps,
    }
    print(
        f"[TIMING] {seq_name}: {timed_frames} frames, "
        f"{total_update_time:.3f} s, {latency_ms:.2f} ms/frame, "
        f"{tracking_fps:.2f} FPS"
    )
    return timing


if __name__ == '__main__':

    args = get_main_args()
    main(args)


