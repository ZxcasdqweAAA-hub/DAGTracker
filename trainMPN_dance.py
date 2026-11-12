import sys
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
from torch.optim import AdamW
from tqdm import tqdm
from torch_geometric.data import DataLoader
import argparse
import logging

import math
import torch
from pathlib import Path

from models.monotracker.model.new_MPN_subgraph import MOTMPNet
from models.monotracker.model.model_util import read_yaml_to_dict
from models.monotracker.model.loss import FocalLoss
from data.dancetrack import DanceTrack

root_path = "/Idea_1/results"


def get_args():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("--use_detection", type=bool, default=True)
    parser.add_argument('--l2_norm_reid', type=bool, default=True)
    parser.add_argument('--emb_alpha', type=float, default=0.0)
    parser.add_argument("--emb", type = str, default="")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--use_depth",type=bool, default=False)
    parser.add_argument("--zero_node", type=bool, default=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    return parser.parse_args()


def resume_model(model, optimizer, args):
    path = os.path.join(root_path, args.exp_name)
    root = Path(path)

    pt_files = [f for f in root.glob("*.pth") if f.stem.isdigit()]

    if not pt_files:
        print("no checkpoint found")
        return 0

    max_file = max(pt_files, key=lambda f: int(f.stem))

    try:
        ckpt = torch.load(max_file, map_location="cpu")

        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            epoch = ckpt.get("epoch", int(max_file.stem))
        else:
            model.load_state_dict(ckpt)
            epoch = int(max_file.stem)

        print("resume success:", max_file)
        return epoch

    except Exception as e:
        print("resume fail:", e)
        return 0


def build_train_logger(log_file_name, append=False):
    logger_train = logging.getLogger('train_logger')
    logger_train.setLevel(logging.INFO)
    logger_train.propagate = False

    if logger_train.handlers:
        for h in logger_train.handlers[:]:
            logger_train.removeHandler(h)
            h.close()

    mode = 'a' if append else 'w'
    file_handler_train = logging.FileHandler(log_file_name, mode=mode)
    file_handler_train.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger_train.addHandler(file_handler_train)
    return logger_train


def train():
    yaml_path = '/cfg/mot17_model.yaml'
    cfg = read_yaml_to_dict(yaml_path)
    model = MOTMPNet(cfg['graph_model_params'])
    model = model.cuda()
    args = get_args()
    dataset = DanceTrack("train", args)


    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    log_dir = "/results/log/"
    os.makedirs(log_dir, exist_ok=True)
    log_file_name = os.path.join(log_dir, args.exp_name + ".txt")
    logger_train = build_train_logger(log_file_name, append=args.resume)

    save_dir = os.path.join(root_path, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    print(args)
    epochs = 40
    lr0 = 3.0e-4
    lr_final = 3.0e-7
    lrf = lr_final / lr0
    optimizer = AdamW(params=model.parameters(), lr=lr0, weight_decay=1e-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    model.train()
    if args.use_depth == False:
        model.use_depth_subgraph = False
    loss_function = FocalLoss(logits=True, gamma=1)
    start = 0
    if args.resume:
        start = resume_model(model, optimizer, args)
        if start > 0:
            scheduler.last_epoch = start - 1
    logger_train.info(args)
    for epoch in range(start, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print('-' * 30)

        running_loss = 0.0
        valid_steps = 0

        for data in tqdm(dataloader):
            if data.edge_index.numel() == 0:
                continue
            optimizer.zero_grad()
            x = data.x.cuda()
            if args.zero_node:
                x = x.zero_()
            edge_index = data.edge_index.cuda()
            edge_attr = data.edge_attr.cuda()
            if args.use_depth:
                x_depth = data.x_depth.cuda()
            else:
                x_depth = None
            edge_label = data.edge_label.cuda()
            frame_id = data.frame_id.cuda()
            num_prev_nodes = data.num_prev_nodes.cuda()
            batch = data.batch.cuda()
            ptr = data.ptr.cuda()
            outputs = model(x, edge_index, edge_attr, x_depth, frame_id, num_prev_nodes, batch, ptr)
            loss = torch.as_tensor([.0], ).cuda()
            num_steps = len(outputs['classified_edges'])
            for step in range(num_steps):
                loss = loss + loss_function(outputs['classified_edges'][step].view(-1), edge_label.view(-1))
            if torch.isnan(loss):
                print("find nan！stop trainning")
                sys.exit()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            valid_steps += 1
        scheduler.step()

        epoch_loss = running_loss / valid_steps if valid_steps > 0 else 0.0
        print(f'Training Loss: {epoch_loss:.4f}')
        logger_train.info('-' * 30)
        logger_train.info(f"\nEpoch {epoch + 1}/{epochs}")
        logger_train.info(f'Training Loss: {epoch_loss:.4f}')

        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(save_dir, f'{epoch + 1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, save_path)


if __name__ == '__main__':

    train()
