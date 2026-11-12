# encoding: utf-8

from torchvision.transforms import functional as F
from fast_reid.fastreid.data.transforms.functional import to_tensor
import torch
from PIL import Image
import numpy as np
import cv2
class SyncCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, depth):
        for t in self.transforms:
            img, depth = t(img, depth)
        return img, depth
class SyncResize:
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img: Image.Image, depth: np.ndarray):
        img = F.resize(img, self.size, self.interpolation)

        depth = cv2.resize(
            depth,
            (img.size[0], img.size[1]),
            interpolation=cv2.INTER_NEAREST
        )
        return img, depth
class SyncToTensor:
    def __call__(self, img: Image.Image, depth: np.ndarray):
        img = to_tensor(img)   # [3, H, W]
        depth = torch.from_numpy(depth.copy()).float()
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)  # [1, H, W]
        else:
            depth = depth.permute(2, 0, 1)
        return img, depth
class SyncRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, depth):
        if np.random.rand() < self.p:
            img = F.hflip(img)
            depth = np.fliplr(depth)
        return img, depth
class SyncPadRandomCrop:
    def __init__(self, padding, crop_size, padding_mode="constant"):
        self.padding = padding
        if isinstance(crop_size, (list, tuple)):
            assert len(crop_size) == 2
            self.crop_size = (int(crop_size[0]), int(crop_size[1]))
        else:
            self.crop_size = (int(crop_size), int(crop_size))
        self.padding_mode = padding_mode

    def __call__(self, img, depth):
        img = F.pad(img, self.padding, padding_mode=self.padding_mode)

        pad = self.padding
        depth = np.pad(
            depth,
            ((pad, pad), (pad, pad)),
            mode=self.padding_mode if self.padding_mode != "constant" else "constant"
        )

        w, h = img.size
        th, tw = self.crop_size

        i = np.random.randint(0, h - th + 1)
        j = np.random.randint(0, w - tw + 1)

        img = F.crop(img, i, j, th, tw)
        depth = depth[i:i+th, j:j+tw]
        return img, depth
class SyncRandomErasing:
    def __init__(self, p: float = 0.5, value_img: list = [0, 0, 0], value_depth: int = 0):
        self.p = p
        self.value_img = value_img
        self.value_depth = value_depth

    def __call__(self, img: torch.Tensor, depth: np.ndarray):
        if np.random.rand() < self.p:
            img_c, img_h, img_w = img.shape
            scale = (0.02, 0.33)
            ratio = (0.3, 3.3)
            area = img_h * img_w
            target_area = np.random.uniform(*scale) * area
            aspect_ratio = np.random.uniform(*ratio)

            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))
            if w < img_w and h < img_h:
                x = np.random.randint(0, img_w - w)
                y = np.random.randint(0, img_h - h)

                value_tensor = torch.tensor(
                    self.value_img,
                    dtype=img.dtype,
                    device=img.device
                ).view(-1, 1, 1)
                img[:, y:y+h, x:x+w] = value_tensor

                if len(depth.shape) == 2:
                    depth[y:y+h, x:x+w] = self.value_depth
                else:
                    depth[y:y+h, x:x+w, :] = self.value_depth
        return img, depth


def build_my_transforms(cfg, is_train=True):
    res = []

    if is_train:
        size_train = cfg.INPUT.SIZE_TRAIN

        if size_train[0] > 0:
            res.append(SyncResize(size_train if len(size_train) > 1 else size_train[0]))

        if cfg.INPUT.PADDING.ENABLED:
            res.append(
                SyncPadRandomCrop(
                    padding=cfg.INPUT.PADDING.SIZE,
                    crop_size=size_train if len(size_train) > 1 else size_train[0],
                    padding_mode=cfg.INPUT.PADDING.MODE,
                )
            )

        if cfg.INPUT.FLIP.ENABLED:
            res.append(SyncRandomHorizontalFlip(cfg.INPUT.FLIP.PROB))

        res.append(SyncToTensor())

        if cfg.INPUT.REA.ENABLED:
            res.append(
                SyncRandomErasing(
                    p=cfg.INPUT.REA.PROB,
                    value_img=cfg.INPUT.REA.VALUE,
                    value_depth=0
                )
            )

    else:
        size_test = cfg.INPUT.SIZE_TEST
        if size_test[0] > 0:
            res.append(SyncResize(size_test if len(size_test) > 1 else size_test[0]))
        res.append(SyncToTensor())

    return SyncCompose(res)

