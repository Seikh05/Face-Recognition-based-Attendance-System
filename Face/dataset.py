

import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from losses import build_targets

# ImageNet mean / std for normalisation
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────
#  WIDER FACE  (training / detection)
# ─────────────────────────────────────────────────────────────

class WiderFaceDataset(Dataset):
    """
    Reads WIDER FACE annotations and builds CenterNet-style targets
    for each FPN level via losses.build_targets().

    Augmentation (training only):
      - random horizontal flip
      - random brightness / contrast jitter
      - letterbox resize to input_size × input_size
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        input_size: int = 640,
        strides: tuple = (8, 16, 32),
    ):
        self.root       = root
        self.split      = split
        self.input_size = input_size
        self.strides    = list(strides)
        self.samples    = self._parse(split)
        print(f'[WiderFace/{split}] {len(self.samples)} images loaded')

    # ── annotation parser ────────────────────────────────────

    def _parse(self, split: str) -> list:
        ann = os.path.join(
            self.root,
            'wider_face_split',
            f'wider_face_{split}_bbx_gt.txt',
        )
        samples, i = [], 0
        with open(ann) as f:
            lines = [l.rstrip() for l in f]

        while i < len(lines):
            rel_path    = lines[i];  i += 1
            num_faces   = int(lines[i]);  i += 1
            boxes       = []

            for _ in range(max(1, num_faces)):
                vals = list(map(int, lines[i].split()));  i += 1
                if num_faces > 0:
                    x, y, w, h = vals[:4]
                    # skip invalid / blurry / occluded ≥70% flags
                    invalid = (len(vals) >= 8 and vals[7] == 1)
                    if w > 5 and h > 5 and not invalid:
                        boxes.append([x, y, x + w, y + h])

            img_path = os.path.join(
                self.root, f'WIDER_{split}', 'images', rel_path
            )
            if os.path.isfile(img_path) and boxes:
                samples.append({'img': img_path, 'boxes': boxes})

        return samples

    # ── augmentation helpers ─────────────────────────────────

    @staticmethod
    def _hflip(img: np.ndarray, boxes: list) -> tuple:
        img   = cv2.flip(img, 1)
        W     = img.shape[1]
        boxes = [[W - b[2], b[1], W - b[0], b[3]] for b in boxes]
        return img, boxes

    @staticmethod
    def _color_jitter(img: np.ndarray) -> np.ndarray:
        alpha = random.uniform(0.7, 1.3)
        beta  = random.randint(-20, 20)
        return np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    @staticmethod
    def _letterbox(img: np.ndarray, boxes: list, size: int) -> tuple:
        H, W  = img.shape[:2]
        scale = size / max(H, W)
        nH, nW = int(H * scale), int(W * scale)
        img   = cv2.resize(img, (nW, nH), interpolation=cv2.INTER_LINEAR)

        pad_h, pad_w = size - nH, size - nW
        pt, pl = pad_h // 2, pad_w // 2
        img = cv2.copyMakeBorder(
            img, pt, pad_h - pt, pl, pad_w - pl,
            cv2.BORDER_CONSTANT, value=127,
        )
        boxes = [
            [b[0]*scale + pl, b[1]*scale + pt,
             b[2]*scale + pl, b[3]*scale + pt]
            for b in boxes
        ]
        return img, boxes

    # ── Dataset interface ─────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s     = self.samples[idx]
        img   = cv2.cvtColor(cv2.imread(s['img']), cv2.COLOR_BGR2RGB)
        boxes = [b[:] for b in s['boxes']]

        if self.split == 'train':
            if random.random() > 0.5:
                img, boxes = self._hflip(img, boxes)
            img = self._color_jitter(img)

        img, boxes = self._letterbox(img, boxes, self.input_size)
        targets    = build_targets([boxes], self.input_size, self.strides)

        # Normalise to tensor
        img_t = (img.astype(np.float32) / 255.0 - _MEAN) / _STD
        img_t = torch.from_numpy(img_t).permute(2, 0, 1).float()

        # Strip batch dim from targets (added by build_targets)
        single_targets = [{k: v[0] for k, v in t.items()} for t in targets]

        return img_t, single_targets


def collate_fn(batch):
    """Custom collate: stack images; re-batch per-level targets."""
    imgs, targets_list = zip(*batch)
    imgs      = torch.stack(imgs, 0)
    n_levels  = len(targets_list[0])

    batched = []
    for lvl in range(n_levels):
        batched.append({
            k: torch.stack([targets_list[b][lvl][k] for b in range(len(batch))], 0)
            for k in targets_list[0][lvl]
        })
    return imgs, batched


# ─────────────────────────────────────────────────────────────
#  LFW PAIRS  (verification evaluation)
# ─────────────────────────────────────────────────────────────

class LFWPairsDataset(Dataset):
    """
    Loads the standard LFW pairs for face verification.

    Each sample: (img1_tensor, img2_tensor, label)
      label=1 → same identity,  label=0 → different

    Pairs file format (pairs.txt):
      Line 1  : N [pairs_per_set]
      Lines + : <name> <i1> <i2>         (same person)
                <name1> <i1> <name2> <i2> (different)
    """

    def __init__(
        self,
        root: str,
        pairs_file: str = 'pairs.txt',
        img_size: int = 112,
    ):
        self.root     = root
        self.img_size = img_size
        self.pairs    = self._parse(os.path.join(root, pairs_file))
        self.tfm      = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])
        print(f'[LFW] {len(self.pairs)} pairs loaded')

    def _parse(self, path: str) -> list:
        pairs = []
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]

        for line in lines[1:]:
            parts = line.split()
            if len(parts) == 3:
                name, i1, i2 = parts
                pairs.append((name, int(i1), name, int(i2), 1))
            elif len(parts) == 4:
                n1, i1, n2, i2 = parts
                pairs.append((n1, int(i1), n2, int(i2), 0))
        return pairs

    def _img_path(self, name: str, idx: int) -> str:
        for sub in ('lfw_funneled', 'lfw', ''):
            p = os.path.join(self.root, sub, name, f'{name}_{idx:04d}.jpg')
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(f'LFW image not found: {name} #{idx}')

    def _load(self, name: str, idx: int) -> torch.Tensor:
        p   = self._img_path(name, idx)
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        return self.tfm(img)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        n1, i1, n2, i2, label = self.pairs[idx]
        return self._load(n1, i1), self._load(n2, i2), label


# ─────────────────────────────────────────────────────────────
#  QUICK SMOKE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        ds = WiderFaceDataset(sys.argv[1], split='train')
        img, tgts = ds[0]
        print('image :', img.shape)
        for lvl, t in enumerate(tgts):
            pos = t['mask'].sum().item()
            print(f'  level {lvl}: hm={tuple(t["heatmap"].shape)}  positives={pos}')