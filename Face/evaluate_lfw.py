

import os
import sys
import json
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from model import DPAFD


# ─────────────────────────────────────────────────────────────
#  FACE EMBEDDER
# ─────────────────────────────────────────────────────────────

class FaceEmbedder(nn.Module):
    """
    Attaches a small embedding head on top of the DPAFD backbone+FPN
    to produce compact, L2-normalised face descriptors for verification.

    Why not use the detection head directly?
    The detection head predicts spatial heatmaps; for verification we
    need a single global vector per face.  We tap P3 (highest resolution
    FPN level) with global average pooling → MLP → L2 norm.
    """

    def __init__(self, dpafd: DPAFD, embed_dim: int = 128):
        super().__init__()
        self.dpafd = dpafd
        fpn_ch = dpafd.fpn.out3.block[0].block[0].out_channels  # read FPN out channels

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),          # fixed 4×4 spatial
            nn.Flatten(),
            nn.Linear(fpn_ch * 16, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )
        self._init()

    def _init(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c3, c4, c5 = self.dpafd.backbone(x)
        p3, _,  _  = self.dpafd.fpn(c3, c4, c5)
        emb = self.head(p3)
        return F.normalize(emb, p=2, dim=1)   # unit-L2 sphere


# ─────────────────────────────────────────────────────────────
#  LFW EVALUATOR
# ─────────────────────────────────────────────────────────────

class LFWEvaluator:

    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        model: nn.Module,
        root: str,
        pairs_file: str = 'pairs.txt',
        img_size: int   = 112,
        batch_size: int = 64,
        device: str     = 'cpu',
    ):
        self.model      = model.to(device)
        self.root       = root
        self.img_size   = img_size
        self.batch_size = batch_size
        self.device     = device
        self.pairs      = self._parse_pairs(os.path.join(root, pairs_file))
        self.tfm        = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self._MEAN, self._STD),
        ])

    # ── pair parsing ─────────────────────────────────────────

    def _parse_pairs(self, path: str) -> list:
        pairs = []
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]
        for line in lines[1:]:
            parts = line.split()
            if len(parts) == 3:
                n, i1, i2 = parts
                pairs.append((n, int(i1), n, int(i2), 1))
            elif len(parts) == 4:
                n1, i1, n2, i2 = parts
                pairs.append((n1, int(i1), n2, int(i2), 0))
        return pairs

    def _img_path(self, name: str, idx: int) -> str:
        for sub in ('lfw_funneled', 'lfw', ''):
            p = os.path.join(self.root, sub, name, f'{name}_{idx:04d}.jpg')
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(f'Not found: {name}/{idx}')

    # ── image loading ─────────────────────────────────────────

    def _load_img(self, name: str, idx: int) -> torch.Tensor:
        p   = self._img_path(name, idx)
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        return self.tfm(img)

    # ── embedding extraction ──────────────────────────────────

    def extract_embeddings(self) -> dict:
        """
        Returns dict: (name, idx) → embedding tensor [128]
        Processes in mini-batches for efficiency.
        """
        self.model.eval()

        # Collect unique (name, idx) keys
        keys = list({(n, i) for n, i, *_ in [
            (p[0], p[1]) for p in self.pairs
        ] + [(p[2], p[3]) for p in self.pairs]})

        embeddings = {}
        buf_keys, buf_tensors = [], []

        def flush():
            if not buf_tensors:
                return
            batch = torch.stack(buf_tensors).to(self.device)
            with torch.no_grad():
                embs = self.model(batch).cpu()
            for k, e in zip(buf_keys, embs):
                embeddings[k] = e
            buf_keys.clear()
            buf_tensors.clear()

        for key in tqdm(keys, desc='Embedding', ncols=80):
            try:
                t = self._load_img(*key)
            except Exception as e:
                print(f'  Warning: {e}')
                continue
            buf_keys.append(key)
            buf_tensors.append(t)
            if len(buf_tensors) == self.batch_size:
                flush()
        flush()

        print(f'Extracted {len(embeddings)} / {len(keys)} embeddings')
        return embeddings

    # ── core evaluation ───────────────────────────────────────

    def evaluate(self, embeddings: dict = None) -> dict:
        """
        Standard LFW 10-fold cross-validation protocol.

        Returns dict with:
          accuracy, accuracy_std, auc,
          tar_at_far_01, tar_at_far_1,
          fpr, tpr  (ROC arrays for plotting)
        """
        if embeddings is None:
            embeddings = self.extract_embeddings()

        sims, labels = [], []
        skipped = 0

        for n1, i1, n2, i2, label in self.pairs:
            k1, k2 = (n1, i1), (n2, i2)
            if k1 not in embeddings or k2 not in embeddings:
                skipped += 1
                continue
            sim = F.cosine_similarity(
                embeddings[k1].unsqueeze(0),
                embeddings[k2].unsqueeze(0),
            ).item()
            sims.append(sim)
            labels.append(label)

        if skipped:
            print(f'  Warning: {skipped} pairs skipped (missing images)')

        sims   = np.array(sims,   dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)

        # ── AUC ──────────────────────────────────────────────
        auc = roc_auc_score(labels, sims)
        fpr, tpr, _ = roc_curve(labels, sims)

        # ── 10-fold cross-validated accuracy ─────────────────
        n         = len(labels)
        fold_sz   = n // 10
        fold_accs = []

        for fold in range(10):
            val_idx   = np.arange(fold * fold_sz, (fold + 1) * fold_sz)
            train_idx = np.concatenate([
                np.arange(0, fold * fold_sz),
                np.arange((fold + 1) * fold_sz, n),
            ])

            best_acc, best_t = 0, 0
            for thresh in np.arange(-1.0, 1.0, 0.005):
                preds = (sims[train_idx] >= thresh).astype(int)
                acc   = (preds == labels[train_idx]).mean()
                if acc > best_acc:
                    best_acc, best_t = acc, thresh

            val_preds = (sims[val_idx] >= best_t).astype(int)
            fold_accs.append((val_preds == labels[val_idx]).mean())

        accuracy     = float(np.mean(fold_accs))
        accuracy_std = float(np.std(fold_accs))

        # ── TAR @ FAR ─────────────────────────────────────────
        tar_01 = float(np.interp(0.001, fpr, tpr))
        tar_1  = float(np.interp(0.01,  fpr, tpr))

        return {
            'n_pairs':        len(sims),
            'accuracy':       accuracy,
            'accuracy_std':   accuracy_std,
            'auc':            float(auc),
            'tar_at_far_0.1%': tar_01,
            'tar_at_far_1%':   tar_1,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
        }

    # ── plotting ──────────────────────────────────────────────

    def plot_roc(
        self,
        results: dict,
        save_path: str,
        label: str = 'DPAFD (Ours)',
        extra_curves: list = None,
    ):
        """
        Plot ROC curve with optional overlay of additional methods.
        extra_curves: list of (fpr_arr, tpr_arr, auc, name) tuples
        """
        fig, ax = plt.subplots(figsize=(7, 6))

        fpr = np.array(results['fpr'])
        tpr = np.array(results['tpr'])
        auc = results['auc']
        acc = results['accuracy']

        ax.plot(fpr, tpr, 'b-', lw=2,
                label=f'{label}  AUC={auc:.4f}  Acc={acc*100:.2f}%')

        if extra_curves:
            colors = ['r', 'g', 'm', 'orange']
            for (ef, et, ea, en), c in zip(extra_curves, colors):
                ax.plot(ef, et, f'{c}-', lw=1.5, label=f'{en}  AUC={ea:.4f}')

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate',  fontsize=12)
        ax.set_title('LFW Face Verification — ROC Curve', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f'ROC plot saved: {save_path}')

    # ── score distribution ────────────────────────────────────

    def plot_score_dist(self, results: dict, save_path: str):
        """Plot similarity score distributions for same vs different pairs."""
        sims   = np.array(results.get('similarities', []))
        labels = np.array(results.get('labels', []))
        if len(sims) == 0:
            return

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(sims[labels == 1], bins=60, alpha=0.6, color='steelblue', label='Same')
        ax.hist(sims[labels == 0], bins=60, alpha=0.6, color='tomato',    label='Different')
        ax.set_xlabel('Cosine Similarity', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Score Distribution — LFW', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f'Score dist plot saved: {save_path}')


# ─────────────────────────────────────────────────────────────
#  HELPER: load FaceEmbedder from checkpoint
# ─────────────────────────────────────────────────────────────

def load_embedder(checkpoint: str, fpn_channels: int, embed_dim: int, device: str):
    dpafd    = DPAFD(fpn_channels=fpn_channels)
    embedder = FaceEmbedder(dpafd, embed_dim=embed_dim)

    if checkpoint:
        ckpt   = torch.load(checkpoint, map_location=device)
        state  = ckpt.get('model_state', ckpt)
        missing, unexpected = embedder.load_state_dict(state, strict=False)
        if missing:
            print(f'  Missing keys  : {len(missing)}  (head weights random — fine-tune first)')
        if unexpected:
            print(f'  Unexpected keys: {len(unexpected)}')
        print(f'  Loaded: {checkpoint}')
    else:
        print('  No checkpoint — using random weights (sanity check only)')

    return embedder.to(device).eval()


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device : {device}')

    embedder  = load_embedder(args.checkpoint, args.fpn_channels, args.embed_dim, device)
    evaluator = LFWEvaluator(
        embedder,
        root       = args.lfw_root,
        pairs_file = args.pairs_file,
        img_size   = args.img_size,
        batch_size = args.batch_size,
        device     = device,
    )

    results = evaluator.evaluate()

    # ── print summary ─────────────────────────────────────────
    print()
    print('═' * 52)
    print('  LFW EVALUATION  —  DPAFD')
    print('═' * 52)
    print(f'  Pairs evaluated  : {results["n_pairs"]}')
    print(f'  Accuracy         : {results["accuracy"]*100:.2f}% ± {results["accuracy_std"]*100:.2f}%')
    print(f'  AUC-ROC          : {results["auc"]:.4f}')
    print(f'  TAR @ FAR=0.1%   : {results["tar_at_far_0.1%"]*100:.2f}%')
    print(f'  TAR @ FAR=1%     : {results["tar_at_far_1%"]*100:.2f}%')
    print('═' * 52)

    # ── save outputs ──────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    result_path = os.path.join(args.output_dir, 'lfw_results.json')
    saveable    = {k: v for k, v in results.items() if k not in ('fpr', 'tpr')}
    with open(result_path, 'w') as f:
        json.dump(saveable, f, indent=2)
    print(f'Results JSON : {result_path}')

    evaluator.plot_roc(
        results,
        save_path=os.path.join(args.output_dir, 'roc_lfw.png'),
    )


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate DPAFD on LFW')
    p.add_argument('--lfw_root',     required=True,          help='LFW root directory')
    p.add_argument('--checkpoint',   default=None,           help='Path to .pth checkpoint')
    p.add_argument('--pairs_file',   default='pairs.txt')
    p.add_argument('--output_dir',   default='eval_results')
    p.add_argument('--fpn_channels', type=int, default=64)
    p.add_argument('--embed_dim',    type=int, default=128)
    p.add_argument('--img_size',     type=int, default=112)
    p.add_argument('--batch_size',   type=int, default=64)
    return p.parse_args()


if __name__ == '__main__':
    main(parse_args())