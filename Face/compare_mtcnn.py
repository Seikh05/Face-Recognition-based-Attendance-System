

import os
import sys
import json
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

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
from model        import DPAFD
from evaluate_lfw import FaceEmbedder, LFWEvaluator, load_embedder

# Try to import facenet-pytorch for MTCNN baseline
try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    print('[Warning] facenet-pytorch not installed.  '
          'Install via: pip install facenet-pytorch\n'
          '  MTCNN comparison will be skipped.\n')


# ─────────────────────────────────────────────────────────────
#  MTCNN EVALUATION  (facenet-pytorch)
# ─────────────────────────────────────────────────────────────

class MTCNNEvaluator:
    """
    Uses facenet-pytorch's MTCNN for detection + cropping and
    InceptionResnetV1 (pretrained on VGGFace2) for embeddings.
    This is the standard MTCNN reference baseline.
    """

    _MEAN = [0.5, 0.5, 0.5]
    _STD  = [0.5, 0.5, 0.5]

    def __init__(self, root: str, pairs_file: str, img_size: int, device: str):
        self.root    = root
        self.device  = device
        self.img_sz  = img_size
        self.pairs   = self._parse(os.path.join(root, pairs_file))

        self.mtcnn  = MTCNN(image_size=img_size, device=device, keep_all=False,
                            min_face_size=20, post_process=True, select_largest=True)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

        self.fallback_tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self._MEAN, self._STD),
        ])

    def _parse(self, path: str) -> list:
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

    def _embed(self, path: str) -> torch.Tensor:
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

        # MTCNN detection + crop
        face = self.mtcnn(img)
        if face is None:
            # Fallback: centre crop
            H, W = img.shape[:2]
            s    = min(H, W)
            crop = img[(H-s)//2:(H-s)//2+s, (W-s)//2:(W-s)//2+s]
            crop = cv2.resize(crop, (self.img_sz, self.img_sz))
            face = self.fallback_tfm(crop)

        with torch.no_grad():
            emb = self.resnet(face.unsqueeze(0).to(self.device))
        return F.normalize(emb, p=2, dim=1).cpu().squeeze(0)

    def evaluate(self, n_pairs: int = None) -> dict:
        sims, labels = [], []
        pairs = self.pairs[:n_pairs] if n_pairs else self.pairs
        fails = 0

        for n1, i1, n2, i2, label in tqdm(pairs, desc='MTCNN eval', ncols=80):
            try:
                e1 = self._embed(self._img_path(n1, i1))
                e2 = self._embed(self._img_path(n2, i2))
                sims.append(F.cosine_similarity(e1.unsqueeze(0),
                                                e2.unsqueeze(0)).item())
                labels.append(label)
            except Exception:
                fails += 1

        if fails:
            print(f'  MTCNN: {fails} pairs failed')

        sims   = np.array(sims,   dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)

        auc         = roc_auc_score(labels, sims)
        fpr, tpr, _ = roc_curve(labels, sims)
        tar_01      = float(np.interp(0.001, fpr, tpr))
        tar_1       = float(np.interp(0.01,  fpr, tpr))

        # Best-threshold accuracy (single pass for speed)
        best_acc = 0.0
        for t in np.arange(-1.0, 1.0, 0.005):
            acc = ((sims >= t) == labels).mean()
            if acc > best_acc:
                best_acc = acc

        return {
            'n_pairs':        len(sims),
            'accuracy':       float(best_acc),
            'accuracy_std':   0.0,
            'auc':            float(auc),
            'tar_at_far_0.1%': tar_01,
            'tar_at_far_1%':   tar_1,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
        }


# ─────────────────────────────────────────────────────────────
#  SPEED BENCHMARK
# ─────────────────────────────────────────────────────────────

def benchmark_speed(model_fn, device: str, input_size=(3, 640, 640),
                    n_warmup: int = 10, n_runs: int = 100) -> dict:
    """Measure inference throughput (ms/img, FPS)."""
    dummy = torch.randn(1, *input_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            model_fn(dummy)

    if device == 'cuda':
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model_fn(dummy)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    ms  = float(np.mean(times))
    std = float(np.std(times))
    fps = 1000.0 / ms
    return {'ms_per_img': ms, 'ms_std': std, 'fps': fps}


# ─────────────────────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────────────────────

def plot_comparison_bars(results: dict, save_path: str):
    """
    3-panel bar chart:  Accuracy | AUC | TAR@FAR=1%
    """
    names   = list(results.keys())
    metrics = [
        ('accuracy',       'Accuracy (%)',    100),
        ('auc',            'AUC-ROC',          1),
        ('tar_at_far_1%',  'TAR @ FAR=1% (%)', 100),
    ]
    colors  = ['#1565C0', '#2E7D32', '#B71C1C', '#6A1B9A', '#E65100']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('DPAFD vs MTCNN — LFW Face Verification',
                 fontsize=15, fontweight='bold', y=1.01)

    for ax, (key, ylabel, scale) in zip(axes, metrics):
        vals  = [results[n][key] * scale for n in names]
        bars  = ax.bar(names, vals,
                       color=colors[:len(names)],
                       width=0.55, edgecolor='white', linewidth=1.5)

        # Value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.008,
                f'{val:.2f}{"%" if scale == 100 else ""}',
                ha='center', va='bottom', fontsize=10, fontweight='bold',
            )

        ax.set_title(ylabel, fontsize=12, fontweight='bold', pad=8)
        ax.set_ylim(max(0, min(vals) - max(vals) * 0.05),
                    max(vals) * 1.10)
        ax.tick_params(axis='x', rotation=15, labelsize=10)
        ax.tick_params(axis='y', labelsize=9)
        ax.grid(axis='y', alpha=0.25, linestyle='--')
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Bar chart saved  : {save_path}')


def plot_roc_overlay(results: dict, save_path: str):
    """Overlay ROC curves for all evaluated models."""
    fig, ax  = plt.subplots(figsize=(7, 6))
    colors   = ['#1565C0', '#B71C1C', '#2E7D32', '#6A1B9A']
    styles   = ['-',       '--',       '-.',       ':']

    for (name, r), c, ls in zip(results.items(), colors, styles):
        if 'fpr' not in r or 'tpr' not in r:
            continue
        fpr = np.array(r['fpr'])
        tpr = np.array(r['tpr'])
        ax.plot(fpr, tpr, color=c, linestyle=ls, lw=2,
                label=f'{name}  AUC={r["auc"]:.4f}  Acc={r["accuracy"]*100:.2f}%')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.35, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate',  fontsize=12)
    ax.set_title('LFW Verification — ROC Curves', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'ROC overlay saved: {save_path}')


def plot_speed_comparison(speed_results: dict, save_path: str):
    """Horizontal bar chart for inference speed."""
    if not speed_results:
        return

    names = list(speed_results.keys())
    fps   = [speed_results[n]['fps'] for n in names]
    ms    = [speed_results[n]['ms_per_img'] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    bars1 = ax1.barh(names, fps, color=['#1565C0', '#B71C1C'][:len(names)], height=0.5)
    ax1.set_xlabel('Frames per Second', fontsize=11)
    ax1.set_title('Throughput (FPS)', fontsize=12, fontweight='bold')
    for bar, v in zip(bars1, fps):
        ax1.text(bar.get_width() + max(fps)*0.01, bar.get_y() + bar.get_height()/2,
                 f'{v:.1f}', va='center', fontsize=10)

    bars2 = ax2.barh(names, ms, color=['#1565C0', '#B71C1C'][:len(names)], height=0.5)
    ax2.set_xlabel('ms / image', fontsize=11)
    ax2.set_title('Latency (ms)', fontsize=12, fontweight='bold')
    for bar, v in zip(bars2, ms):
        ax2.text(bar.get_width() + max(ms)*0.01, bar.get_y() + bar.get_height()/2,
                 f'{v:.1f}', va='center', fontsize=10)

    for ax in (ax1, ax2):
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=0.25, linestyle='--')

    plt.suptitle('Inference Speed Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Speed chart saved: {save_path}')


# ─────────────────────────────────────────────────────────────
#  METRICS TABLE (terminal)
# ─────────────────────────────────────────────────────────────

def print_table(results: dict, speed: dict = None):
    header_cols = ['Model', 'Accuracy', '±', 'AUC', 'TAR@0.1%', 'TAR@1%', 'Params(M)']
    if speed:
        header_cols += ['FPS', 'ms/img']

    row_fmt = '{:<28} {:>10} {:>6} {:>8} {:>10} {:>8} {:>10}'
    if speed:
        row_fmt += ' {:>7} {:>8}'

    sep = '─' * (90 + 20 * (speed is not None))

    print()
    print('╔' + '═' * (len(sep)-2) + '╗')
    print('║  LFW VERIFICATION RESULTS' + ' '*(len(sep)-28) + '║')
    print('╠' + '═' * (len(sep)-2) + '╣')
    print(sep)
    hdr = row_fmt.format(*header_cols)
    print(hdr)
    print(sep)

    for name, r in results.items():
        sp = speed.get(name, {}) if speed else {}
        row = [
            name,
            f'{r["accuracy"]*100:.2f}%',
            f'{r["accuracy_std"]*100:.2f}',
            f'{r["auc"]:.4f}',
            f'{r["tar_at_far_0.1%"]*100:.2f}%',
            f'{r["tar_at_far_1%"]*100:.2f}%',
            f'{r.get("params_M", "—"):>}',
        ]
        if speed:
            row += [
                f'{sp.get("fps", 0):.1f}',
                f'{sp.get("ms_per_img", 0):.1f}',
            ]
        print(row_fmt.format(*[str(v) for v in row]))

    print(sep)
    print()


# ─────────────────────────────────────────────────────────────
#  MAIN COMPARISON RUNNER
# ─────────────────────────────────────────────────────────────

def run_comparison(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device : {device}\n')

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}
    all_speed   = {}

    # ── 1.  DPAFD ─────────────────────────────────────────────
    print('─' * 50)
    print('[1/2]  DPAFD (Ours)')
    print('─' * 50)

    dpafd    = DPAFD(fpn_channels=args.fpn_channels)
    embedder = FaceEmbedder(dpafd, embed_dim=args.embed_dim)

    if args.dpafd_checkpoint:
        ckpt  = torch.load(args.dpafd_checkpoint, map_location=device)
        state = ckpt.get('model_state', ckpt)
        embedder.load_state_dict(state, strict=False)
        print(f'  Loaded: {args.dpafd_checkpoint}')
    else:
        print('  No checkpoint — random weights (sanity check)')

    embedder = embedder.to(device).eval()

    evaluator = LFWEvaluator(
        embedder,
        root       = args.lfw_root,
        pairs_file = args.pairs_file,
        img_size   = args.img_size,
        batch_size = args.batch_size,
        device     = device,
    )
    dpafd_res = evaluator.evaluate()
    dpafd_res['params_M'] = round(dpafd.count_params() / 1e6, 2)
    all_results['DPAFD (Ours)'] = dpafd_res

    # Speed benchmark
    if args.benchmark_speed:
        print('  Benchmarking DPAFD speed …')
        sp = benchmark_speed(
            lambda x: dpafd(x), device,
            input_size=(3, args.img_size, args.img_size),
        )
        all_speed['DPAFD (Ours)'] = sp
        print(f'  Speed: {sp["fps"]:.1f} FPS  ({sp["ms_per_img"]:.1f} ms)')

    # ── 2.  MTCNN baseline ────────────────────────────────────
    if MTCNN_AVAILABLE:
        print()
        print('─' * 50)
        print('[2/2]  Standard MTCNN (facenet-pytorch + InceptionResnetV1)')
        print('─' * 50)

        mtcnn_eval = MTCNNEvaluator(
            root       = args.lfw_root,
            pairs_file = args.pairs_file,
            img_size   = args.img_size,
            device     = device,
        )
        mtcnn_res = mtcnn_eval.evaluate(n_pairs=args.n_pairs)
        mtcnn_res['params_M'] = 29.6   # InceptionResnetV1 ~29.6M
        all_results['Standard MTCNN'] = mtcnn_res

        if args.benchmark_speed:
            print('  Benchmarking MTCNN speed …')
            dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            t_runs = []
            for _ in range(50):
                t0 = time.perf_counter()
                _ = mtcnn_eval.mtcnn(dummy_img)
                t_runs.append((time.perf_counter() - t0) * 1000)
            ms = float(np.mean(t_runs))
            all_speed['Standard MTCNN'] = {
                'ms_per_img': ms, 'ms_std': float(np.std(t_runs)),
                'fps': 1000.0 / ms,
            }
            print(f'  Speed: {1000/ms:.1f} FPS  ({ms:.1f} ms)')
    else:
        print('\n[2/2]  MTCNN skipped (facenet-pytorch not installed)')
        # Insert a placeholder from your paper's results for the table
        all_results['Standard MTCNN (reported)'] = {
            'n_pairs':        6000,
            'accuracy':       0.9593,    # from your paper Table 1
            'accuracy_std':   0.0,
            'auc':            0.9901,
            'tar_at_far_0.1%': 0.812,
            'tar_at_far_1%':   0.921,
            'params_M':        29.6,
            'fpr': [], 'tpr': [],
        }

    # ── Print table ───────────────────────────────────────────
    print_table(all_results, all_speed if args.benchmark_speed else None)

    # ── Save plots ────────────────────────────────────────────
    plot_comparison_bars(
        all_results,
        os.path.join(args.output_dir, 'comparison_bars.png'),
    )
    plot_roc_overlay(
        all_results,
        os.path.join(args.output_dir, 'roc_overlay.png'),
    )
    if args.benchmark_speed and all_speed:
        plot_speed_comparison(
            all_speed,
            os.path.join(args.output_dir, 'speed_comparison.png'),
        )

    # ── Save JSON summary ─────────────────────────────────────
    summary_path = os.path.join(args.output_dir, 'comparison_summary.json')
    saveable = {
        k: {mk: mv for mk, mv in v.items() if mk not in ('fpr', 'tpr')}
        for k, v in all_results.items()
    }
    if all_speed:
        for k in all_speed:
            saveable[k]['speed'] = all_speed[k]

    with open(summary_path, 'w') as f:
        json.dump(saveable, f, indent=2)
    print(f'Summary JSON saved: {summary_path}')
    print(f'All outputs in    : {args.output_dir}/')

    return all_results


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Compare DPAFD vs MTCNN on LFW')

    p.add_argument('--lfw_root',          required=True)
    p.add_argument('--dpafd_checkpoint',  default=None)
    p.add_argument('--pairs_file',        default='pairs.txt')
    p.add_argument('--output_dir',        default='comparison_results')

    p.add_argument('--fpn_channels',  type=int, default=64)
    p.add_argument('--embed_dim',     type=int, default=128)
    p.add_argument('--img_size',      type=int, default=112)
    p.add_argument('--batch_size',    type=int, default=64)
    p.add_argument('--n_pairs',       type=int, default=None,
                   help='Limit number of pairs for MTCNN (None = all)')

    p.add_argument('--benchmark_speed', action='store_true', default=False,
                   help='Run inference speed benchmarks')

    return p.parse_args()


if __name__ == '__main__':
    run_comparison(parse_args())