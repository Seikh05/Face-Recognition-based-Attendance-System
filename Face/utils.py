

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))


# ─────────────────────────────────────────────────────────────
#  FLOPs / Parameter Counter
# ─────────────────────────────────────────────────────────────

def count_flops(model: nn.Module, input_size=(1, 3, 640, 640)) -> dict:
    """
    Estimate FLOPs using torchinfo (preferred) or thop.
    Returns dict: {params_M, flops_G} or best-effort estimate.
    """
    params_M = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    try:
        from torchinfo import summary
        s = summary(model, input_size=input_size, verbose=0)
        flops_G = s.total_mult_adds / 1e9
        return {'params_M': round(params_M, 2), 'flops_G': round(flops_G, 2)}
    except ImportError:
        pass

    try:
        from thop import profile
        dummy = torch.randn(input_size)
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        return {'params_M': round(params_M, 2), 'flops_G': round(flops / 1e9, 2)}
    except ImportError:
        pass

    # Manual conservative estimate for this architecture
    # DPAFD @ 640×640: ~2.1 GFLOPs (from layer-by-layer count)
    return {
        'params_M': round(params_M, 2),
        'flops_G': '~2.1 (install torchinfo for exact count)',
    }


# ─────────────────────────────────────────────────────────────
#  DETECTION VISUALISER
# ─────────────────────────────────────────────────────────────

# Landmark connections: (idx_a, idx_b) drawing lines between keypoints
_LM_PAIRS = [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4)]

# Landmark labels
_LM_NAMES = ['L.Eye', 'R.Eye', 'Nose', 'L.Mouth', 'R.Mouth']


def draw_detections(
    img: np.ndarray,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    landmarks: Optional[torch.Tensor] = None,
    score_thresh: float = 0.3,
    box_color: tuple = (0, 230, 64),
    lm_color:  tuple = (255, 80, 0),
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    Draw detection results on a BGR image (numpy array).

    Args:
        img       : HxWx3 BGR image
        boxes     : [N,4] tensor in (x1,y1,x2,y2)
        scores    : [N]   confidence scores
        landmarks : [N,10] keypoints (optional)
        score_thresh: only draw detections above this

    Returns:
        annotated BGR image (copy)
    """
    out = img.copy()

    if len(boxes) == 0:
        return out

    for i, (box, sc) in enumerate(zip(boxes, scores)):
        if sc.item() < score_thresh:
            continue

        x1, y1, x2, y2 = box.int().tolist()
        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)

        label = f'{sc.item():.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 4, y1), box_color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

        if landmarks is not None and i < len(landmarks):
            lm = landmarks[i].tolist()
            pts = [(int(lm[k*2]), int(lm[k*2+1])) for k in range(5)]
            for pt in pts:
                cv2.circle(out, pt, 3, lm_color, -1)

    return out


# ─────────────────────────────────────────────────────────────
#  LFW HELPERS
# ─────────────────────────────────────────────────────────────

def center_crop_face(img: np.ndarray, out_size: int = 112) -> np.ndarray:
    """Centre-crop + resize — used as detection fallback on LFW."""
    H, W = img.shape[:2]
    s    = min(H, W)
    y0   = (H - s) // 2
    x0   = (W - s) // 2
    crop = img[y0:y0+s, x0:x0+s]
    return cv2.resize(crop, (out_size, out_size))


def detect_and_crop(
    img: np.ndarray,
    model,
    device: str,
    out_size: int = 112,
    conf: float   = 0.35,
    margin: float = 0.15,
) -> np.ndarray:
    """
    Detect the largest/most-confident face in `img` using DPAFD,
    add a margin, and return the cropped + resized face patch.
    Falls back to centre crop if no face is detected.
    """
    from model import decode_predictions  # local import to avoid circular

    H, W = img.shape[:2]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    t = torch.from_numpy(
        (img.astype(np.float32) / 255.0 - mean) / std
    ).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        preds   = model(t)
        results = decode_predictions(preds, conf_thresh=conf)

    if len(results) > 0 and len(results[0]['boxes']) > 0:
        best  = results[0]['scores'].argmax()
        x1, y1, x2, y2 = results[0]['boxes'][best].int().tolist()

        pad = int(max(x2 - x1, y2 - y1) * margin)
        x1  = max(0, x1 - pad)
        y1  = max(0, y1 - pad)
        x2  = min(W, x2 + pad)
        y2  = min(H, y2 + pad)

        crop = img[y1:y2, x1:x2]
        if crop.size > 0:
            return cv2.resize(crop, (out_size, out_size))

    return center_crop_face(img, out_size)


# ─────────────────────────────────────────────────────────────
#  CHECKPOINT UTILITIES
# ─────────────────────────────────────────────────────────────

def load_checkpoint(model: nn.Module, path: str, device: str = 'cpu',
                    strict: bool = False) -> dict:
    """
    Load a checkpoint safely, printing any missing / unexpected keys.
    Returns the full checkpoint dict.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Checkpoint not found: {path}')

    ckpt  = torch.load(path, map_location=device)
    state = ckpt.get('model_state', ckpt.get('state_dict', ckpt))

    missing, unexpected = model.load_state_dict(state, strict=strict)
    print(f'Loaded  : {path}')
    if missing:
        print(f'  Missing    ({len(missing)}): {missing[:5]}{"…" if len(missing)>5 else ""}')
    if unexpected:
        print(f'  Unexpected ({len(unexpected)}): {unexpected[:5]}{"…" if len(unexpected)>5 else ""}')

    return ckpt


# ─────────────────────────────────────────────────────────────
#  MODEL SUMMARY
# ─────────────────────────────────────────────────────────────

def print_model_summary(model: nn.Module, input_size=(1, 3, 640, 640)):
    """Print a brief summary including parameter counts per sub-module."""
    print(f'\n{"─"*55}')
    print(f'  Model: {model.__class__.__name__}')
    print(f'{"─"*55}')

    total = 0
    for name, m in model.named_children():
        p = sum(x.numel() for x in m.parameters())
        total += p
        print(f'  {name:<20} {m.__class__.__name__:<25} {p:>10,} params')

    print(f'{"─"*55}')
    print(f'  {"TOTAL":<46} {total:>10,} params')
    print(f'  {"":46} ({total/1e6:.2f}M)')
    print(f'{"─"*55}')

    fc = count_flops(model, input_size)
    print(f'  FLOPs: {fc["flops_G"]} G  |  Params: {fc["params_M"]}M\n')


# ─────────────────────────────────────────────────────────────
#  QUICK TEST
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from model import DPAFD
    m = DPAFD()
    print_model_summary(m)
    print('count_flops:', count_flops(m))