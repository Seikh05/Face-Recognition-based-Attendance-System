

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
#  FOCAL LOSS  (CenterNet variant)
# ─────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Modified focal loss for Gaussian heatmap regression.
    Handles the severe positive/negative imbalance in dense prediction.

    Positive positions (target == 1): standard focal penalty on misses.
    Negative positions (target <  1): down-weighted by (1-target)^β
        so near-center negatives contribute less.

    α=2 controls sharpness of focus on hard examples.
    β=4 controls how much soft-negative suppression happens.
    """
    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred   : [B, 1, H, W]  sigmoid scores ∈ (0,1)
        target : [B, 1, H, W]  Gaussian heatmap ∈ [0,1]
        """
        pos = target.eq(1).float()
        neg = target.lt(1).float()

        pos_loss = (
            torch.log(pred.clamp(min=1e-6))
            * torch.pow(1 - pred, self.alpha)
            * pos
        )
        neg_loss = (
            torch.log((1 - pred).clamp(min=1e-6))
            * torch.pow(pred,      self.alpha)
            * torch.pow(1 - target, self.beta)
            * neg
        )

        num_pos = pos.sum().clamp(min=1)
        return -(pos_loss + neg_loss).sum() / num_pos


# ─────────────────────────────────────────────────────────────
#  GIoU LOSS
# ─────────────────────────────────────────────────────────────

def giou_loss(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Generalised Intersection-over-Union loss.
    Gradient flows even when IoU=0, unlike standard IoU loss.

    pred, gt : [N, 4]  format (x1, y1, x2, y2)
    Returns  : scalar
    """
    px1, py1, px2, py2 = pred.unbind(1)
    gx1, gy1, gx2, gy2 = gt.unbind(1)

    inter_x1 = torch.max(px1, gx1)
    inter_y1 = torch.max(py1, gy1)
    inter_x2 = torch.min(px2, gx2)
    inter_y2 = torch.min(py2, gy2)

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    p_area = (px2 - px1) * (py2 - py1)
    g_area = (gx2 - gx1) * (gy2 - gy1)
    union  = p_area + g_area - inter + eps
    iou    = inter / union

    enc_x1 = torch.min(px1, gx1)
    enc_y1 = torch.min(py1, gy1)
    enc_x2 = torch.max(px2, gx2)
    enc_y2 = torch.max(py2, gy2)
    enc    = (enc_x2 - enc_x1) * (enc_y2 - enc_y1) + eps

    giou = iou - (enc - union) / enc
    return (1 - giou).mean()


# ─────────────────────────────────────────────────────────────
#  WING LOSS  (landmark regression)
# ─────────────────────────────────────────────────────────────

class WingLoss(nn.Module):
    """
    Wing loss for facial landmark regression.
    Uses log penalty for small errors (amplifies gradient near ground truth)
    and linear penalty for large errors (robust to outliers).

    Outperforms L1 and L2 for landmark tasks.
    Ref: Feng et al., CVPR 2018 — https://arxiv.org/abs/1711.06753
    """
    def __init__(self, w: float = 10.0, epsilon: float = 2.0):
        super().__init__()
        self.w   = w
        self.eps = epsilon
        self.C   = w - w * math.log(1 + w / epsilon)   # continuity constant

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = (pred - target).abs()
        loss = torch.where(
            diff < self.w,
            self.w * torch.log(1 + diff / self.eps),
            diff - self.C,
        )
        return loss.mean()


# ─────────────────────────────────────────────────────────────
#  GAUSSIAN TARGET UTILITIES
# ─────────────────────────────────────────────────────────────

def gaussian_radius(det_size: tuple, min_overlap: float = 0.7) -> int:
    """
    Compute Gaussian blob radius for a detection of given size,
    such that the Gaussian peak overlaps the GT box by ≥ min_overlap.
    Adapted from CornerNet / CenterNet.
    """
    h, w = det_size

    def solve(a, b, c):
        disc = b * b - 4 * a * c
        if disc < 0:
            return 0
        return (-b + math.sqrt(disc)) / (2 * a)

    r1 = solve(1,       h + w,        w * h * (1 - min_overlap) / (1 + min_overlap))
    r2 = solve(4,       2 * (h + w),  (1 - min_overlap) * w * h)
    r3 = solve(4 * min_overlap,
               -2 * min_overlap * (h + w),
               (min_overlap - 1) * w * h)

    return max(0, int(min(r1, r2, r3)))


def draw_gaussian(heatmap: torch.Tensor, center: tuple, radius: int) -> None:
    """
    Overlay a 2D Gaussian blob centred at `center` (row, col) onto `heatmap`.
    Uses element-wise maximum so overlapping faces compound cleanly.
    Modifies heatmap in-place.
    """
    H, W = heatmap.shape
    cy, cx = int(center[0]), int(center[1])
    diameter = 2 * radius + 1
    sigma    = diameter / 6.0

    # Build Gaussian kernel
    k = torch.arange(-radius, radius + 1, dtype=torch.float32)
    gauss = torch.exp(-(k[None] ** 2 + k[:, None] ** 2) / (2 * sigma ** 2))

    # Clip to heatmap bounds
    left   = min(cx,         radius)
    right  = min(W - cx,     radius + 1)
    top    = min(cy,         radius)
    bottom = min(H - cy,     radius + 1)

    if left + right <= 0 or top + bottom <= 0:
        return

    hm_roi  = heatmap[cy - top:cy + bottom, cx - left:cx + right]
    g_roi   = gauss[radius - top:radius + bottom, radius - left:radius + right]

    if hm_roi.shape == g_roi.shape:
        torch.maximum(hm_roi, g_roi.to(heatmap.device), out=hm_roi)


# ─────────────────────────────────────────────────────────────
#  TRAINING TARGET BUILDER
# ─────────────────────────────────────────────────────────────

def build_targets(
    boxes_batch: list,
    img_size: int,
    strides: list = (8, 16, 32),
    landmarks_batch: list = None,
) -> list:
    """
    Build per-FPN-level targets for a batch of images.

    Args:
        boxes_batch      : list of [N,4] tensors (x1,y1,x2,y2) per image
        img_size         : square input size (e.g. 640)
        strides          : FPN strides [8,16,32]
        landmarks_batch  : list of [N,10] tensors per image (optional)

    Returns:
        List of level-dicts, each containing batched tensors:
            heatmap  [B,1,H,W]
            wh       [B,2,H,W]
            offset   [B,2,H,W]
            landmarks[B,10,H,W]
            mask     [B,H,W]  bool — which positions are positive
    """
    B = len(boxes_batch)
    level_targets = []

    for stride in strides:
        fH = fW = img_size // stride

        # Assign face-size range per level
        size_lo = {8: 0,    16: 32,  32: 96 }[stride]
        size_hi = {8: 64,   16: 128, 32: 9999}[stride]

        hm   = torch.zeros(B, 1,  fH, fW)
        wh   = torch.zeros(B, 2,  fH, fW)
        off  = torch.zeros(B, 2,  fH, fW)
        lm   = torch.zeros(B, 10, fH, fW)
        mask = torch.zeros(B, fH, fW, dtype=torch.bool)

        for b, boxes in enumerate(boxes_batch):
            if boxes is None or len(boxes) == 0:
                continue
            lms_b = landmarks_batch[b] if landmarks_batch else None

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                face_w = x2 - x1
                face_h = y2 - y1
                face_size = max(face_w, face_h)

                if not (size_lo <= face_size < size_hi):
                    continue

                # Map to feature-map coords
                fx1, fy1 = x1 / stride, y1 / stride
                fx2, fy2 = x2 / stride, y2 / stride
                fcx, fcy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
                fbw, fbh  = fx2 - fx1, fy2 - fy1

                cx_int = int(fcx)
                cy_int = int(fcy)

                if not (0 <= cx_int < fW and 0 <= cy_int < fH):
                    continue
                if fbw < 0.5 or fbh < 0.5:
                    continue

                # Gaussian heatmap
                r = gaussian_radius((max(1, int(fbh)), max(1, int(fbw))))
                draw_gaussian(hm[b, 0], (cy_int, cx_int), r)

                # Regression targets at peak
                wh[b, 0, cy_int, cx_int] = fbw
                wh[b, 1, cy_int, cx_int] = fbh
                off[b, 0, cy_int, cx_int] = fcx - cx_int
                off[b, 1, cy_int, cx_int] = fcy - cy_int
                mask[b, cy_int, cx_int]   = True

                # Landmarks (if provided)
                if lms_b is not None and i < len(lms_b):
                    for k in range(5):
                        lm[b, k*2,   cy_int, cx_int] = lms_b[i, k*2]   / stride - cx_int
                        lm[b, k*2+1, cy_int, cx_int] = lms_b[i, k*2+1] / stride - cy_int

        level_targets.append({
            'heatmap':   hm,
            'wh':        wh,
            'offset':    off,
            'landmarks': lm,
            'mask':      mask,
        })

    return level_targets


# ─────────────────────────────────────────────────────────────
#  COMBINED DPAFD LOSS
# ─────────────────────────────────────────────────────────────

class DPAFDLoss(nn.Module):
    """
    L_total = λ_hm·L_focal + λ_wh·L_L1(wh) + λ_off·L_L1(offset) + λ_lm·L_wing
    Summed across all three FPN levels.
    """
    def __init__(
        self,
        lambda_hm:  float = 1.0,
        lambda_wh:  float = 0.1,
        lambda_off: float = 1.0,
        lambda_lm:  float = 0.1,
    ):
        super().__init__()
        self.focal      = FocalLoss()
        self.wing       = WingLoss()
        self.lambda_hm  = lambda_hm
        self.lambda_wh  = lambda_wh
        self.lambda_off = lambda_off
        self.lambda_lm  = lambda_lm

    def forward(self, preds: list, targets: list):
        """
        preds   : list of dicts (one per FPN level)
        targets : list of dicts (one per FPN level), keys as from build_targets

        Returns: (total_loss, loss_dict)
        """
        total = torch.tensor(0.0, device=preds[0]['heatmap'].device)
        ld = dict(hm=0.0, wh=0.0, off=0.0, lm=0.0)

        for pred, tgt in zip(preds, targets):
            dev = pred['heatmap'].device
            hm_tgt = tgt['heatmap'].to(dev)
            wh_tgt = tgt['wh'].to(dev)
            off_tgt = tgt['offset'].to(dev)
            lm_tgt  = tgt['landmarks'].to(dev)
            mask    = tgt['mask'].to(dev)           # [B,H,W]
            num_pos = mask.sum().clamp(min=1).float()

            # Heatmap focal loss
            l_hm  = self.focal(pred['heatmap'], hm_tgt)

            # Regression losses only at positive positions
            if mask.sum() > 0:
                # [B,H,W] → index along batch,H,W simultaneously
                p_wh  = pred['wh'].permute(0,2,3,1)[mask]       # [N,2]
                t_wh  = wh_tgt.permute(0,2,3,1)[mask]
                l_wh  = F.l1_loss(p_wh,  t_wh,  reduction='sum') / num_pos

                p_off = pred['offset'].permute(0,2,3,1)[mask]
                t_off = off_tgt.permute(0,2,3,1)[mask]
                l_off = F.l1_loss(p_off, t_off, reduction='sum') / num_pos

                p_lm  = pred['landmarks'].permute(0,2,3,1)[mask] # [N,10]
                t_lm  = lm_tgt.permute(0,2,3,1)[mask]
                l_lm  = self.wing(p_lm, t_lm)
            else:
                zero  = torch.tensor(0.0, device=dev)
                l_wh  = l_off = l_lm = zero

            lvl_loss = (
                self.lambda_hm  * l_hm
                + self.lambda_wh  * l_wh
                + self.lambda_off * l_off
                + self.lambda_lm  * l_lm
            )
            total = total + lvl_loss
            ld['hm']  += l_hm.item()
            ld['wh']  += l_wh.item()
            ld['off'] += l_off.item()
            ld['lm']  += l_lm.item()

        return total, ld