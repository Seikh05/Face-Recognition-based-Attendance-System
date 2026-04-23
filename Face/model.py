

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms


# ─────────────────────────────────────────────────────────────
#  PRIMITIVE BLOCKS
# ─────────────────────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    """Conv2d + BatchNorm + ReLU6"""
    def __init__(self, in_c, out_c, k=3, s=1, p=1, d=1, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, dilation=d, groups=groups, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class InvertedResidual(nn.Module):
    """MobileNetV2 Inverted Residual Block"""
    def __init__(self, in_c, out_c, stride=1, expand_ratio=6):
        super().__init__()
        hidden = int(in_c * expand_ratio)
        self.use_res = (stride == 1 and in_c == out_c)
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_c, hidden, k=1, p=0))
        layers += [
            ConvBNReLU(hidden, hidden, k=3, s=stride, p=1, groups=hidden),
            nn.Conv2d(hidden, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)


# ─────────────────────────────────────────────────────────────
#  DUAL-PATH ENCODER BLOCK
# ─────────────────────────────────────────────────────────────

class DualPathBlock(nn.Module):
    """
    Parallel texture + semantic extraction paths, fused with
    learned channel-wise attention weights.

    Texture path  : standard 3×3 conv  → fine edges / high-freq detail
    Semantic path : dilated 3×3 (d=2)  → wider receptive field / semantics

    The fusion attention lets the network decide, per-channel and per-image,
    how much texture vs semantic information to weight — critical for
    handling varied illumination and partial occlusion.
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        mid = out_c // 2

        # Texture path: two 3×3 convs
        self.tex = nn.Sequential(
            nn.Conv2d(in_c, mid, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU6(inplace=True),
            nn.Conv2d(mid, mid, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU6(inplace=True),
        )

        # Semantic path: dilated 3×3 (effective 5×5 receptive field)
        self.sem = nn.Sequential(
            nn.Conv2d(in_c, mid, 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU6(inplace=True),
            nn.Conv2d(mid, mid, 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU6(inplace=True),
        )

        # Fusion attention: squeeze → excite over concatenated features
        self.fusion = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_c, max(out_c // 4, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(max(out_c // 4, 8), out_c),
            nn.Sigmoid(),
        )

        # Skip connection projection if needed
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c),
            )
            if in_c != out_c else nn.Identity()
        )

    def forward(self, x):
        t = self.tex(x)
        s = self.sem(x)
        concat = torch.cat([t, s], dim=1)           # [B, out_c, H, W]
        w = self.fusion(concat).unsqueeze(-1).unsqueeze(-1)  # [B, out_c, 1, 1]
        return concat * w + self.skip(x)


# ─────────────────────────────────────────────────────────────
#  CBAM ATTENTION
# ─────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """Squeeze both avg-pool and max-pool, excite via shared MLP"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1, bias=False),
        )
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.mx  = nn.AdaptiveMaxPool2d(1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return self.sig(self.mlp(self.avg(x)) + self.mlp(self.mx(x)))


class SpatialAttention(nn.Module):
    """Channel-wise avg + max → 7×7 conv → spatial gate"""
    def __init__(self, kernel=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel, padding=kernel // 2, bias=False)
        self.sig  = nn.Sigmoid()

    def forward(self, x):
        avg_c = x.mean(dim=1, keepdim=True)
        max_c, _ = x.max(dim=1, keepdim=True)
        return self.sig(self.conv(torch.cat([avg_c, max_c], dim=1)))


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Applies channel attention THEN spatial attention sequentially.
    Improvement over SE blocks (channel-only): spatial gate forces the
    network to locate where the face is, not just which features matter.
    """
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = x * self.ca(x)   # channel recalibration
        x = x * self.sa(x)   # spatial localization
        return x


# ─────────────────────────────────────────────────────────────
#  BACKBONE
# ─────────────────────────────────────────────────────────────

class DPAFDBackbone(nn.Module):
    """
    Lightweight backbone that outputs three feature maps:
      C3  @ stride=8   (32 ch)   — small face detail
      C4  @ stride=16  (64 ch)   — medium faces
      C5  @ stride=32  (96 ch)   — large faces / global context

    DualPathBlocks are inserted at C3, C4, C5 to capture both
    texture and semantic information at each scale.
    """
    def __init__(self):
        super().__init__()

        # Stem: 3→16,  /2
        self.stem = ConvBNReLU(3, 16, k=3, s=2, p=1)

        # Stage 1: 16→24, /4
        self.s1 = nn.Sequential(
            InvertedResidual(16, 16, stride=1, expand_ratio=1),
            InvertedResidual(16, 24, stride=2, expand_ratio=6),
            InvertedResidual(24, 24, stride=1, expand_ratio=6),
        )

        # Stage 2: 24→32 + DualPath, /8  → C3
        self.s2 = nn.Sequential(
            InvertedResidual(24, 32, stride=2, expand_ratio=6),
            DualPathBlock(32, 32),
            InvertedResidual(32, 32, stride=1, expand_ratio=6),
        )

        # Stage 3: 32→64 + DualPath, /16 → C4
        self.s3 = nn.Sequential(
            InvertedResidual(32, 64, stride=2, expand_ratio=6),
            DualPathBlock(64, 64),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
        )

        # Stage 4: 64→96 + DualPath, /32 → C5
        self.s4 = nn.Sequential(
            InvertedResidual(64, 96, stride=2, expand_ratio=6),
            DualPathBlock(96, 96),
            InvertedResidual(96, 96, stride=1, expand_ratio=6),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x  = self.stem(x)
        x  = self.s1(x)
        c3 = self.s2(x)   # [B, 32, H/8,  W/8]
        c4 = self.s3(c3)  # [B, 64, H/16, W/16]
        c5 = self.s4(c4)  # [B, 96, H/32, W/32]
        return c3, c4, c5


# ─────────────────────────────────────────────────────────────
#  FEATURE PYRAMID NETWORK
# ─────────────────────────────────────────────────────────────

class FPN(nn.Module):
    """
    3-level top-down FPN: P3, P4, P5.
    Top-down pathway merges high-level semantics into fine-grained maps.
    CBAM applied at each level after merging.
    """
    def __init__(self, in_channels=(32, 64, 96), out_channels=64):
        super().__init__()
        c3, c4, c5 = in_channels

        # Lateral 1×1 projections
        self.lat5 = nn.Conv2d(c5, out_channels, 1)
        self.lat4 = nn.Conv2d(c4, out_channels, 1)
        self.lat3 = nn.Conv2d(c3, out_channels, 1)

        # 3×3 output smoothing
        self.out5 = ConvBNReLU(out_channels, out_channels)
        self.out4 = ConvBNReLU(out_channels, out_channels)
        self.out3 = ConvBNReLU(out_channels, out_channels)

        # CBAM on each pyramid level
        self.cbam5 = CBAM(out_channels)
        self.cbam4 = CBAM(out_channels)
        self.cbam3 = CBAM(out_channels)

    def forward(self, c3, c4, c5):
        # Top-down merge
        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')

        # Smooth then CBAM
        p5 = self.cbam5(self.out5(p5))
        p4 = self.cbam4(self.out4(p4))
        p3 = self.cbam3(self.out3(p3))

        return p3, p4, p5   # increasing strides: 8, 16, 32


# ─────────────────────────────────────────────────────────────
#  ANCHOR-FREE DETECTION HEAD
# ─────────────────────────────────────────────────────────────

class DetectionHead(nn.Module):
    """
    CenterNet-style anchor-free head.
    Shared weights across all FPN levels (better generalization).

    Per pixel predicts:
      heatmap  [1]   — Gaussian blob at face center
      wh       [2]   — face width + height (in feature-map units)
      offset   [2]   — fractional center offset (sub-pixel accuracy)
      landmarks[10]  — 5 facial keypoints × (dx, dy) offsets
    """
    def __init__(self, in_channels=64):
        super().__init__()
        mid = in_channels

        self.shared = nn.Sequential(
            ConvBNReLU(mid, mid),
            ConvBNReLU(mid, mid),
        )

        # Heatmap: sigmoid output ∈ (0,1)
        self.hm = nn.Sequential(
            ConvBNReLU(mid, mid // 2),
            nn.Conv2d(mid // 2, 1, 1),
        )

        # Width/height: ReLU (positive)
        self.wh = nn.Sequential(
            ConvBNReLU(mid, mid // 2),
            nn.Conv2d(mid // 2, 2, 1),
            nn.ReLU(),
        )

        # Sub-pixel offset: tanh ∈ (-1, 1)
        self.off = nn.Sequential(
            ConvBNReLU(mid, mid // 2),
            nn.Conv2d(mid // 2, 2, 1),
            nn.Tanh(),
        )

        # Landmark offsets: 5 × (dx, dy) = 10
        self.lm = nn.Sequential(
            ConvBNReLU(mid, mid),
            ConvBNReLU(mid, mid // 2),
            nn.Conv2d(mid // 2, 10, 1),
        )

        self._init_bias()

    def _init_bias(self):
        # Prior probability = 0.1 → bias = log(0.1/0.9) ≈ -2.19
        # Drastically reduces false positives early in training
        for m in self.hm.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.bias, -2.19)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None and m not in [
                    sub for sub in self.hm.modules()
                    if isinstance(sub, nn.Conv2d)
                ]:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        feat = self.shared(x)
        return {
            'heatmap':   torch.sigmoid(self.hm(feat)),
            'wh':        self.wh(feat),
            'offset':    self.off(feat),
            'landmarks': self.lm(feat),
        }


# ─────────────────────────────────────────────────────────────
#  FULL DPAFD MODEL
# ─────────────────────────────────────────────────────────────

class DPAFD(nn.Module):
    """
    DualPath Anchor-Free Face Detector.

    Face scale assignment per FPN level:
      P3 (stride=8)  — small  faces  < 64 px
      P4 (stride=16) — medium faces  64–128 px
      P5 (stride=32) — large  faces  > 128 px
    """
    def __init__(self, fpn_channels=64):
        super().__init__()
        self.backbone = DPAFDBackbone()
        self.fpn      = FPN(in_channels=(32, 64, 96), out_channels=fpn_channels)
        self.head     = DetectionHead(in_channels=fpn_channels)
        self.strides  = [8, 16, 32]

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.fpn(c3, c4, c5)

        outputs = []
        for feat, stride in zip([p3, p4, p5], self.strides):
            pred = self.head(feat)
            pred['stride'] = stride
            outputs.append(pred)
        return outputs

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_params_by_module(self):
        return {
            'backbone': sum(p.numel() for p in self.backbone.parameters()),
            'fpn':      sum(p.numel() for p in self.fpn.parameters()),
            'head':     sum(p.numel() for p in self.head.parameters()),
            'total':    self.count_params(),
        }


# ─────────────────────────────────────────────────────────────
#  POST-PROCESSING: decode to bounding boxes + NMS
# ─────────────────────────────────────────────────────────────

def decode_predictions(outputs, conf_thresh=0.3, nms_thresh=0.45):
    """
    Decode per-level anchor-free predictions to bounding boxes.

    Args:
        outputs     : list of dicts (one per FPN level), batch first
        conf_thresh : minimum heatmap score
        nms_thresh  : IoU threshold for NMS

    Returns:
        list of dicts (one per image in batch):
            {'boxes': [N,4], 'scores': [N], 'landmarks': [N,10]}
    """
    device = outputs[0]['heatmap'].device
    B = outputs[0]['heatmap'].shape[0]

    all_boxes  = [[] for _ in range(B)]
    all_scores = [[] for _ in range(B)]
    all_lms    = [[] for _ in range(B)]

    for pred in outputs:
        hm     = pred['heatmap']    # [B,1,H,W]
        wh     = pred['wh']         # [B,2,H,W]
        offset = pred['offset']     # [B,2,H,W]
        lms    = pred['landmarks']  # [B,10,H,W]
        stride = pred['stride']

        _, _, H, W = hm.shape

        # Build grids
        ys = torch.arange(H, device=device, dtype=torch.float32)
        xs = torch.arange(W, device=device, dtype=torch.float32)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')   # [H,W]

        scores = hm[:, 0]  # [B,H,W]

        for b in range(B):
            sc = scores[b]                          # [H,W]
            mask = sc > conf_thresh
            if mask.sum() == 0:
                continue

            s  = sc[mask]                           # [N]
            cx = (gx[mask] + offset[b, 0][mask]) * stride
            cy = (gy[mask] + offset[b, 1][mask]) * stride
            bw = wh[b, 0][mask] * stride
            bh = wh[b, 1][mask] * stride

            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2

            boxes = torch.stack([x1, y1, x2, y2], dim=1)   # [N,4]
            lm    = lms[b][:, mask].T * stride              # [N,10]

            all_boxes[b].append(boxes)
            all_scores[b].append(s)
            all_lms[b].append(lm)

    results = []
    for b in range(B):
        if not all_boxes[b]:
            results.append({
                'boxes':     torch.zeros((0, 4)),
                'scores':    torch.zeros(0),
                'landmarks': torch.zeros((0, 10)),
            })
            continue

        boxes  = torch.cat(all_boxes[b],  dim=0)
        scores = torch.cat(all_scores[b], dim=0)
        lm     = torch.cat(all_lms[b],    dim=0)

        keep   = nms(boxes, scores, nms_thresh)
        results.append({
            'boxes':     boxes[keep].cpu(),
            'scores':    scores[keep].cpu(),
            'landmarks': lm[keep].cpu(),
        })

    return results


# ─────────────────────────────────────────────────────────────
#  QUICK SANITY CHECK
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    model = DPAFD(fpn_channels=64)
    x = torch.randn(2, 3, 640, 640)
    preds = model(x)

    pc = model.count_params_by_module()
    print(f"Backbone : {pc['backbone']:>8,} params")
    print(f"FPN      : {pc['fpn']:>8,} params")
    print(f"Head     : {pc['head']:>8,} params")
    print(f"Total    : {pc['total']:>8,} params  ({pc['total']/1e6:.2f}M)")
    print()
    for i, p in enumerate(preds):
        s = p['stride']
        print(f"P{i+3} (stride={s:2d}): "
              f"hm={tuple(p['heatmap'].shape)}  "
              f"wh={tuple(p['wh'].shape)}  "
              f"lm={tuple(p['landmarks'].shape)}")