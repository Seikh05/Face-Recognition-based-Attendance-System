

import os
import sys
import time
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# ── local imports ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from model   import DPAFD
from losses  import DPAFDLoss
from dataset import WiderFaceDataset, collate_fn


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def move_targets(targets, device):
    """Move all tensors in a list-of-dicts to `device`."""
    return [{k: v.to(device, non_blocking=True) for k, v in t.items()}
            for t in targets]


def pretty(loss_dict: dict) -> str:
    return '  '.join(f'{k}:{v:.3f}' for k, v in loss_dict.items())


# ─────────────────────────────────────────────────────────────
#  TRAIN ONE EPOCH
# ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch, cfg):
    model.train()
    running = 0.0
    comp_acc = dict(hm=0.0, wh=0.0, off=0.0, lm=0.0)
    steps = len(loader)

    pbar = tqdm(loader, desc=f'Epoch {epoch:3d}', leave=False, ncols=100)
    for step, (imgs, targets) in enumerate(pbar):
        imgs    = imgs.to(device, non_blocking=True)
        targets = move_targets(targets, device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=cfg.amp):
            preds = model(imgs)
            loss, ld = criterion(preds, targets)

        if cfg.amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        running += loss.item()
        for k in comp_acc:
            comp_acc[k] += ld[k]

        if step % max(1, steps // 10) == 0:
            pbar.set_postfix({'loss': f'{loss.item():.3f}', **{k: f'{v:.3f}' for k, v in ld.items()}})

    n = len(loader)
    return running / n, {k: v / n for k, v in comp_acc.items()}


# ─────────────────────────────────────────────────────────────
#  VALIDATION
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device, cfg):
    model.eval()
    running = 0.0
    for imgs, targets in tqdm(loader, desc='  Val', leave=False, ncols=80):
        imgs    = imgs.to(device, non_blocking=True)
        targets = move_targets(targets, device)
        with autocast(enabled=cfg.amp):
            preds = model(imgs)
            loss, _ = criterion(preds, targets)
        running += loss.item()
    return running / len(loader)


# ─────────────────────────────────────────────────────────────
#  MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────

def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    print(f'Config : {vars(cfg)}\n')

    # ── datasets ──────────────────────────────────────────────
    train_ds = WiderFaceDataset(cfg.wider_root, 'train', cfg.input_size)
    val_ds   = WiderFaceDataset(cfg.wider_root, 'val',   cfg.input_size)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.workers, collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda'), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.workers, collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda'),
    )

    # ── model ─────────────────────────────────────────────────
    model = DPAFD(fpn_channels=cfg.fpn_channels).to(device)
    pc    = model.count_params_by_module()
    print(f'Model params : {pc["total"]:,}  ({pc["total"]/1e6:.2f}M)')
    print(f'  Backbone   : {pc["backbone"]:,}')
    print(f'  FPN        : {pc["fpn"]:,}')
    print(f'  Head       : {pc["head"]:,}\n')

    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        print(f'Resumed from {cfg.resume}')

    # ── optimiser + schedule ──────────────────────────────────
    criterion = DPAFDLoss(
        lambda_hm=1.0, lambda_wh=0.1, lambda_off=1.0, lambda_lm=0.1
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.wd
    )

    # Linear warmup for first `warmup_epochs` epochs
    def lr_lambda(ep):
        if ep < cfg.warmup:
            return (ep + 1) / cfg.warmup
        # Cosine decay after warmup
        t = (ep - cfg.warmup) / max(cfg.epochs - cfg.warmup, 1)
        return 0.5 * (1 + torch.tensor(t * 3.14159265).cos().item()) * (1 - 1e-3) + 1e-3

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = GradScaler(enabled=cfg.amp)

    # ── training loop ─────────────────────────────────────────
    os.makedirs(cfg.save_dir, exist_ok=True)
    history   = dict(train=[], val=[], lr=[])
    best_val  = float('inf')

    print(f'{"Epoch":>6}  {"Train":>8}  {"Val":>8}  {"LR":>10}  {"Time":>6}  Components')
    print('─' * 80)

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_loss, comp = train_epoch(
            model, train_loader, criterion,
            optimizer, scaler, device, epoch, cfg
        )
        val_loss = validate(model, val_loader, criterion, device, cfg)
        scheduler.step()

        lr  = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        print(
            f'{epoch:6d}  {train_loss:8.4f}  {val_loss:8.4f}  '
            f'{lr:10.6f}  {elapsed:5.0f}s  [{pretty(comp)}]'
        )

        history['train'].append(train_loss)
        history['val'].append(val_loss)
        history['lr'].append(lr)

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'val_loss':    val_loss,
                'cfg':         vars(cfg),
            }, os.path.join(cfg.save_dir, 'best.pth'))
            print(f'  ✓ best model saved  (val={val_loss:.4f})')

        # Periodic checkpoint
        if epoch % cfg.save_every == 0:
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'val_loss':    val_loss,
            }, os.path.join(cfg.save_dir, f'epoch_{epoch:04d}.pth'))

        with open(os.path.join(cfg.save_dir, 'history.json'), 'w') as fh:
            json.dump(history, fh, indent=2)

    print(f'\nTraining complete.  Best val loss: {best_val:.4f}')
    print(f'Checkpoints saved to: {cfg.save_dir}')


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train DPAFD on WIDER FACE')

    # Paths
    p.add_argument('--wider_root',  required=True,             help='WIDER FACE root directory')
    p.add_argument('--save_dir',    default='checkpoints',      help='Directory for checkpoints')
    p.add_argument('--resume',      default=None,               help='Resume from checkpoint path')

    # Architecture
    p.add_argument('--fpn_channels', type=int, default=64,      help='FPN output channels')
    p.add_argument('--input_size',   type=int, default=640,     help='Square input size')

    # Training
    p.add_argument('--epochs',      type=int,   default=100)
    p.add_argument('--batch_size',  type=int,   default=16)
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--wd',          type=float, default=1e-4,   help='Weight decay')
    p.add_argument('--warmup',      type=int,   default=5,      help='Warmup epochs')
    p.add_argument('--grad_clip',   type=float, default=10.0)
    p.add_argument('--workers',     type=int,   default=4)
    p.add_argument('--save_every',  type=int,   default=10)
    p.add_argument('--amp',         action='store_true', default=True,
                   help='Mixed-precision training (default: on)')
    p.add_argument('--no_amp',      dest='amp', action='store_false')

    return p.parse_args()


if __name__ == '__main__':
    main(parse_args())