# -*- coding: utf-8 -*-
"""
Appendix W: training of the standard architectures on the source domain (WHU-1000 subset), **mirroring stage 1 (F.3) item by item**, with no tuning for the new architectures.
**Committed before it produced any number.**

Correspondence with main() of src/gaplsegnet_v5_ch5.py:
  - data subset and 800/200 split: the same rng logic as create_dataloaders (sample_seed=seed => bit-identical split to GAPL for the same seed);
  - training augmentation BuildingDataset(train=True) (flips + 90-degree rotations);
  - AdamW (lr, wd, betas 0.9/0.999), warm-up 5 + cosine (the same set_lr function, per iteration), AMP, gradient clipping 1.0;
  - loss: per-pixel BCE-with-logits (= the base term of GAPL arm A);
  - model selection: the epoch with the highest val IoU (threshold 0.5); the test (WHU test) IoU at threshold 0.5 goes to results.json (same definition as GAPL test_metrics.iou).
The checkpoint dict carries one extra key, 'arch', for the branch in scripts/model_loader.py.
"""
import argparse, json, math, pathlib, platform, socket, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import gaplsegnet_v5_ch5 as M
from arch_baselines import ArchSeg, ARCHS
sys.stdout.reconfigure(encoding='utf-8')


def split_indices(n_train_pairs: int, sample_size: int, sample_seed: int):
    """Sampling and split identical to M.create_dataloaders."""
    if sample_size > n_train_pairs:
        raise ValueError(f'sample_size={sample_size} exceeds {n_train_pairs} pairs')
    rng = np.random.default_rng(sample_seed)
    selected = rng.choice(n_train_pairs, sample_size, replace=False).tolist()
    rng.shuffle(selected)
    val_count = min(max(1, round(sample_size * 0.2)), sample_size - 2)
    return selected[:val_count], selected[val_count:]


def run_epoch(model, loader, optimizer, base_lrs, device, epoch, epochs, scaler, amp, train):
    model.train(train)
    metrics = M.SegmentationMetrics(); loss_sum = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for index, (images, masks) in enumerate(loader):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            if train:
                M.set_lr(optimizer, base_lrs, epoch - 1 + (index + 1) / max(len(loader), 1), epochs)
                optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp):
                out = model(images)
                loss = F.binary_cross_entropy_with_logits(out['logits'], masks.float())
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            metrics.update(out['predictions'], masks)
            loss_sum += float(loss.detach())
    res = metrics.compute(); res['loss'] = loss_sum / max(len(loader), 1)
    if train: res['lr'] = optimizer.param_groups[0]['lr']
    return res


def main():
    ap = argparse.ArgumentParser(description='Appendix W: standard-architecture baseline training (mirrors stage 1)')
    ap.add_argument('--arch', choices=ARCHS, required=True)
    ap.add_argument('--dataset-path', default='data_view/whu_building')
    ap.add_argument('--sample-size', type=int, default=1000)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--sample-seed', type=int, default=None, help='default = seed (bit-identical subset split to GAPL for the same seed)')
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--support-shots', type=int, default=4, help='placeholder: only written to args for create_dataloaders in eval_crossdomain')
    ap.add_argument('--gpu', type=int, default=None)
    ap.add_argument('--enable-amp', action='store_true')
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()
    if args.sample_seed is None: args.sample_seed = args.seed
    device = M.setup_device(args.gpu)
    M.seed_everything(args.seed)
    out_dir = pathlib.Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    root = pathlib.Path(args.dataset_path)
    train_plain = M.BuildingDataset(root / 'train' / 'image', root / 'train' / 'label', train=False)
    train_aug = M.BuildingDataset(root / 'train' / 'image', root / 'train' / 'label', train=True)
    test_plain = M.BuildingDataset(root / 'test' / 'image', root / 'test' / 'label', train=False)
    val_idx, train_idx = split_indices(len(train_plain), args.sample_size, args.sample_seed)
    common = {'num_workers': args.num_workers, 'pin_memory': True}
    if args.num_workers: common['persistent_workers'] = True
    train_loader = DataLoader(Subset(train_aug, train_idx), batch_size=args.batch_size, shuffle=True, **common)
    val_loader = DataLoader(Subset(train_plain, val_idx), batch_size=args.batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_plain, batch_size=8, shuffle=False, **common)

    model = ArchSeg(args.arch, pretrained=True).to(device)
    print(f'arch={args.arch} params={model.n_params()/1e6:.2f}M  train={len(train_idx)} val={len(val_idx)} test={len(test_plain)}', flush=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    base_lrs = [g['lr'] for g in optimizer.param_groups]
    amp = bool(args.enable_amp and device.type == 'cuda')
    scaler = GradScaler(enabled=amp)
    history, best_iou, best_epoch = [], -1.0, 0
    ckpt = out_dir / 'best_arch.pth'
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, base_lrs, device, epoch, args.epochs, scaler, amp, True)
        va = run_epoch(model, val_loader, optimizer, base_lrs, device, epoch, args.epochs, scaler, amp, False)
        rec = {'epoch': epoch, 'train': tr, 'val': va}; history.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        if va['iou'] > best_iou:
            best_iou, best_epoch = va['iou'], epoch
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'best_val_iou': best_iou, 'args': vars(args),
                        'arch': args.arch, 'version': 'arch-baseline-W'}, ckpt)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state['model'])
    te = run_epoch(model, test_loader, optimizer, base_lrs, device, args.epochs, args.epochs, scaler, amp, False)
    results = {'version': 'arch-baseline-W', 'arch': args.arch, 'protocol': 'single-input segmentation; support ignored',
               'elapsed_seconds': time.time() - started, 'train_samples': len(train_idx), 'validation_samples': len(val_idx),
               'best_epoch': best_epoch, 'best_validation_iou': best_iou, 'test_metrics': te, 'args': vars(args),
               'n_params': model.n_params(), 'hostname': socket.gethostname(), 'torch': torch.__version__,
               'platform': platform.platform(), 'history': history}
    (out_dir / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"DONE arch={args.arch} seed={args.seed} best_epoch={best_epoch} val={best_iou:.6f} test={te['iou']:.6f} "
          f"elapsed={results['elapsed_seconds']/60:.1f}min", flush=True)


if __name__ == '__main__':
    main()
