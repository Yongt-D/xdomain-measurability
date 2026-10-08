# -*- coding: utf-8 -*-
"""
Widened threshold sweep that also stores the per-threshold pixel counts (pre-registration Appendix AD.2).
**Committed before it produced any number.**

This is a copy of `threshold_range_diag.py` (Appendix P.4) with one addition: the output records, for every
threshold of the 0.05-0.95 grid, the global (tp, fp, fn) counts and the total number of query pixels, so that
the predicted-positive fraction, precision, recall and a partial PR curve can be computed afterwards.
`threshold_range_diag.py` itself is left unchanged.

Discipline (as in P.4): these numbers are diagnostic, never method performance; the primary curve is the IoU
at the source-frozen threshold; any threshold chosen with target labels is labelled as such.
"""
import argparse, json, sys, pathlib, time
import torch
from torch.utils.data import DataLoader
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import gaplsegnet_v5_ch5 as M
from eval_crossdomain import FixedSupportSet, sweep_counts, iou_of
from model_loader import load_checkpoint

WIDE = [round(0.05 + 0.01 * i, 2) for i in range(91)]      # 0.05 .. 0.95


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--target-view', required=True)
    ap.add_argument('--manifest', default=str(ROOT / 'results' / 'support_manifests' / 'inria_support_manifests.json'))
    ap.add_argument('--manifest-key', default='fmin0.01_K4_r0')
    ap.add_argument('--frozen-threshold', type=float, required=True,
                    help='the threshold selected on the source validation set and frozen for this checkpoint (from the primary-protocol result JSON)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--num-workers', type=int, default=0)
    ap.add_argument('--gpu', type=int, default=None)
    a = ap.parse_args()

    device = M.setup_device(a.gpu)
    model, state, info = load_checkpoint(a.checkpoint, device)
    print(f"loaded {a.checkpoint}  {info['desc']}  frozen threshold={a.frozen_threshold}")

    tv = pathlib.Path(a.target_view)
    man = json.loads(pathlib.Path(a.manifest).read_text(encoding='utf-8'))[a.manifest_key]
    q_ds = M.BuildingDataset(tv / 'test' / 'image', tv / 'test' / 'label', train=False)
    s_ds = M.BuildingDataset(tv / 'train' / 'image', tv / 'train' / 'label', train=False)
    s_names = sorted(p.name for p in (tv / 'train' / 'image').iterdir())
    pos = {n: i for i, n in enumerate(s_names)}
    ds = FixedSupportSet(q_ds, s_ds, [pos[f] for f in man['files']])
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                        num_workers=a.num_workers, pin_memory=False)
    n_pixels = len(q_ds) * int(ds[0][1].numel())

    t0 = time.time()
    acc = sweep_counts(model, loader, device, WIDE)
    ious = {t: iou_of(acc[t]) for t in WIDE}
    best_t = max(WIDE, key=lambda t: ious[t])
    ft = round(a.frozen_threshold, 2)
    frozen = ious.get(ft)
    if frozen is None:
        ft = min(WIDE, key=lambda t: abs(t - a.frozen_threshold))
        frozen = ious[ft]
    label_pos = acc[WIDE[0]][0] + acc[WIDE[0]][2]      # tp + fn: the same at every threshold

    res = {
        'checkpoint': str(a.checkpoint), 'model_kind': info['kind'], 'arch': info['arch'], 'seed': info['seed'], 'target_view': str(tv),
        'manifest_key': a.manifest_key, 'n_query': len(q_ds), 'n_pixels': n_pixels, 'label_positive_pixels': int(label_pos),
        'sweep_range': [WIDE[0], WIDE[-1]], 'n_thresholds': len(WIDE),
        'frozen_threshold': a.frozen_threshold,
        'iou_frozen': frozen,
        'oracle_threshold': best_t, 'iou_oracle_wide': ious[best_t],
        'frozen_over_oracle': frozen / max(ious[best_t], 1e-12),
        'oracle_at_boundary': bool(best_t in (WIDE[0], WIDE[-1])),
        'iou_by_threshold': {str(t): ious[t] for t in WIDE},
        'counts_by_threshold': {str(t): [int(c) for c in acc[t]] for t in WIDE},   # [tp, fp, fn]
        'seconds': time.time() - t0,
    }
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"  frozen {ft} -> IoU {frozen:.6f}   oracle {best_t} -> IoU {ious[best_t]:.6f}   label positive fraction {label_pos / n_pixels:.4f}")
    print(f"  -> {out}   ({res['seconds']:.0f}s)")


if __name__ == '__main__':
    main()
