# -*- coding: utf-8 -*-
"""
Widened threshold-sweep diagnostic (pre-registration Appendix P.4). **Committed before it produced any number.**

Background: `scripts/eval_crossdomain.py` has swept **0.40-0.60** since its first version;
`results/stage0b/`, `results/cal_30141/`, L2 and stage 1 all use that range, which **must not be changed**
(changing it would make the batches incomparable). In the L2 pilot, however, the argmax at every GSD point sat
**at the lower bound 0.40**, so its `oracle` is only a **lower bound** of the true oracle and `frozen/oracle` only an **upper bound**.

This script sweeps separately over **0.05-0.95 (step 0.01)** and is used **only** to answer:
"of the collapse caused by degradation, how much is actually calibration drift (the threshold did not follow)?"

**Discipline (P.4 verbatim)**:
- the numbers of this script **must not** replace any number of the primary protocol and **must not** enter the results table;
- the `oracle` uses target labels and **can only be labelled an upper bound / diagnostic**, never as method performance;
- the primary curve is always the IoU at the **source-frozen threshold**.
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
    model, state, info = load_checkpoint(a.checkpoint, device)      # Appendix W.2: unified loader for GAPL / standard architectures
    targs = state['args']
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

    t0 = time.time()
    acc = sweep_counts(model, loader, device, WIDE)
    ious = {t: iou_of(acc[t]) for t in WIDE}
    best_t = max(WIDE, key=lambda t: ious[t])
    ft = round(a.frozen_threshold, 2)
    frozen = ious.get(ft)
    if frozen is None:                     # frozen threshold not on the grid: take the nearest point
        ft = min(WIDE, key=lambda t: abs(t - a.frozen_threshold))
        frozen = ious[ft]

    res = {
        'checkpoint': str(a.checkpoint), 'model_kind': info['kind'], 'arch': info['arch'], 'seed': info['seed'], 'target_view': str(tv),
        'manifest_key': a.manifest_key, 'n_query': len(q_ds),
        'sweep_range': [WIDE[0], WIDE[-1]], 'n_thresholds': len(WIDE),
        'frozen_threshold': a.frozen_threshold,
        'iou_frozen': frozen,
        'oracle_threshold': best_t, 'iou_oracle_wide': ious[best_t],
        'frozen_over_oracle': frozen / max(ious[best_t], 1e-12),
        'oracle_at_boundary': bool(best_t in (WIDE[0], WIDE[-1])),
        'iou_by_threshold': {str(t): ious[t] for t in WIDE},
        'seconds': time.time() - t0,
    }
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"  frozen {ft} -> IoU {frozen:.6f}")
    print(f"  widened-range oracle: threshold {best_t} -> IoU {ious[best_t]:.6f}   "
          f"frozen/oracle = {res['frozen_over_oracle']:.3f}"
          f"{'   (warning: still at the range boundary)' if res['oracle_at_boundary'] else ''}")
    print(f"  -> {out}   ({res['seconds']:.0f}s)")


if __name__ == '__main__':
    main()
