# -*- coding: utf-8 -*-
"""
Cross-machine environment calibration (pre-registration Appendix L.2.3). **Criteria E1/E2/E3 were fixed in commit `cc8fb64`.**

Compares the cross-domain IoU of the same checkpoint, manifest and data on different machines:
  reference = dyt (`results/stage0b/<tag>_seed<s>_fmin0.01_K4_r0.json`, committed long before)
  other     = another machine (`results/cal_<host>/<tag>_seed<s>_fmin0.01_K4_r0.json`)

Criteria (L.2.3, the largest |d| over the 6 checkpoints):
  E1  max|d| < 0.0005            -> machines interchangeable; inference-type work (L2) may run across machines
  E2  0.0005 <= max|d| < 0.0037  -> L2 may run across machines, but every GSD curve must stay on one machine, and the drift must be reported
  E3  max|d| >= 0.0037           -> no cross-machine work

Whatever the tier, **stage-1 training and criterion evaluation stay on dyt** (L.2.4, never relaxed).

Disclosure: this script was written after 5 of the compared values had been seen in run logs.
The criteria E1/E2/E3 and their thresholds were committed before that; the script only subtracts and takes the maximum, with no discretion.
"""
import json, sys, pathlib, argparse
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
TAGS = ('Aplain', 'Cconfidence')
SEEDS = (0, 1, 2)
KEY = 'fmin0.01_K4_r0'
E1, E2 = 0.0005, 0.0037


def iou(p):
    return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))['target']['iou']


ap = argparse.ArgumentParser()
ap.add_argument('--other', required=True, help='comparison directory, e.g. results/cal_30141')
ap.add_argument('--label', default=None)
a = ap.parse_args()
other = ROOT / a.other
label = a.label or other.name

print(f'cross-machine calibration: {label}  vs  dyt (reference)')
print(f"{'arm':14s}{'seed':>5s}{'dyt':>12s}{f'{label}':>14s}{'d':>13s}")
diffs = []
for tag in TAGS:
    for s in SEEDS:
        p_dyt = ROOT / 'results' / 'stage0b' / f'{tag}_seed{s}_{KEY}.json'
        p_oth = other / f'{tag}_seed{s}_{KEY}.json'
        if not p_oth.exists():
            print(f'{tag:14s}{s:>5d}  missing {p_oth.name}')
            continue
        a_, b_ = iou(p_dyt), iou(p_oth)
        d = b_ - a_
        diffs.append(abs(d))
        print(f'{tag:14s}{s:>5d}{a_:>12.6f}{b_:>14.6f}{d:>+13.6f}')

if not diffs:
    print('  nothing to compare'); sys.exit(1)
m = float(max(diffs))
print(f'\n  n = {len(diffs)}   max|d| = {m:.8f}   mean|d| = {np.mean(diffs):.8f}')
if m < E1:
    v = f'**E1**: max|d| < {E1} => machines interchangeable; inference-type work (L2) may run across machines'
elif m < E2:
    v = f'**E2**: {E1} <= max|d| < {E2} => L2 may run across machines, but each GSD curve must stay on one machine, and the drift must be reported'
else:
    v = f'**E3**: max|d| >= {E2} => **no cross-machine work**; L2 queues on dyt'
print(f'  => {v}')
print(f'\n  note (L.2.4): whatever the tier, **stage-1 training and criterion evaluation stay on dyt**; never relaxed.')
