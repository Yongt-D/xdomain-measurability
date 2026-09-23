# -*- coding: utf-8 -*-
"""
Attribution diagnostic for the O.3 cross-check: cross-machine drift at equal thresholds.

**Disclosure: this script was written after gate 1 of `l2_analyze.py` had reported a failure**
(2026-09-11, arm A seed105, |d| = 0.00071).
It **does not change** the gate-1 criterion (still |IoU_{f=1} - IoU_{L1}| < 0.0005 at each run's own re-selected frozen threshold);
it only answers an attribution question: does the difference come from an error in the L2 pipeline (view / mapping / supports),
or from "the source-validation IoU at two adjacent thresholds is nearly tied and a tiny cross-machine drift flipped the argmax"?

Method: the L2 result JSON has carried `target_threshold_sensitivity` (0.40-0.60) since its first version, so the target IoU of the
L2 run **at the L1 frozen threshold** can be read directly and subtracted from the L1 target IoU -- this is the cross-machine drift at
equal thresholds. If the equal-threshold difference is << 0.0005 while the re-selected thresholds differ => attributed to a near-tie
flip, not to a pipeline error.

Disposition (decided before this script and independent of the criterion): the curve whose threshold flipped is re-run entirely on
dyt (the L1 machine); the original 30141 files are kept in results/l2_superseded_30141/ for the record.
"""
import argparse, json, sys, pathlib
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
TAGS = ('Aplain', 'Bplain', 'Cconfidence')


def L(p):
    return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--l1-stage', default='stage1')
    ap.add_argument('--l2-dir', default='l2', help='L2 results directory to diagnose (e.g. l2 or l2_superseded_30141)')
    ap.add_argument('--seeds', default=','.join(str(s) for s in range(100, 112)))
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    rows, mx_eq, mx_raw = [], (0, None), (0, None)
    for t in TAGS:
        for s in seeds:
            p1 = ROOT / 'results' / a.l1_stage / f'{t}_seed{s}_fmin0.01_K4_r0.json'
            p2 = ROOT / 'results' / a.l2_dir / f'{t}_seed{s}_gsd1.json'
            if not p2.exists():
                continue
            l1, l2 = L(p1), L(p2)
            t1, t2 = l1['source_val_threshold'], l2['source_val_threshold']
            d_raw = l2['target']['iou'] - l1['target']['iou']
            d_eq = l2['target_threshold_sensitivity'][str(t1)] - l1['target']['iou']
            rows.append((t, s, t1, t2, d_raw, d_eq))
            if abs(d_raw) > mx_raw[0]:
                mx_raw = (abs(d_raw), (t, s))
            if abs(d_eq) > mx_eq[0]:
                mx_eq = (abs(d_eq), (t, s))
            if t1 != t2:
                print(f'  thresholds differ: {t} seed{s}  L1 frozen {t1} (WHU val IoU {l1["source_val_iou_at_threshold"]:.7f})'
                      f'  L2 re-selected {t2} (WHU val IoU {l2["source_val_iou_at_threshold"]:.7f})')
                print(f'    difference at each run\'s own threshold d = {d_raw:+.8f};  difference at the equal threshold ({t1}) d = {d_eq:+.8f}')
                print(f'    control: L1 target IoU at {t2} vs L2 at {t2} = '
                      f'{l2["target"]["iou"] - l1["target_threshold_sensitivity"][str(t2)]:+.8f}')
    print(f'  n = {len(rows)} f=1 points')
    print(f'  at each run\'s own re-selected threshold: max|d| = {mx_raw[0]:.8f} @ {mx_raw[1]}')
    print(f'  at the equal (L1 frozen) threshold: max|d| = {mx_eq[0]:.8f} @ {mx_eq[1]}')
    print('  (E1 threshold 0.0005; an equal-threshold difference far below it => the difference comes from a near-tie flip, not from an L2 pipeline error)')


if __name__ == '__main__':
    main()
