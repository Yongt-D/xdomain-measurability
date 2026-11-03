# -*- coding: utf-8 -*-
"""
Threshold sensitivity of stage 1 (pre-registration **Appendix Q.3, items 2/3**).
**Committed before stage 1 produced any evaluation number.**

Q.3 requires that, next to the primary criterion (`C-A` at the source-frozen threshold), the `C-A` curve over
**all 21 thresholds of 0.40-0.60** is reported, together with **whether its sign flips inside that range**;
if it flips, the paper must state that "the arm-level conclusion of L1 depends on the operating point and is
unstable inside the threshold range of this setting".

Data source: `scripts/eval_crossdomain.py` has written `target_threshold_sensitivity` into every result JSON
**since its first version**, so this analysis **needs no new experiment**.

Aggregation as in the primary analysis (section 3.2 / A.4):
  - independent unit = **training seed**;
  - arm C is first averaged over its R=5 support draws; arms A/B are structurally draw-invariant (F.3), R=1.

This script **does not change the primary criterion** and **must not** be used to pick a more favourable operating point (Appendix Q.4).
"""
import argparse, json, sys, pathlib
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
THRESHOLDS = [round(0.40 + 0.01 * i, 2) for i in range(21)]


def load(stage, tag, seed, k, r):
    p = ROOT / 'results' / stage / f'{tag}_seed{seed}_fmin0.01_K{k}_r{r}.json'
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding='utf-8'))


def curve(stage, tag, seed, k):
    """IoU of this (arm, seed, K) at the 21 thresholds; arm C is averaged over R first."""
    rs = range(5) if tag == 'Cconfidence' else range(1)
    acc = []
    for r in rs:
        d = load(stage, tag, seed, k, r)
        if d is None:
            return None, None
        ts = d['target_threshold_sensitivity']
        acc.append([ts[str(t)] for t in THRESHOLDS])
    frozen = load(stage, tag, seed, k, 0)['source_val_threshold']
    return np.mean(acc, axis=0), frozen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='stage1')
    ap.add_argument('--seeds', default=','.join(str(s) for s in range(100, 112)))
    ap.add_argument('--ks', default='1,4,8')
    ap.add_argument('--lhs', default='Cconfidence')
    ap.add_argument('--rhs', default='Aplain')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    out = {}

    print('Appendix Q.3: threshold sensitivity of `C - A` (the primary criterion remains the source-frozen threshold; this table is reported alongside)')
    for k in [int(x) for x in a.ks.split(',')]:
        rows, frozen_deltas = [], []
        for s in seeds:
            cl, cf = curve(a.stage, a.lhs, s, k)
            al, af = curve(a.stage, a.rhs, s, k)
            if cl is None or al is None:
                continue
            rows.append(cl - al)
            # primary-criterion point: each arm at **its own** source-frozen threshold (as in eval_crossdomain)
            ci = THRESHOLDS.index(round(cf, 2)) if round(cf, 2) in THRESHOLDS else None
            ai = THRESHOLDS.index(round(af, 2)) if round(af, 2) in THRESHOLDS else None
            if ci is not None and ai is not None:
                frozen_deltas.append(cl[ci] - al[ai])
        if not rows:
            print(f'\n--- K = {k} ---  no data'); continue
        M = np.stack(rows)                      # (n_seed, 21)
        mean = M.mean(0)
        pos = (M > 0).sum(0)
        sign_flips = bool((mean > 0).any() and (mean < 0).any())
        print(f'\n--- K = {k} ---  n = {M.shape[0]} seeds')
        print(f"  {'thr':>6s}{'C-A mean':>12s}{'SD':>10s}{'positive seeds':>16s}")
        for i, t in enumerate(THRESHOLDS):
            print(f'  {t:>6.2f}{mean[i]:>+12.6f}{M[:, i].std(ddof=1):>10.6f}{pos[i]:>12d}/{M.shape[0]}')
        print(f'  sign of the mean `C-A` flips inside the range: **{"yes" if sign_flips else "no"}**'
              f'   (min {mean.min():+.6f} @ {THRESHOLDS[int(mean.argmin())]},'
              f' max {mean.max():+.6f} @ {THRESHOLDS[int(mean.argmax())]})')
        if frozen_deltas:
            fd = np.asarray(frozen_deltas)
            print(f'  reference: `C-A` at the frozen thresholds = {fd.mean():+.6f} +/- {fd.std(ddof=1):.6f}'
                  f'  (positive {int((fd > 0).sum())}/{fd.size})')
        if sign_flips:
            print('  => per Appendix Q.3 item 3 the paper **must** state that'
                  ' "the arm-level conclusion of L1 depends on the operating point and is unstable inside the threshold range of this setting".')
        out[f'K{k}'] = {'n': int(M.shape[0]), 'thresholds': THRESHOLDS,
                        'mean': mean.tolist(), 'sd': M.std(0, ddof=1).tolist(),
                        'n_positive': pos.tolist(), 'sign_flips': sign_flips,
                        'frozen_delta_mean': (float(np.mean(frozen_deltas)) if frozen_deltas else None)}
    if a.out:
        p = pathlib.Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'\n-> {p}')
    print('\nReminder (Appendix Q.4): the primary result must not be reported at the target-optimal threshold, '
          'the operating point of the primary criterion must not be changed because another threshold is more favourable, and no favourable subset of thresholds may be reported alone.')


if __name__ == '__main__':
    main()
