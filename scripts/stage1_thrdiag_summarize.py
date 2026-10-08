# -*- coding: utf-8 -*-
"""
Summary for Appendix Q.3 item 4: the widened threshold diagnostic (0.05-0.95) on the stage-1 checkpoints.
**Committed before the diagnostic produced any number.**

Nature: **upper bound / diagnostic** (P.4 / Q.3.4). The oracle uses target labels, so
  - it must not replace any number of the primary protocol and must not enter the results table;
  - it must not be used to change the operating point of the primary criterion (Q.4);
  - its only use is to state the size of the calibration mismatch at the L1 operating point and the sign of `C-A`
    with each arm at its own widened oracle, reported next to the primary criterion (frozen threshold) as a fragility note.
This script computes no p-value; CIs are descriptive.
"""
import argparse, json, sys, pathlib
import numpy as np
from scipy import stats
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent


def ci(d):
    d = np.asarray(d, float); n = d.size
    m, sd = float(d.mean()), float(d.std(ddof=1))
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return m, sd, m - half, m + half, int((d > 0).sum()), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='thrdiag_stage1')
    ap.add_argument('--seeds', default=','.join(str(s) for s in range(100, 112)))
    ap.add_argument('--gsd', type=int, default=1, help='degradation factor f (M5 of Appendix R.6 uses 2/4/8; default 1 = the L1 operating point of Q.3.4)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    D = ROOT / 'results' / a.dir
    g = a.gsd
    tags = [t for t in ('Aplain', 'Bplain', 'Cconfidence') if (D / f'{t}_seed{seeds[0]}_gsd{g}.json').exists()]
    R = {}
    for t in tags:
        for s in seeds:
            p = D / f'{t}_seed{s}_gsd{g}.json'
            if not p.exists():
                print(f'  missing {p}'); sys.exit(1)
            R[(t, s)] = json.loads(p.read_text(encoding='utf-8'))

    out = {'seeds': seeds, 'per_arm': {}, 'C_minus_A': {},
           'note': 'upper bound / diagnostic (P.4 / Q.3.4); the oracle uses target labels; not in the results table; no significance claims'}
    print('=' * 76)
    print(f'Appendix Q.3.4 / P.4 / R.6: widened threshold diagnostic (0.05-0.95) on the stage-1 checkpoints, GSD f={g} (0.3 x f m) -- **upper bound / diagnostic, not in the results table**')
    print('=' * 76)
    print(f"  {'arm':>12s}{'IoU_frozen':>12s}{'IoU_oracle':>12s}{'frozen/oracle':>15s}{'oracle thr (med)':>18s}{'boundary':>10s}")
    for t in tags:
        fr = np.array([R[(t, s)]['iou_frozen'] for s in seeds])
        orc = np.array([R[(t, s)]['iou_oracle_wide'] for s in seeds])
        ratio = fr / np.maximum(orc, 1e-12)
        thr = np.array([R[(t, s)]['oracle_threshold'] for s in seeds])
        nb = sum(int(R[(t, s)]['oracle_at_boundary']) for s in seeds)
        print(f'  {t:>12s}{fr.mean():>12.6f}{orc.mean():>12.6f}{ratio.mean():>15.3f}{np.median(thr):>18.2f}{nb:>7d}/{len(seeds)}')
        out['per_arm'][t] = {'iou_frozen_mean': float(fr.mean()), 'iou_frozen_sd': float(fr.std(ddof=1)),
                             'iou_oracle_wide_mean': float(orc.mean()), 'iou_oracle_wide_sd': float(orc.std(ddof=1)),
                             'frozen_over_oracle_mean': float(ratio.mean()),
                             'oracle_threshold_per_seed': thr.tolist(), 'n_oracle_at_boundary': nb}
    if 'Aplain' in tags and 'Cconfidence' in tags:
        print('\n  `C - A` (same reading as the Q.1 table; the primary criterion remains the frozen-threshold column):')
        for name, key in (('frozen threshold', 'iou_frozen'), ('each arm at its own widened oracle', 'iou_oracle_wide')):
            d = [R[('Cconfidence', s)][key] - R[('Aplain', s)][key] for s in seeds]
            m, sd, lo, hi, pos, n = ci(d)
            print(f'    {name:>34s}: effect {m:+.6f}   descriptive CI [{lo:+.6f}, {hi:+.6f}]   positive {pos}/{n}')
            out['C_minus_A'][key] = {'effect': m, 'sd_d': sd, 'ci95_descriptive': [lo, hi],
                                     'direction_positive': pos, 'n': n, 'per_seed': d}
        s1 = np.sign(out['C_minus_A']['iou_frozen']['effect'])
        s2 = np.sign(out['C_minus_A']['iou_oracle_wide']['effect'])
        print(f'    sign between the frozen threshold and the widened oracle: {"**flips**" if s1 != s2 else "same"}'
              f' (the Q.3 item-3 judgement is based on the 21-point curve inside 0.40-0.60; this is supplementary)')
    outp = pathlib.Path(a.out) if a.out else ROOT / 'results' / (f'{a.dir}_summary.json' if g == 1 else f'{a.dir}_gsd{g}_summary.json')
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n  -> {outp}')
    print('\nReminder (Q.4): the primary result must not be reported at the target-optimal threshold; the operating point of the primary criterion must not be changed because another threshold is more favourable.')


if __name__ == '__main__':
    main()
