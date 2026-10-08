# -*- coding: utf-8 -*-
"""
Stage-1 **in-domain** (WHU test) reference summary -- **not the primary criterion, background only** (section 1: the primary criterion is cross-domain, never in-domain).
**Committed before its numbers were written into any document.**

Purpose: answer whether the cross-domain deficit of arm B relative to A also exists inside the source domain, to help interpret the
cross-domain results; it must not be used for any claim that "the method works". No p-values; CIs descriptive.
Source: results/stage1_train/{tag}_seed{s}.results.json (copies of outputs/stage1_*/results.json).
"""
import argparse, json, sys, pathlib
import numpy as np
from scipy import stats
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
TAGS = ('Aplain', 'Bplain', 'Cconfidence')


def ci(d):
    d = np.asarray(d, float); n = d.size
    m, sd = float(d.mean()), float(d.std(ddof=1))
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return m, sd, m - half, m + half, int((d > 0).sum()), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='stage1_train')
    ap.add_argument('--seeds', default=','.join(str(s) for s in range(100, 112)))
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    D = ROOT / 'results' / a.dir
    R = {}
    for t in TAGS:
        for s in seeds:
            p = D / f'{t}_seed{s}.results.json'
            if not p.exists():
                print(f'  missing {p}'); sys.exit(1)
            R[(t, s)] = json.loads(p.read_text(encoding='utf-8'))
    out = {'seeds': seeds, 'note': 'in-domain (WHU test) background numbers, not the primary criterion; no significance claims', 'arms': {}, 'diff': {}}
    print('in-domain (WHU test) IoU -- **not the primary criterion**, background for the cross-domain results only')
    for t in TAGS:
        v = np.array([R[(t, s)]['test_metrics']['iou'] for s in seeds])
        be = [R[(t, s)].get('best_epoch', R[(t, s)].get('best_epoch_idx')) for s in seeds]
        print(f'  {t:12s} mean {v.mean():.6f}  SD {v.std(ddof=1):.6f}  [{v.min():.4f}, {v.max():.4f}]  median best_epoch {np.median(be):.0f}')
        out['arms'][t] = {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)), 'per_seed': v.tolist(), 'best_epoch': be}
    for lhs, rhs in (('Cconfidence', 'Aplain'), ('Cconfidence', 'Bplain'), ('Bplain', 'Aplain')):
        d = [R[(lhs, s)]['test_metrics']['iou'] - R[(rhs, s)]['test_metrics']['iou'] for s in seeds]
        m, sd, lo, hi, pos, n = ci(d)
        print(f'  {lhs[:1]} - {rhs[:1]} : effect {m:+.6f}   descriptive CI [{lo:+.6f}, {hi:+.6f}]   positive {pos}/{n}')
        out['diff'][f'{lhs[:1]}_minus_{rhs[:1]}'] = {'effect': m, 'sd_d': sd, 'ci95_descriptive': [lo, hi],
                                                   'direction_positive': pos, 'n': n, 'per_seed': d}
    outp = pathlib.Path(a.out) if a.out else ROOT / 'results' / f'{a.dir}_indomain_summary.json'
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'  -> {outp}')


if __name__ == '__main__':
    main()
