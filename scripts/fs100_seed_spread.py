# -*- coding: utf-8 -*-
"""
Fixed-subset seed repetition vs stage 1: does the cross-domain amplification of the seed spread (R1) survive when the
WHU subset, the 800/200 split and the loader order are held fixed and only the initialisation varies?
Rebuttal preparation, NOT part of the pre-specified protocol. Descriptive only; no significance claims.

Inputs (same layout and selection rule as scripts/rq_analyses.py and Fig. 5 of make_paper_figures2.py):
  in-domain : results/{stage1_train | fs100_train}/{tag}_seed{s}.results.json  -> test_metrics.iou (WHU test)
  Inria     : results/{stage1 | fs100_inria}/{tag}_seed{s}_fmin0.01_K4_r{r}.json -> target.iou, arm C averaged over r0-r4
  Mass.     : results/{mass | fs100_mass}/{tag}_seed{s}_fmin0.01_K4_r{r}.json   -> overall.iou, arm C averaged over r0-r4
Output: results/fs100_seed_spread_summary.json
"""
import argparse, json, pathlib, sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
TAGS = ('Aplain', 'Bplain', 'Cconfidence')
SHORT = {'Aplain': 'A', 'Bplain': 'B', 'Cconfidence': 'C'}
SEEDS = list(range(100, 112))
SETS = {'stage1': {'train': 'stage1_train', 'inria': 'stage1', 'mass': 'mass'},
        'fixed': {'train': 'fs100_train', 'inria': 'fs100_inria', 'mass': 'fs100_mass'}}


def L(p):
    p = pathlib.Path(p)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding='utf-8'))


def sd(x):
    return float(np.std(np.asarray(x, float), ddof=1))


def per_seed(dirs, dom, tag):
    """Per-seed IoU for one domain; None if any file is missing."""
    out = []
    for s in SEEDS:
        if dom == 'train':
            d = L(ROOT / 'results' / dirs['train'] / f'{tag}_seed{s}.results.json')
            if d is None:
                return None
            out.append(d['test_metrics']['iou'])
        else:
            rs = range(5) if tag == 'Cconfidence' else range(1)
            vals = []
            for r in rs:
                d = L(ROOT / 'results' / dirs[dom] / f'{tag}_seed{s}_fmin0.01_K4_r{r}.json')
                if d is None:
                    return None
                vals.append(d['target']['iou'] if dom == 'inria' else d['overall']['iou'])
            out.append(float(np.mean(vals)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'results' / 'fs100_seed_spread_summary.json'))
    a = ap.parse_args()
    V = {k: {dom: {t: per_seed(dirs, dom, t) for t in TAGS} for dom in ('train', 'inria', 'mass')} for k, dirs in SETS.items()}
    out = {'seeds': SEEDS, 'note': 'fixed subset (sample-seed 100) vs stage 1 (sample-seed = seed); K=4, arm C averaged over draws r0-r4; '
                                   'SD = seed-to-seed standard deviation (ddof=1); descriptive only', 'sd': {}, 'amplification': {}, 'sd_ratio_fixed_over_stage1': {},
           'mean': {}, 'contrast_C_minus_A': {}}
    print('seed-to-seed SD of IoU (n = 12 seeds), K = 4, C averaged over r0-r4')
    print(f"{'':14s}{'in-domain':>22s}{'Inria':>22s}{'Massachusetts':>22s}")
    print(f"{'arm':14s}" + ''.join(f"{'stage1':>11s}{'fixed':>11s}" for _ in range(3)))
    for t in TAGS:
        row = f'{SHORT[t]:14s}'
        for dom in ('train', 'inria', 'mass'):
            for k in ('stage1', 'fixed'):
                v = V[k][dom][t]
                row += f'{sd(v):11.4f}' if v else f"{'--':>11s}"
                if v:
                    out['sd'][f'{k}_{dom}_{SHORT[t]}'] = sd(v)
                    out['mean'][f'{k}_{dom}_{SHORT[t]}'] = float(np.mean(v))
        print(row)
    print('\namplification = SD(cross-domain) / SD(in-domain), per arm')
    print(f"{'arm':14s}{'Inria stage1':>14s}{'Inria fixed':>14s}{'Mass stage1':>14s}{'Mass fixed':>14s}")
    for t in TAGS:
        row = f'{SHORT[t]:14s}'
        for dom in ('inria', 'mass'):
            for k in ('stage1', 'fixed'):
                vi, vc = V[k]['train'][t], V[k][dom][t]
                if vi and vc:
                    amp = sd(vc) / sd(vi)
                    out['amplification'][f'{k}_{dom}_{SHORT[t]}'] = amp
                    row += f'{amp:14.1f}'
                else:
                    row += f"{'--':>14s}"
        print(row)
    print('\nSD ratio fixed / stage1 (how much of the stage-1 spread the subset re-sampling accounts for; <1 = fixed subset is tighter)')
    for dom in ('train', 'inria', 'mass'):
        parts = []
        for t in TAGS:
            v1, v2 = V['stage1'][dom][t], V['fixed'][dom][t]
            if v1 and v2:
                r = sd(v2) / sd(v1)
                out['sd_ratio_fixed_over_stage1'][f'{dom}_{SHORT[t]}'] = r
                parts.append(f'{SHORT[t]} {r:.2f}')
        print(f'  {dom:10s} ' + '   '.join(parts))
    print('\nC - A paired contrast (mean, positive seeds) on the fixed-subset runs vs stage 1')
    for dom in ('inria', 'mass'):
        for k in ('stage1', 'fixed'):
            va, vc = V[k][dom]['Aplain'], V[k][dom]['Cconfidence']
            if va and vc:
                d = np.asarray(vc) - np.asarray(va)
                out['contrast_C_minus_A'][f'{k}_{dom}'] = {'effect': float(d.mean()), 'sd_d': sd(d), 'positive': int((d > 0).sum()), 'n': int(d.size), 'per_seed': d.tolist()}
                print(f'  {dom:6s} {k:7s} C-A {d.mean():+.4f}  SD_d {sd(d):.4f}  positive {(d > 0).sum()}/{d.size}')
    out['per_seed'] = {f'{k}_{dom}_{SHORT[t]}': V[k][dom][t] for k in V for dom in V[k] for t in TAGS if V[k][dom][t]}
    pathlib.Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n-> {a.out}')


if __name__ == '__main__':
    main()
