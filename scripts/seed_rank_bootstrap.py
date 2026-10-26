# -*- coding: utf-8 -*-
"""Seed-level bootstrap intervals for the rank correlation of per-seed IoU between target domains (descriptive).

Reads the committed summaries (stage1_summary.json, mass_summary.json, w_summary.json), computes the Spearman
correlation of per-seed IoU between Inria and Massachusetts for each arm / architecture at K=4 and the
source-frozen threshold, and a percentile bootstrap interval over seeds (resampling the 12 (Inria, Mass) pairs
with replacement, 10 000 replicates, fixed RNG seed). Writes results/seed_rank_bootstrap_summary.json.
"""
import json, io, pathlib
import numpy as np
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT / 'results'
B = 10_000


def find(o, path):
    for k in path.split('.'):
        o = o[k]
    return np.asarray(o, dtype=float)


def search(o, want, pre=''):
    """Return the first key path whose name contains all tokens in `want` and holds a 12-vector."""
    if isinstance(o, dict):
        for k, v in o.items():
            p = f'{pre}{k}'
            if isinstance(v, list) and len(v) == 12 and all(t in p for t in want):
                return p
            r = search(v, want, p + '.')
            if r:
                return r
    return None


def boot(x, y, rng):
    r0 = spearmanr(x, y).correlation
    n = len(x); rs = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if len(set(i)) < 3:
            continue
        rs.append(spearmanr(x[i], y[i]).correlation)
    rs = np.asarray(rs)
    return float(r0), [float(np.nanpercentile(rs, 2.5)), float(np.nanpercentile(rs, 97.5))]


def main():
    l2 = json.load(io.open(R / 'l2_summary.json', encoding='utf-8'))      # Inria f=1 per-seed IoU (K=4, draw r0)
    ms = json.load(io.open(R / 'mass_summary.json', encoding='utf-8'))    # Massachusetts per-seed IoU (K=4, C over 5 draws)
    w = json.load(io.open(R / 'w_summary.json', encoding='utf-8'))
    rng = np.random.default_rng(0)
    out = {'note': 'Spearman rank correlation of per-seed IoU (K=4, source-frozen threshold) between Inria (f=1 view, draw r0) '
                   'and Massachusetts (C averaged over five draws); percentile bootstrap over seeds, 10 000 replicates; descriptive',
           'pairs': {}}
    for arm in 'ABC':
        x = find(l2, f'main_curve.{arm}_gsd1.per_seed'); y = find(ms, f'main.K4_arms.{arm}.per_seed')
        r, ci = boot(x, y, rng)
        out['pairs'][f'GAPL_{arm}'] = {'rho': r, 'ci95': ci}
        print(f'GAPL {arm}: rho {r:+.2f}  CI [{ci[0]:+.2f}, {ci[1]:+.2f}]')
    for arch in ('deeplabv3_r50', 'segformer_b1'):   # the arrays behind W-H3 rho_inria_mass (frozen threshold)
        x = find(w, f'per_arch.{arch}.inria_frozen'); y = find(w, f'per_arch.{arch}.mass_frozen')
        r, ci = boot(x, y, rng)
        out['pairs'][arch] = {'rho': r, 'ci95': ci}
        print(f'{arch}: rho {r:+.2f}  CI [{ci[0]:+.2f}, {ci[1]:+.2f}]')
    (R / 'seed_rank_bootstrap_summary.json').write_text(json.dumps(out, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
