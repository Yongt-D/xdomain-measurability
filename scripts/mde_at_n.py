# -*- coding: utf-8 -*-
"""
Instrument resolution: the minimum detectable effect (MDE) at a given n, and the paired correlation.
**This is not an effect estimate** -- the script uses only s_d (the dispersion of the paired difference) and prints no arm-contrast point estimate.

Written after the s_d of stage0b had been seen (recorded in the stage-0b calibration judgement, section 5).
It is legitimate because the degraded criterion of section 3.3 **requires** the resolution limit of the test bed to be reported;
this script only restates the same s_d already computed by stage_analyze.py and introduces no new effect claim.

Usage: python scripts/mde_at_n.py --stage stage0b
"""
import json, sys, argparse, pathlib
import numpy as np
from scipy import stats
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(); ap.add_argument('--stage', required=True, choices=['stage0', 'stage0b'])
ST = ap.parse_args().stage

per_seed = {}
for tag in ('Aplain', 'Cconfidence'):
    for seed in (0, 1, 2):
        v = [json.loads((ROOT / 'results' / ST / f'{tag}_seed{seed}_fmin0.01_K4_r{r}.json')
             .read_text(encoding='utf-8'))['target']['iou'] for r in range(5)]
        per_seed[(tag, seed)] = float(np.mean(v))
a = np.array([per_seed[('Aplain', s)] for s in (0, 1, 2)])
c = np.array([per_seed[('Cconfidence', s)] for s in (0, 1, 2)])
d = c - a
s_d = float(d.std(ddof=1))
s_a, s_c = float(a.std(ddof=1)), float(c.std(ddof=1))
r = float(np.corrcoef(a, c)[0, 1])

print(f'[{ST}] instrument resolution (dispersion only; no effect-size point estimate)')
print(f'  per-arm seed SD        s_A = {s_a:.6f}   s_C = {s_c:.6f}')
print(f'  paired correlation     r   = {r:+.4f}   (n=3, very unstable; qualitative only)')
print(f'  paired-difference SD   s_d = {s_d:.6f}')
print(f'  unpaired equivalent SD sqrt(s_A^2+s_C^2) = {np.hypot(s_a, s_c):.6f}'
      f'   -> variance reduction from pairing {np.hypot(s_a, s_c)/max(s_d,1e-12):.2f}x')

print('\n  minimum detectable effect MDE at 80% power for a given n (two-sided alpha=0.05):')
for n in (3, 5, 6, 8, 10, 12):
    lo, hi = 1e-5, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        ncp = mid / (s_d / np.sqrt(n)); crit = stats.t.ppf(0.975, n - 1)
        p = stats.nct.sf(crit, n - 1, ncp) + stats.nct.cdf(-crit, n - 1, ncp)
        if p >= 0.80: hi = mid
        else: lo = mid
    print(f'    n = {n:2d}  ->  MDE = {hi:.4f} IoU')
print('\n  note: the ceiling n=12 is fixed in section 3.3. The MDE is an instrument property and **must not** be used to raise the'
      ' target effect size from delta=0.01 to any larger value after the fact -- that is exactly the "change the setting until it passes" forbidden by Appendix D.4.')
