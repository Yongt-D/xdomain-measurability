# -*- coding: utf-8 -*-
"""
Stage-0 instrument diagnostic: decompose the cross-domain variance into "training-seed variance" and "support-draw variance".
This is **instrument characterisation**, not an effect estimate -- the script deliberately computes and prints no arm contrast.
"""
import json, sys, pathlib
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
S = ROOT / 'results' / 'stage0'

print('cross-domain IoU (Inria test, 2997 patches, source-validation threshold)')
print(f"{'arm':12s}{'seed':>5s}{'mean(R)':>10s}{'SD(R)':>10s}   per draw")
tbl = {}
for tag in ('Aplain', 'Cconfidence'):
    for seed in (0, 1, 2):
        v = [json.loads((S / f'{tag}_seed{seed}_fmin0.01_K4_r{r}.json').read_text(encoding='utf-8'))['target']['iou']
             for r in range(5)]
        tbl[(tag, seed)] = v
        print(f"{tag:12s}{seed:>5d}{np.mean(v):>10.4f}{np.std(v, ddof=1):>10.4f}   {[round(x,4) for x in v]}")

print('\nvariance decomposition:')
for tag in ('Aplain', 'Cconfidence'):
    seed_means = [np.mean(tbl[(tag, s)]) for s in (0, 1, 2)]
    within = np.mean([np.std(tbl[(tag, s)], ddof=1) for s in (0, 1, 2)])
    across = np.std(seed_means, ddof=1)
    print(f"  {tag:12s} SD across seeds = {across:.4f}   |   mean SD over support draws (R) = {within:.4f}"
          f"   |   ratio {across/max(within,1e-9):.1f}x")

print('\nthresholds (selected on the source validation set; target labels never used):')
for tag in ('Aplain', 'Cconfidence'):
    th = [json.loads((S / f'{tag}_seed{s}_fmin0.01_K4_r0.json').read_text(encoding='utf-8'))['source_val_threshold']
          for s in (0, 1, 2)]
    print(f"  {tag:12s} {th}")

print('\nconclusion (about the instrument only):')
a_across = np.std([np.mean(tbl[('Aplain', s)]) for s in (0,1,2)], ddof=1)
c_across = np.std([np.mean(tbl[('Cconfidence', s)]) for s in (0,1,2)], ddof=1)
a_within = np.mean([np.std(tbl[('Aplain', s)], ddof=1) for s in (0,1,2)])
print(f"  the training-seed variance (A {a_across:.4f} / C {c_across:.4f}) is an order of magnitude larger than the support-draw variance ({a_within:.4f}),")
print(f"  => increasing R does not help; the variance bottleneck is **the instability of source-domain training itself** (WHU-100 has only 100 source samples).")
