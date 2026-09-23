# -*- coding: utf-8 -*-
"""
Stage-1 analysis (degraded criterion of section 3.3 + Appendix D.4 + Appendix F.3). **Committed before any stage-1 number existed.**

**No significance claims.** Appendix D.4 was fixed before stage 0b ran: after two rounds of calibration, detecting
delta=0.01 would still need n=24 > the ceiling of 12, so the primary criterion was degraded to "direction consistency +
effect size with confidence interval". This script therefore **computes and prints no p-value**; the confidence
intervals it prints are **descriptive** and must not be used to claim "significance".

Order of steps (must not be changed):
  [gate 1] empirical check of the K-invariance of arms A/B (Appendix F.3): for the same seed the IoU at K in {1,4,8}
           must be **exactly equal**; otherwise the pipeline contradicts the code structure => **stop**.
  [gate 2] sanity (as in Appendix E.2): mean in-domain test IoU of arm A >= 0.846343; every best_epoch >= 2;
           verify_fork.py exit code 0.
  [primary] executed only if both gates pass.
"""
import json, sys, argparse, pathlib, subprocess, itertools
import numpy as np
from scipy import stats
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
CH4_A_MEAN, GATE = 0.856343, 0.01
SEEDS_DEFAULT = list(range(100, 112))
TAGS = ('Aplain', 'Bplain', 'Cconfidence')
KS = (1, 4, 8)

ap = argparse.ArgumentParser()
ap.add_argument('--stage', default='stage1')
ap.add_argument('--seeds', default=','.join(map(str, SEEDS_DEFAULT)))
a = ap.parse_args()
ST = a.stage
SEEDS = [int(s) for s in a.seeds.split(',')]


def tgt(tag, seed, k, r):
    p = ROOT / 'results' / ST / f'{tag}_seed{seed}_fmin0.01_K{k}_r{r}.json'
    if not p.exists():
        print(f'  missing {p.name}')
        sys.exit(1)
    return json.loads(p.read_text(encoding='utf-8'))['target']['iou']


def run_json(tag, seed):
    p = ROOT / 'outputs' / f'{ST}_{tag}_seed{seed}' / 'results.json'
    if not p.exists():
        print(f'  missing {p}')
        sys.exit(1)
    return json.loads(p.read_text(encoding='utf-8'))


print('=' * 72)
print(f'[gate 1] empirical K-invariance of arms A/B (Appendix F.3) -- n={len(SEEDS)} seeds')
print('=' * 72)
bad = []
for tag in ('Aplain', 'Bplain'):
    for seed in SEEDS:
        vals = [tgt(tag, seed, k, 0) for k in KS]
        if len(set(vals)) != 1:
            bad.append((tag, seed, vals))
if bad:
    print('  **not invariant**, contradicting the code structure in which supports never enter the forward pass when prototype_mode=="none":')
    for tag, seed, vals in bad[:10]:
        print(f'    {tag} seed{seed}: {vals}')
    print('  => **STOP**; find the pipeline bug first; do not analyse')
    sys.exit(2)
print(f'  A/B exactly equal across K in {KS} for every seed  -> pass')

print('\n' + '=' * 72)
print('[gate 2] sanity (as in Appendix E.2)')
print('=' * 72)
runs = {(t, s): run_json(t, s) for t in TAGS for s in SEEDS}
a_in = [runs[('Aplain', s)]['test_metrics']['iou'] for s in SEEDS]
lo = CH4_A_MEAN - GATE
g1 = float(np.mean(a_in)) >= lo
print(f'  E.2.1 arm A mean in-domain test IoU {np.mean(a_in):.6f} {">=" if g1 else "<"} {lo:.6f}  -> {"pass" if g1 else "**FAIL**"}')
be = [v.get('best_epoch', v.get('best_epoch_idx')) for v in runs.values()]
g2 = all(e is not None and e >= 2 for e in be)
print(f'  E.2.2 smallest best_epoch {min(e for e in be if e is not None)}  all >=2 -> {"pass" if g2 else "**FAIL**"}')
r = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'verify_fork.py')], capture_output=True)
g3 = r.returncode == 0
print(f'  E.2.3 verify_fork.py exit code {r.returncode}  -> {"pass" if g3 else "**FAIL**"}')
if not (g1 and g2 and g3):
    print('  => **gate failed, stop**')
    sys.exit(2)
print('  => all gates passed')

print('\n' + '=' * 72)
print('[primary] degraded criterion: direction consistency + effect size + descriptive CI')
print('**No significance claims (section 3.3 + Appendix D.4); this script computes no p-value.**')
print('=' * 72)


def seed_value(tag, seed, k):
    """A.4: arm C is averaged over R=5 draws; A/B are structurally identical across draws, R=1."""
    rs = range(5) if tag == 'Cconfidence' else range(1)
    return float(np.mean([tgt(tag, seed, k, r) for r in rs]))


summary = {}
for k in KS:
    print(f'\n--- K = {k} ---')
    vals = {tag: np.array([seed_value(tag, s, k) for s in SEEDS]) for tag in TAGS}
    for tag in TAGS:
        v = vals[tag]
        print(f'  {tag:12s} mean {v.mean():.6f}  seed SD {v.std(ddof=1):.6f}  '
              f'[{v.min():.4f}, {v.max():.4f}]')
    for lhs, rhs in (('Cconfidence', 'Aplain'), ('Cconfidence', 'Bplain'), ('Bplain', 'Aplain')):
        d = vals[lhs] - vals[rhs]
        n = len(d)
        m, sd = float(d.mean()), float(d.std(ddof=1))
        half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)      # descriptive, not a test
        pos = int((d > 0).sum())
        print(f'  {lhs[:1]} - {rhs[:1]} : effect {m:+.6f}   95% descriptive CI [{m-half:+.6f}, {m+half:+.6f}]   '
              f'paired-diff SD {sd:.6f}   direction consistency {pos}/{n} positive')
        summary[f'K{k}_{lhs[:1]}_minus_{rhs[:1]}'] = {
            'n': n, 'effect': m, 'sd_d': sd, 'ci95_descriptive': [m - half, m + half],
            'direction_positive': pos, 'per_seed': d.tolist(),
        }

out = ROOT / 'results' / f'{ST}_summary.json'
out.write_text(json.dumps({'seeds': SEEDS, 'summary': summary,
                           'note': 'degraded criterion: no significance claims, CIs are descriptive (section 3.3 + appendix D.4)'},
                          ensure_ascii=False, indent=2), encoding='utf-8')
print(f'\n  -> {out}')
print('\nReminder: point estimates from stage 0/0b must not enter the conclusions; the numbers of this section are the reportable results.')
