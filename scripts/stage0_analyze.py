# -*- coding: utf-8 -*-
"""
Stage-0 analysis (pre-registration section 3.3 + Appendix C.4). **Committed before any number existed.**

Two steps, in this order:
  [gate] pipeline-fidelity check (C.4): if the 3-seed mean **in-domain** test IoU of arm A differs from the chapter-4
         archive value 0.856343 by more than 0.01, the pipeline is judged unfaithful and the analysis **stops**.
  [calibration] only if the gate passes: estimate s_d of the cross-domain paired difference and derive the stage-1 n at 80% power.

Effect sizes and point estimates of stage 0 **must never be written into the paper or reported as results** (section 3.3).
This script therefore prints only s_d and n, **not the point estimate of the cross-domain effect**.
"""
import json, sys, pathlib
import numpy as np
from scipy import stats
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
CH4_A_MEAN, CH4_A_STD = 0.856343, 0.003593      # chapter-4 retrospective section 3.5, arm A
CH4_V5_MEAN, CH4_V5_STD = 0.862081, 0.002342    # chapter-4 retrospective section 3.5, V5 confidence
GATE = 0.01                                      # C.4 criterion

def load_indomain(tag, seed):
    p = ROOT / 'outputs' / f'stage0_{tag}_seed{seed}' / 'results.json'
    return json.loads(p.read_text(encoding='utf-8'))['test_metrics']['iou']

print('=' * 68)
print('[gate] pipeline-fidelity check (pre-registration Appendix C.4) -- this is not a result of the study')
print('=' * 68)
a_iou = [load_indomain('Aplain', s) for s in (0, 1, 2)]
c_iou = [load_indomain('Cconfidence', s) for s in (0, 1, 2)]
for name, vals, ref_m, ref_s in (('arm A', a_iou, CH4_A_MEAN, CH4_A_STD),
                                 ('C+conf', c_iou, CH4_V5_MEAN, CH4_V5_STD)):
    m = float(np.mean(vals))
    print(f'  {name:7s} in-domain test IoU  per seed {[round(v,6) for v in vals]}')
    print(f'          mean {m:.6f}   chapter-4 archive {ref_m:.6f} +/- {ref_s:.6f}   delta = {m-ref_m:+.6f}')
gate_delta = abs(float(np.mean(a_iou)) - CH4_A_MEAN)
ok = gate_delta <= GATE
print(f'\n  gate criterion: |arm A mean - {CH4_A_MEAN}| = {gate_delta:.6f}  {"<=" if ok else ">"} {GATE}')
print(f'  => {"pipeline faithful; continue" if ok else "**pipeline unfaithful; stop per C.4**"}')
if not ok:
    sys.exit(2)

print('\n' + '=' * 68)
print('[calibration] s_d of the cross-domain paired difference (section 3.3) -- s_d and n only, no effect-size point estimate')
print('=' * 68)
rows = {}
for tag in ('Aplain', 'Cconfidence'):
    for seed in (0, 1, 2):
        vals = []
        for r in range(5):
            p = ROOT / 'results' / 'stage0' / f'{tag}_seed{seed}_fmin0.01_K4_r{r}.json'
            if not p.exists():
                print(f'  missing {p.name}'); sys.exit(1)
            vals.append(json.loads(p.read_text(encoding='utf-8'))['target']['iou'])
        rows[(tag, seed)] = float(np.mean(vals))   # A.4: average over R first, then use as the seed's value
diffs = np.array([rows[('Cconfidence', s)] - rows[('Aplain', s)] for s in (0, 1, 2)])
s_d = float(diffs.std(ddof=1))
n0 = len(diffs)
print(f'  n0 = {n0} (independent unit = training seed; R=5 averaged first per A.4)')
print(f'  sample SD of the paired difference  s_d = {s_d:.6f} IoU')
print(f'  direction consistency: {int((diffs > 0).sum())}/{n0} positive   '
      f'(**the point estimate is not reported here, per section 3.3**)')

def need_n(delta, s_d, target=0.80, alpha=0.05):
    for n in range(3, 401):
        ncp = delta / (s_d / np.sqrt(n)); crit = stats.t.ppf(1 - alpha / 2, n - 1)
        if stats.nct.sf(crit, n - 1, ncp) + stats.nct.cdf(-crit, n - 1, ncp) >= target:
            return n
    return None

print('\n  training seeds needed to detect each effect size at 80% power:')
for d in (0.005, 0.01, 0.02, 0.03):
    n = need_n(d, s_d)
    print(f'    delta = {d:.3f} IoU  ->  n = {n if n else ">400"}')
n_at_001 = need_n(0.01, s_d)
print(f'\n  section 3.3 criterion: detecting delta=0.01 needs n = {n_at_001}')
if n_at_001 is None or n_at_001 > 12:
    print('  => **exceeds the ceiling of 12**; per section 3.3 this test bed cannot support significance claims,')
    print('     the primary criterion is degraded to "direction consistency + effect size with confidence interval", and the paper states that no significance claim is made.')
else:
    print(f'  => stage-1 n = max(5, {n_at_001}) = {max(5, n_at_001)} (ceiling 12)')
