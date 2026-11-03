# -*- coding: utf-8 -*-
"""
Calibration-stage analysis (parameterised; pre-registration section 3.3 + Appendix E). **Committed before any 0b number existed.**

Usage: python scripts/stage_analyze.py --stage stage0b

Two steps, in this order:
  [gate] stage0  -> Appendix C.4 (two-sided comparison with the chapter-4 value 0.856343)
         stage0b -> Appendix E.2 (three sanity gates; C.4 does not apply by construction)
  [calibration] only if all gates pass: estimate s_d of the cross-domain paired difference and derive the stage-1 n at 80% power.

Effect sizes and point estimates of stage 0/0b **must never be written into the paper or reported as results** (section 3.3).
This script therefore prints only s_d and n, **not the point estimate of the cross-domain effect**.
"""
import json, sys, argparse, pathlib, subprocess
import numpy as np
from scipy import stats
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
CH4_A_MEAN = 0.856343       # chapter-4 retrospective section 3.5, arm A, WHU-100/160 ep
GATE = 0.01

ap = argparse.ArgumentParser()
ap.add_argument('--stage', required=True, choices=['stage0', 'stage0b'])
args = ap.parse_args()
ST = args.stage

def load_run(tag, seed):
    p = ROOT / 'outputs' / f'{ST}_{tag}_seed{seed}' / 'results.json'
    if not p.exists():
        print(f'  missing {p}'); sys.exit(1)
    return json.loads(p.read_text(encoding='utf-8'))

print('=' * 70)
print(f'[gate] {ST}  --  this is not a result of the study')
print('=' * 70)
runs = {(t, s): load_run(t, s) for t in ('Aplain', 'Cconfidence') for s in (0, 1, 2)}
a_iou = [runs[('Aplain', s)]['test_metrics']['iou'] for s in (0, 1, 2)]
c_iou = [runs[('Cconfidence', s)]['test_metrics']['iou'] for s in (0, 1, 2)]
print(f"  arm A   in-domain test IoU  {[round(v,6) for v in a_iou]}  mean {np.mean(a_iou):.6f}")
print(f"  C+conf  in-domain test IoU  {[round(v,6) for v in c_iou]}  mean {np.mean(c_iou):.6f}")

ok = True
if ST == 'stage0':
    d = abs(float(np.mean(a_iou)) - CH4_A_MEAN)
    ok = d <= GATE
    print(f'\n  C.4: |A mean - {CH4_A_MEAN}| = {d:.6f} {"<=" if ok else ">"} {GATE}  -> {"pass" if ok else "**FAIL**"}')
else:
    lo = CH4_A_MEAN - GATE
    g1 = float(np.mean(a_iou)) >= lo
    print(f'\n  E.2.1 one-sided: A mean {np.mean(a_iou):.6f} {">=" if g1 else "<"} {lo:.6f}  -> {"pass" if g1 else "**FAIL**"}')
    be = {k: v.get('best_epoch', v.get('best_epoch_idx')) for k, v in runs.items()}
    g2 = all((e is not None and e >= 2) for e in be.values())
    print(f'  E.2.2 best_epoch  {{{", ".join(f"{k[0][:1]}{k[1]}:{v}" for k, v in be.items())}}}  '
          f'all >=2 -> {"pass" if g2 else "**FAIL**"}')
    r = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'verify_fork.py')],
                       capture_output=True, text=True)
    g3 = r.returncode == 0
    print(f'  E.2.3 verify_fork.py exit code {r.returncode}  -> {"pass" if g3 else "**FAIL**"}')
    ok = g1 and g2 and g3
if not ok:
    print('\n  => **gate failed; stop; calibration not allowed**'); sys.exit(2)
print('  => all gates passed; calibration allowed')

print('\n' + '=' * 70)
print('[calibration] s_d of the cross-domain paired difference (section 3.3) -- s_d and n only, no effect-size point estimate')
print('=' * 70)
rows = {}
for tag in ('Aplain', 'Cconfidence'):
    for seed in (0, 1, 2):
        vals = []
        for r_ in range(5):
            p = ROOT / 'results' / ST / f'{tag}_seed{seed}_fmin0.01_K4_r{r_}.json'
            if not p.exists():
                print(f'  missing {p.name}'); sys.exit(1)
            vals.append(json.loads(p.read_text(encoding='utf-8'))['target']['iou'])
        rows[(tag, seed)] = float(np.mean(vals))     # A.4: average over R first, then use as the seed's value
diffs = np.array([rows[('Cconfidence', s)] - rows[('Aplain', s)] for s in (0, 1, 2)])
s_d = float(diffs.std(ddof=1))
n0 = len(diffs)
print(f'  n0 = {n0} (independent unit = training seed; R=5 averaged first per A.4)')
print(f'  sample SD of the paired difference  s_d = {s_d:.6f} IoU')
print(f'  direction consistency: {int((diffs > 0).sum())}/{n0} positive   (**the point estimate is not reported here, per section 3.3**)')

def need_n(delta, s_d, target=0.80, alpha=0.05):
    for n in range(3, 401):
        ncp = delta / (s_d / np.sqrt(n)); crit = stats.t.ppf(1 - alpha / 2, n - 1)
        if stats.nct.sf(crit, n - 1, ncp) + stats.nct.cdf(-crit, n - 1, ncp) >= target:
            return n
    return None

print('\n  training seeds needed to detect each effect size at 80% power:')
for d_ in (0.005, 0.01, 0.02, 0.03):
    n = need_n(d_, s_d)
    print(f'    delta = {d_:.3f} IoU  ->  n = {n if n else ">400"}')
n_at_001 = need_n(0.01, s_d)
print(f'\n  section 3.3 criterion: detecting delta=0.01 needs n = {n_at_001}')
if n_at_001 is None or n_at_001 > 12:
    print('  => **exceeds the ceiling of 12**. Per Appendix D.4: **no third budget may be tried**.')
    print('     The primary criterion is degraded to "direction consistency + effect size with confidence interval"; the paper states explicitly that no significance claim is made,')
    print('     and reports the s_d of both calibrations and the resolution limit of this test bed.')
else:
    print(f'  => stage-1 n = max(5, {n_at_001}) = {max(5, n_at_001)} (ceiling 12)')
