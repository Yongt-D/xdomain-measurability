# -*- coding: utf-8 -*-
"""
Appendix U.9 analysis: the widened threshold diagnostic (0.05-0.95) on Massachusetts -- is the gain an operating-point / mask-morphology artefact?
**Committed before any of its numbers existed.** Nature: upper bound / diagnostic (Q.4); the oracle uses target labels and never enters the results table; no significance claims.
Criteria (U.9):
  U9-H1  delta_oracle = mean(IoU_C^oracle - IoU_A^oracle) >= MDE_mass and direction >= 10/12 => the gain survives at each arm's own optimal operating point (not only an operating-point artefact)
  U9-H2  A's oracle threshold < frozen threshold in >= 9/12 seeds (A under-predicts at the frozen operating point; lowering the threshold = mask dilation)
  U9-H3  IoU_A^oracle >= IoU_C^frozen in >= 9/12 seeds => A catches up with C's frozen result by a threshold change alone (strong artefact evidence)
  U9-E1  calibration share 1 - frozen/oracle per arm; side by side with Inria f=1 (A 0.037, C 0.025); the oracle difference B-A; boundary-censoring counts
"""
import argparse, json, sys, pathlib
import numpy as np
from scipy import stats
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
SEEDS = list(range(100, 112)); TAGS = ('Aplain', 'Bplain', 'Cconfidence'); SHORT = {'Aplain': 'A', 'Bplain': 'B', 'Cconfidence': 'C'}
KEY = 'fmin0.01_K4_r0'; E1 = 0.0005


def L(p):
    p = pathlib.Path(p)
    if not p.exists(): print(f'  missing {p}'); sys.exit(1)
    return json.loads(p.read_text(encoding='utf-8'))


def paired(d):
    d = np.asarray(d); n = len(d); m = float(d.mean()); sd = float(d.std(ddof=1))
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return {'n': n, 'effect': m, 'sd_d': sd, 'ci95_descriptive': [m - half, m + half], 'direction_positive': int((d > 0).sum()), 'per_seed': d.tolist()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dir', default='thrdiag_mass')
    ap.add_argument('--mass-dir', default='mass', help='name of the main-evaluation results directory (U: mass; V: mass_full); the MDE and the iou_frozen reference are taken from its summary/results')
    ap.add_argument('--l1-dir', default='stage1', help='directory the frozen thresholds come from (Appendix Y: y_whu)')
    ap.add_argument('--ref-dir', default=None, help='stratified reference directory (overall.iou / n_query); defaults to --mass-dir; Appendix Y WHU target: y_whu_strata (the MDE still comes from the --mass-dir summary)')
    a = ap.parse_args()
    D = ROOT / 'results' / a.dir
    R = {(t, s): L(D / f'{t}_seed{s}.json') for t in TAGS for s in SEEDS}
    mde = L(ROOT / 'results' / f'{a.mass_dir}_summary.json')['main']['MDE_K4_CA']
    out = {'seeds': SEEDS, 'note': 'U.9 upper bound / diagnostic; the oracle uses target labels; not in the results table; no significance claims', 'MDE_mass': mde, 'gates': {}}
    print('=' * 76); print('[gate] frozen threshold = L1; key; n_query = 501; iou_frozen agrees with the same key in results/mass (< 0.0005)'); print('=' * 76)
    bad = []; worst = 0.0
    for (t, s), d in R.items():
        l1 = L(ROOT / 'results' / a.l1_dir / f'{t}_seed{s}_{KEY}.json')
        if abs(d['frozen_threshold'] - l1['source_val_threshold']) > 1e-9: bad.append((t, s, 'threshold'))
        refj = L(ROOT / 'results' / (a.ref_dir or a.mass_dir) / f'{t}_seed{s}_{KEY}.json')
        if d['manifest_key'] != KEY or d['n_query'] != refj['n_query']: bad.append((t, s, 'key/n'))
        ref = refj['overall']['iou']
        dv = abs(d['iou_frozen'] - ref); worst = max(worst, dv)
        if dv >= E1: bad.append((t, s, f'iou_frozen vs mass {dv:.2e}'))
    if bad: print('  mismatch:', bad[:10]); print('  => stop'); sys.exit(2)
    print(f'  all passed (max|d| of iou_frozen vs results/mass = {worst:.2e})'); out['gates']['max_abs_vs_mass'] = worst

    print('\n' + '=' * 76); print('[arms] frozen vs widened oracle (0.05-0.95); calibration share = 1 - frozen/oracle'); print('=' * 76)
    print(f"  {'arm':>3s}{'IoU_frozen':>12s}{'IoU_oracle':>12s}{'share':>8s}{'frozen thr (med)':>18s}{'oracle thr (med)':>18s}{'oracle<frozen':>14s}{'boundary':>10s}")
    per = {}
    for t in TAGS:
        fr = np.array([R[(t, s)]['iou_frozen'] for s in SEEDS]); orc = np.array([R[(t, s)]['iou_oracle_wide'] for s in SEEDS])
        thr = np.array([R[(t, s)]['oracle_threshold'] for s in SEEDS]); fz = np.array([R[(t, s)]['frozen_threshold'] for s in SEEDS])
        nb = sum(int(R[(t, s)]['oracle_at_boundary']) for s in SEEDS); below = int((thr < fz).sum())
        share = 1 - fr / orc
        per[SHORT[t]] = {'iou_frozen': fr.tolist(), 'iou_oracle': orc.tolist(), 'share': share.tolist(), 'oracle_threshold': thr.tolist(),
                         'frozen_threshold': fz.tolist(), 'n_oracle_below_frozen': below, 'n_at_boundary': nb}
        print(f"  {SHORT[t]:>3s}{fr.mean():12.6f}{orc.mean():12.6f}{share.mean():8.3f}{np.median(fz):18.2f}{np.median(thr):18.2f}{below:>11d}/12{nb:>8d}/12")
    out['per_arm'] = per

    print('\n' + '=' * 76); print('[U9-H1] C-A / B-A / C-B with each arm at its own oracle'); print('=' * 76)
    dCA_f = paired(np.array(per['C']['iou_frozen']) - np.array(per['A']['iou_frozen']))
    dCA_o = paired(np.array(per['C']['iou_oracle']) - np.array(per['A']['iou_oracle']))
    dBA_o = paired(np.array(per['B']['iou_oracle']) - np.array(per['A']['iou_oracle']))
    dCB_o = paired(np.array(per['C']['iou_oracle']) - np.array(per['B']['iou_oracle']))
    for name, pr in (('C-A frozen', dCA_f), ('C-A oracle', dCA_o), ('B-A oracle', dBA_o), ('C-B oracle', dCB_o)):
        print(f"  {name:12s} effect {pr['effect']:+.6f}  CI [{pr['ci95_descriptive'][0]:+.6f}, {pr['ci95_descriptive'][1]:+.6f}]  positive {pr['direction_positive']}/12")
    op_share = 1 - dCA_o['effect'] / dCA_f['effect'] if dCA_f['effect'] != 0 else float('nan')
    print(f"  share attributable to the operating point, 1 - delta_oracle/delta_frozen = {op_share:.3f}")
    out['CA_frozen'] = dCA_f; out['CA_oracle'] = dCA_o; out['BA_oracle'] = dBA_o; out['CB_oracle'] = dCB_o; out['operating_point_share'] = op_share

    print('\n' + '=' * 76); print('[U9-H3] can A at its own oracle threshold catch up with C at the frozen threshold?'); print('=' * 76)
    catch = [float(ao >= cf) for ao, cf in zip(per['A']['iou_oracle'], per['C']['iou_frozen'])]
    n_catch = int(sum(catch)); gap = np.array(per['C']['iou_frozen']) - np.array(per['A']['iou_oracle'])
    print(f"  IoU_A^oracle >= IoU_C^frozen: {n_catch}/12; mean C_frozen - A_oracle {gap.mean():+.4f}")
    out['H3_A_oracle_ge_C_frozen'] = n_catch; out['C_frozen_minus_A_oracle'] = gap.tolist()

    # curves: mean IoU(threshold) of each arm over 0.05-0.95 (exploratory)
    print('\n' + '=' * 76); print('[exploratory] curve shape: mean IoU(threshold) of A and C over 0.05-0.95'); print('=' * 76)
    grid = [round(0.05 + 0.01 * i, 2) for i in range(91)]
    curves = {}
    for t in TAGS:
        c = np.mean([[R[(t, s)]['iou_by_threshold'][str(g)] for g in grid] for s in SEEDS], 0); curves[SHORT[t]] = c.tolist()
    for g in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        i = grid.index(g); print(f"  t={g:.2f}  A {curves['A'][i]:.4f}  B {curves['B'][i]:.4f}  C {curves['C'][i]:.4f}  C-A {curves['C'][i]-curves['A'][i]:+.4f}")
    out['mean_curves'] = {'grid': grid, **curves}
    ref_inria = {}
    try:
        st = L(ROOT / 'results' / 'thrdiag_stage1_summary.json')
        ref_inria = {k: st['per_arm'][k].get('frozen_over_oracle_mean') for k in st.get('per_arm', {})}
    except Exception: pass
    out['inria_f1_frozen_over_oracle'] = ref_inria

    print('\n' + '=' * 76); print('[verdicts]'); print('=' * 76)
    h1 = bool(dCA_o['effect'] >= mde and dCA_o['direction_positive'] >= 10)
    h2 = per['A']['n_oracle_below_frozen'] >= 9
    h3 = n_catch >= 9
    print(f"  U9-H1  delta_oracle {dCA_o['effect']:+.4f} vs MDE {mde:.4f}, direction {dCA_o['direction_positive']}/12 -> {'gain survives at the optimal operating point (not only an operating-point artefact)' if h1 else 'gain is mainly an operating-point / morphology effect'}")
    print(f"  U9-H2  A oracle threshold < frozen {per['A']['n_oracle_below_frozen']}/12 -> {'supported (A under-predicts; lowering the threshold dilates)' if h2 else 'not supported'}; C: {per['C']['n_oracle_below_frozen']}/12; B: {per['B']['n_oracle_below_frozen']}/12")
    print(f"  U9-H3  A_oracle >= C_frozen {n_catch}/12 -> {'strong artefact evidence' if h3 else 'does not hold'}")
    print(f"  U9-E1  share A {np.mean(per['A']['share']):.3f} B {np.mean(per['B']['share']):.3f} C {np.mean(per['C']['share']):.3f} (Inria f=1: A 0.037, C 0.025); at boundary A {per['A']['n_at_boundary']} B {per['B']['n_at_boundary']} C {per['C']['n_at_boundary']}")
    out['verdict'] = {'U9-H1': h1, 'U9-H2': h2, 'U9-H3': h3}
    outp = ROOT / 'results' / f'{a.dir}_summary.json'
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8'); print(f'\n  -> {outp}')
    print('\nReminder (Q.4): oracle numbers must not replace the primary-protocol numbers and must not enter the results table.')


if __name__ == '__main__':
    main()
