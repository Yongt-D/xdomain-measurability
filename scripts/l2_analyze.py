# -*- coding: utf-8 -*-
"""
12-seed analysis of L2 (the GSD-degradation curve); pre-registration §5, §6, Appendices L.3 / O.2 / O.3 / O.5 / P.3.
**Committed before any 12-seed L2 number existed.**

**No significance claims** (§3.3 + Appendix D.4 triggered): this script computes and prints **no p-value**;
every confidence interval is **descriptive**. §5 states that L2 is a descriptive result with no significance threshold.

Order of steps (must not be changed):
  [gate 0] provenance: the checkpoint_sha256 of every L2 result must equal that of the L1 result for the same
           (arm, seed) (guards against a mismatched checkpoint copied to another machine); the four GSD points of
           one (arm, seed) must share exactly the same source_val_threshold (O.2: one threshold for all four points).
  [gate 1] O.3 cross-check: |IoU_{f=1} − IoU_{L1}(K=4, r=0)| < 0.0005 for every (arm, seed);
           otherwise the L2 pipeline is wrong and the analysis **stops**. The difference is also reported per machine
           (L.3: it is one more independent measurement of cross-machine drift).
  [curves] A / B / C+conf at the frozen threshold: seed mean, seed SD and range at every GSD point (§5);
           retention IoU_f / IoU_1 (computed per seed, then aggregated).
  [contrasts] C−A, C−B, B−A at every GSD point: effect size + descriptive 95% CI + direction consistency
           (P.3 item 2: the degraded reading of §3.3).
  [L3]     the operationalisation of §6: g(s) = IoU(C+conf) − IoU(B), a(s) = IoU(B) − IoU(A).
           The "sign relation" is defined as sign(g(s) − a(s)), i.e. whether the geometry-pathway gain exceeds the
           appearance-pathway gain; if that sign (of the seed mean) changes across the four GSD points it is recorded
           as "reversal observed". The sign of C−A itself across GSD ("curve crossing" in the L2 pilot) is also reported.
           **Whatever the outcome, L3 never enters the primary criterion** (§6).
  [O.5]    frozen-vs-oracle diagnostic inside 0.40–0.60: IoU_frozen, IoU_oracle, argmax, ratio, and the number of
           seeds whose argmax sits at the lower bound 0.40. The oracle uses target labels and is **an upper bound /
           diagnostic only**, never a basis for arm comparisons; when the argmax sits at the boundary the oracle is a
           lower bound of the true oracle and the ratio an upper bound.

Aggregation as in L1: the independent unit is the training seed; K = 4, R = 1 (O.2).
"""
import argparse, json, sys, pathlib, collections
import numpy as np
from scipy import stats
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
TAGS = ('Aplain', 'Bplain', 'Cconfidence')
SHORT = {'Aplain': 'A', 'Bplain': 'B', 'Cconfidence': 'C'}
FACTORS = (1, 2, 4, 8)
GSD = {1: '0.3 m', 2: '0.6 m', 4: '1.2 m', 8: '2.4 m'}
E1 = 0.0005
THR = [round(0.40 + 0.01 * i, 2) for i in range(21)]


def load(p):
    if not p.exists():
        print(f'  missing {p}')
        sys.exit(1)
    return json.loads(p.read_text(encoding='utf-8'))


def ci(d):
    """Descriptive 95% CI (t quantile × SE); not a test."""
    d = np.asarray(d, float); n = d.size
    m, sd = float(d.mean()), float(d.std(ddof=1))
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return m, sd, m - half, m + half, int((d > 0).sum()), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--l1-stage', default='stage1')
    ap.add_argument('--l2-dir', default='l2')
    ap.add_argument('--seeds', default=','.join(str(s) for s in range(100, 112)))
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    L2 = ROOT / 'results' / a.l2_dir
    L1 = ROOT / 'results' / a.l1_stage

    # machine map (appended by l2_eval_all.sh; 'unknown' when absent)
    machines = {}
    mt = L2 / 'machines.tsv'
    if mt.exists():
        for line in mt.read_text(encoding='utf-8').splitlines():
            parts = line.split('\t')
            if len(parts) == 4:
                machines[(parts[0], int(parts[1]), int(parts[2]))] = parts[3]

    R = {}   # (tag, seed, f) -> json
    for t in TAGS:
        for s in seeds:
            for f in FACTORS:
                R[(t, s, f)] = load(L2 / f'{t}_seed{s}_gsd{f}.json')
    L1R = {(t, s): load(L1 / f'{t}_seed{s}_fmin0.01_K4_r0.json') for t in TAGS for s in seeds}

    print('=' * 76)
    print(f'[gate 0] provenance (checkpoint sha256 equals L1; same threshold at the four GSD points) -- n={len(seeds)} seeds')
    print('=' * 76)
    bad = []
    for t in TAGS:
        for s in seeds:
            shas = {R[(t, s, f)]['checkpoint_sha256'] for f in FACTORS}
            thr = {R[(t, s, f)]['source_val_threshold'] for f in FACTORS}
            if shas != {L1R[(t, s)]['checkpoint_sha256']}:
                bad.append((t, s, 'sha256 differs from the L1 checkpoint'))
            if len(thr) != 1:
                bad.append((t, s, f'threshold differs across the four GSD points {sorted(thr)}'))
            if R[(t, s, 1)]['manifest_key'] != 'fmin0.01_K4_r0':
                bad.append((t, s, 'manifest_key is not fmin0.01_K4_r0'))
    # labels are not down-sampled (O.1) => tp+fn (total positive label pixels) and n_query must be identical in every
    # evaluation; a difference means a batch was lost or the view/label configuration is wrong (an fd error once
    # occurred while dataloader workers were shutting down).
    posref = {(tp + fn, d['target']['n_query']) for d in R.values() for tp, fp, fn in [d['target']['tp_fp_fn']]}
    if len(posref) != 1:
        bad.append(('*', '*', f'tp+fn / n_query not unique: {sorted(posref)}'))
    l1pos = {(tp + fn, d['target']['n_query']) for d in L1R.values() for tp, fp, fn in [d['target']['tp_fp_fn']]}
    if l1pos != posref:
        bad.append(('*', '*', f'L2 tp+fn / n_query {sorted(posref)} differ from L1 {sorted(l1pos)}'))
    if bad:
        for b in bad[:20]:
            print('  ', b)
        print('  => **STOP**: provenance mismatch; check checkpoint copies / view configuration first')
        sys.exit(2)
    print('  all passed')

    print('\n' + '=' * 76)
    print(f'[gate 1] O.3 cross-check: |IoU_(f=1) - IoU_L1| < {E1} (same checkpoint, K=4, r=0)')
    print('=' * 76)
    by_machine = collections.defaultdict(list)
    worst = (0.0, None)
    for t in TAGS:
        for s in seeds:
            d = R[(t, s, 1)]['target']['iou'] - L1R[(t, s)]['target']['iou']
            m = machines.get((t, s, 1), 'unknown')
            by_machine[m].append(d)
            if abs(d) > worst[0]:
                worst = (abs(d), (t, s, m))
            dthr = R[(t, s, 1)]['source_val_threshold'] - L1R[(t, s)]['source_val_threshold']
            if abs(dthr) > 1e-9:
                print(f'  note: {t} seed{s} frozen threshold L2={R[(t, s, 1)]["source_val_threshold"]} '
                      f'!= L1={L1R[(t, s)]["source_val_threshold"]} (near-tie argmax flipped across machines)')
    for m, ds in sorted(by_machine.items()):
        ds = np.abs(np.asarray(ds))
        print(f'  machine {m:20s} n={ds.size:2d}  max|d| = {ds.max():.8f}  mean|d| = {ds.mean():.8f}')
    print(f'  global max|d| = {worst[0]:.8f} @ {worst[1]}')
    if worst[0] >= E1:
        print(f'  => **FAIL** (>= {E1}); the L2 pipeline is wrong; stop, do not analyse')
        sys.exit(2)
    print('  => pass (note: same-machine runs should be bit-identical; cross-machine differences are the drift L.3 asks to report)')

    # ---------- main curves ----------
    print('\n' + '=' * 76)
    print('[curves] IoU at the frozen threshold (selected on the WHU source validation set) -- the only curve reportable as a result (O.5)')
    print('=' * 76)
    V = {t: np.array([[R[(t, s, f)]['target']['iou'] for f in FACTORS] for s in seeds]) for t in TAGS}  # (n, 4)
    out = {'seeds': seeds, 'factors': list(FACTORS), 'main_curve': {}, 'retention': {},
           'arm_diff': {}, 'L3': {}, 'O5_diag': {},
           'note': 'L2 descriptive result; no significance claims; CIs are descriptive (section 3.3 + appendix D.4)'}
    print(f"  {'GSD':>7s}" + ''.join(f'{SHORT[t]+" mean":>14s}{"SD":>10s}{"[min,max]":>20s}' for t in TAGS))
    for j, f in enumerate(FACTORS):
        row = f'  {GSD[f]:>7s}'
        for t in TAGS:
            v = V[t][:, j]
            row += f'{v.mean():>14.6f}{v.std(ddof=1):>10.6f}{"[" + format(v.min(), ".4f") + ", " + format(v.max(), ".4f") + "]":>20s}'
            out['main_curve'][f'{SHORT[t]}_gsd{f}'] = {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)),
                                                     'min': float(v.min()), 'max': float(v.max()),
                                                     'per_seed': v.tolist()}
        print(row)

    print('\n  retention IoU_f / IoU_1 (per seed, then aggregated; descriptive)')
    print(f"  {'GSD':>7s}" + ''.join(f'{SHORT[t]+" mean":>14s}{"SD":>10s}' for t in TAGS))
    for j, f in enumerate(FACTORS):
        row = f'  {GSD[f]:>7s}'
        for t in TAGS:
            r = V[t][:, j] / V[t][:, 0]
            row += f'{r.mean():>14.4f}{r.std(ddof=1):>10.4f}'
            out['retention'][f'{SHORT[t]}_gsd{f}'] = {'mean': float(r.mean()), 'sd': float(r.std(ddof=1)),
                                                    'per_seed': r.tolist()}
        print(row)

    # ---------- arm contrasts ----------
    print('\n' + '=' * 76)
    print('[contrasts] effect size + descriptive 95% CI + direction consistency at every GSD point (no significance claims)')
    print('=' * 76)
    for lhs, rhs in (('Cconfidence', 'Aplain'), ('Cconfidence', 'Bplain'), ('Bplain', 'Aplain')):
        print(f'  {SHORT[lhs]} - {SHORT[rhs]}:')
        for j, f in enumerate(FACTORS):
            m, sd, lo, hi, pos, n = ci(V[lhs][:, j] - V[rhs][:, j])
            print(f'    {GSD[f]:>7s}  effect {m:+.6f}   CI [{lo:+.6f}, {hi:+.6f}]   paired-diff SD {sd:.6f}   positive {pos}/{n}')
            out['arm_diff'][f'{SHORT[lhs]}_minus_{SHORT[rhs]}_gsd{f}'] = {
                'effect': m, 'sd_d': sd, 'ci95_descriptive': [lo, hi], 'direction_positive': pos, 'n': n,
                'per_seed': (V[lhs][:, j] - V[rhs][:, j]).tolist()}

    # ---------- L3 ----------
    print('\n' + '=' * 76)
    print('[L3] section 6: g(s) = C - B (geometry-pathway gain), a(s) = B - A (appearance-pathway gain); sign relation := sign(g - a)')
    print('     **L3 never enters the primary criterion** (section 6).')
    print('=' * 76)
    g = V['Cconfidence'] - V['Bplain']
    aa = V['Bplain'] - V['Aplain']
    rel_sign, ca_sign = [], []
    print(f"  {'GSD':>7s}{'g mean':>12s}{'g>0':>6s}{'a mean':>12s}{'a>0':>6s}{'g-a mean':>12s}{'g>a':>6s}{'C-A mean':>12s}{'C>A':>6s}")
    for j, f in enumerate(FACTORS):
        gm, am = float(g[:, j].mean()), float(aa[:, j].mean())
        gma = g[:, j] - aa[:, j]
        cam = float((V['Cconfidence'][:, j] - V['Aplain'][:, j]).mean())
        rel_sign.append(int(np.sign(gma.mean())))
        ca_sign.append(int(np.sign(cam)))
        print(f'  {GSD[f]:>7s}{gm:>+12.6f}{int((g[:, j] > 0).sum()):>4d}/{len(seeds):<2d}{am:>+12.6f}'
              f'{int((aa[:, j] > 0).sum()):>4d}/{len(seeds):<2d}{gma.mean():>+12.6f}{int((gma > 0).sum()):>4d}/{len(seeds):<2d}'
              f'{cam:>+12.6f}{int(((V["Cconfidence"][:, j] - V["Aplain"][:, j]) > 0).sum()):>4d}/{len(seeds):<2d}')
        out['L3'][f'gsd{f}'] = {'g_mean': gm, 'a_mean': am, 'g_minus_a_mean': float(gma.mean()),
                                'g_pos': int((g[:, j] > 0).sum()), 'a_pos': int((aa[:, j] > 0).sum()),
                                'g_gt_a': int((gma > 0).sum()), 'C_minus_A_mean': cam}
    rev = len({s for s in rel_sign if s != 0}) > 1
    ca_cross = len({s for s in ca_sign if s != 0}) > 1
    print(f'  sign(g - a) by GSD: {rel_sign}  => sign-relation reversal: **{"observed" if rev else "not observed"}**')
    print(f'  sign(C - A) by GSD: {ca_sign}  => C/A curve crossing (pilot wording): **{"observed" if ca_cross else "not observed"}**')
    if rev:
        print('  => per section 6: report as a finding; state that it runs against the general conclusion of arXiv:1911.09071 / 2602.13859 and needs independent replication;')
    else:
        print('  => per section 6: report as a boundary condition; withdraw the conjecture of literature scan 3, section 3.4;')
    print('  and a crossing enters the paper only if all three conditions of appendix P.3 hold (control X judged X2 already holds; this script provides item 2; item 3 comes from the widened diagnostic).')
    out['L3']['sign_g_minus_a_by_gsd'] = rel_sign
    out['L3']['sign_C_minus_A_by_gsd'] = ca_sign
    out['L3']['reversal_observed'] = rev
    out['L3']['CA_crossing_observed'] = ca_cross

    # ---------- O.5 ----------
    print('\n' + '=' * 76)
    print('[O.5 diagnostic] frozen threshold vs oracle inside 0.40-0.60 (uses target labels: **upper bound / diagnostic**, not in the results table, no arm comparison)')
    print('=' * 76)
    print(f"  {'arm':>3s}{'GSD':>8s}{'IoU_frozen':>12s}{'IoU_oracle':>12s}{'frozen/oracle':>15s}{'argmax=0.40':>13s}")
    for t in TAGS:
        for j, f in enumerate(FACTORS):
            fr, orc, ratio, atlo = [], [], [], 0
            for s in seeds:
                d = R[(t, s, f)]
                ts = d['target_threshold_sensitivity']
                best = max(THR, key=lambda x: ts[str(x)])
                fr.append(d['target']['iou']); orc.append(ts[str(best)])
                ratio.append(d['target']['iou'] / max(ts[str(best)], 1e-12))
                atlo += int(best == THR[0])
            print(f'  {SHORT[t]:>3s}{GSD[f]:>8s}{np.mean(fr):>12.6f}{np.mean(orc):>12.6f}{np.mean(ratio):>15.3f}'
                  f'{atlo:>9d}/{len(seeds)}')
            out['O5_diag'][f'{SHORT[t]}_gsd{f}'] = {'iou_frozen_mean': float(np.mean(fr)),
                                                  'iou_oracle_0.40_0.60_mean': float(np.mean(orc)),
                                                  'frozen_over_oracle_mean': float(np.mean(ratio)),
                                                  'n_argmax_at_lower_bound': atlo}
    print('  note: when the argmax sits at 0.40 the oracle is a lower bound of the true oracle and the ratio an upper bound (P.3 item 3); the widened sweep is in threshold_range_diag.')

    outp = pathlib.Path(a.out) if a.out else ROOT / 'results' / f'{a.l2_dir}_summary.json'
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n  -> {outp}')
    print('\nReminder: L2 absolute values must not be pooled with L1 (L.3); under this operationalisation the pixel size is '
          'unchanged and only the information content drops, so it must not be described as a "pixel-size threshold" (O.1.1).')


if __name__ == '__main__':
    main()
