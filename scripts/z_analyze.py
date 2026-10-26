# -*- coding: utf-8 -*-
"""
Appendix Z analysis: gates, stop-loss and Z-H1..H5 for the 36 stage-1 checkpoints on the third/fourth target domains (z_sat1 / z_sat2).
**Committed before any Z number existed.** No significance claims; CIs descriptive; the MDE uses the same algorithm as mass_analyze.
Usage: python scripts/z_analyze.py --domain sat1   (once per domain; the four-domain correlation of Z-H4 is computed when both domains exist)
Writes results/z_{dom}_summary.json (with main.MDE_K4_CA, used by mass_thrdiag_analyze.py --mass-dir z_{dom}).
"""
import argparse, glob, json, sys, pathlib
import numpy as np
from scipy import stats
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
SEEDS = list(range(100, 112)); TAGS = ('Aplain', 'Bplain', 'Cconfidence'); SHORT = {'Aplain': 'A', 'Bplain': 'B', 'Cconfidence': 'C'}
KEYS_C = [f'fmin0.01_K{k}_r{r}' for k in (1, 4, 8) for r in range(5)]; KEY = 'fmin0.01_K4_r0'; E1 = 0.0005
S1, S4, BIG_EDGE, BIG_INT, NEAR, FAR = [0, 1], [6, 7], [4, 6], [5, 7], 8, 9


def L(p):
    p = pathlib.Path(p)
    if not p.exists(): print(f'  missing {p}'); sys.exit(1)
    return json.loads(p.read_text(encoding='utf-8'))


def mde(s_d, n=12):
    lo, hi = 1e-5, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2; ncp = mid / (s_d / np.sqrt(n)); crit = stats.t.ppf(0.975, n - 1)
        p = stats.nct.sf(crit, n - 1, ncp) + stats.nct.cdf(-crit, n - 1, ncp)
        if p >= 0.80: hi = mid
        else: lo = mid
    return float(hi)


def paired(d):
    d = np.asarray(d, float); n = len(d); m = float(d.mean()); sd = float(d.std(ddof=1)); half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return {'n': n, 'effect': m, 'sd_d': sd, 'MDE': mde(sd, n), 'ci95_descriptive': [m - half, m + half], 'direction_positive': int((d > 0).sum()), 'per_seed': d.tolist()}


def rec(rows, idx):
    n = np.array([r['n'] for r in rows]); p1 = np.array([r['p1'] for r in rows]); return p1[:, idx].sum() / max(n[:, idx].sum(), 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--domain', required=True, choices=['sat1', 'sat2']); a = ap.parse_args()
    R = ROOT / 'results'; D = R / f'z_{a.domain}'
    out = {'domain': a.domain, 'seeds': SEEDS, 'note': 'appendix Z; descriptive; no significance claims', 'gates': {}, 'H': {}}
    res = {}
    for t in TAGS:
        for s in SEEDS:
            for k in (KEYS_C if t == 'Cconfidence' else [KEY]):
                res[(t, s, k)] = L(D / f'{t}_seed{s}_{k}.json')
    man = L(R / 'support_manifests' / f'z_{a.domain}_support_manifests.json')
    print('=' * 78); print('[gate U.3.1] provenance: sha256 = same checkpoint as results/stage1; frozen threshold = L1; support list = manifest'); print('=' * 78)
    bad = []
    for (t, s, k), d in res.items():
        l1 = L(R / 'stage1' / f'{t}_seed{s}_{KEY}.json')
        if d['checkpoint_sha256'] != l1['checkpoint_sha256']: bad.append((t, s, k, 'sha'))
        if abs(d['threshold'] - l1['source_val_threshold']) > 1e-9: bad.append((t, s, k, 'thr'))
        if d['support_files'] != man[k]['files']: bad.append((t, s, k, 'support'))
    if bad: print('  mismatch:', bad[:8]); sys.exit(2)
    print('  all passed'); out['gates']['provenance'] = 'pass'
    print('\n' + '=' * 78); print('[gate U.3.2] cross-machine calibration: A seed100 K4_r0, local machine vs evaluation machine, < 0.0005'); print('=' * 78)
    worst = 0.0; calib = sorted(glob.glob(str(R / f'z_{a.domain}_calib' / '*.json')))
    ref = res[('Aplain', 100, KEY)]['overall']['iou']
    for c in calib:
        v = L(c)['overall']['iou']; worst = max(worst, abs(v - ref)); print(f"  {pathlib.Path(c).name}: {v:.6f} vs {ref:.6f} d={v-ref:+.2e}")
    out['gates']['calib_max_abs'] = worst; out['gates']['calib_pass'] = bool(calib) and worst < E1
    print(f"  => {'pass' if out['gates']['calib_pass'] else 'fail / no calibration file (must be disclosed)'}")
    print('\n' + '=' * 78); print('[gate U.3.3] label consistency: stratum totals bit-identical'); print('=' * 78)
    tots = {tuple(d['strata_n_total']) for d in res.values()}; tots_alt = {json.dumps(d.get('strata_n_total_alt', {}), sort_keys=True) for d in res.values()}
    if len(tots) != 1 or len(tots_alt) != 1: print('  mismatch => stop'); sys.exit(2)
    tot = next(iter(tots)); pos = sum(tot[:8]); neg = tot[8] + tot[9]
    print('  pass. positive-pixel shares: ' + ' '.join(f"S{i+1}({'edge' if j == 0 else 'int'}) {tot[2*i+j]/pos*100:.1f}%" for i in range(4) for j in range(2)) + f"; near band {tot[8]/neg*100:.1f}%")
    out['gates']['strata_n_total'] = list(tot); out['gates']['n_query'] = res[('Aplain', 100, KEY)]['n_query']

    def iou(t, s, k): return res[(t, s, k)]['overall']['iou']
    A = np.array([iou('Aplain', s, KEY) for s in SEEDS]); B = np.array([iou('Bplain', s, KEY) for s in SEEDS])
    print('\n' + '=' * 78); print(f"[stop-loss U.4] arm A K4_r0 mean >= 0.15: {A.mean():.4f} -> {'triggered: descriptive only' if A.mean() < 0.15 else 'not triggered'}"); print('=' * 78)
    out['stop_loss'] = {'A_mean': float(A.mean()), 'triggered': bool(A.mean() < 0.15)}
    main = {}
    print('\n' + '=' * 78); print('[primary] frozen threshold; arm C averaged over R=5 draws; A/B structurally identical across K'); print('=' * 78)
    for K in (1, 4, 8):
        C = np.array([np.mean([iou('Cconfidence', s, f'fmin0.01_K{K}_r{r}') for r in range(5)]) for s in SEEDS])
        ca, cb, ba = paired(C - A), paired(C - B), paired(B - A)
        main[f'K{K}_C_minus_A'], main[f'K{K}_C_minus_B'], main[f'K{K}_B_minus_A'] = ca, cb, ba
        main[f'K{K}_arms'] = {'A': {'mean': float(A.mean()), 'sd': float(A.std(ddof=1)), 'per_seed': A.tolist()}, 'B': {'mean': float(B.mean()), 'sd': float(B.std(ddof=1)), 'per_seed': B.tolist()}, 'C': {'mean': float(C.mean()), 'sd': float(C.std(ddof=1)), 'per_seed': C.tolist()}}
        print(f"  K={K}: A {A.mean():.4f} ({A.std(ddof=1):.4f})  B {B.mean():.4f} ({B.std(ddof=1):.4f})  C {C.mean():.4f} ({C.std(ddof=1):.4f})  C-A {ca['effect']:+.4f} [{ca['ci95_descriptive'][0]:+.4f}, {ca['ci95_descriptive'][1]:+.4f}] {ca['direction_positive']}/12 MDE {ca['MDE']:.4f}  C-B {cb['effect']:+.4f} ({cb['direction_positive']}/12)  B-A {ba['effect']:+.4f} ({ba['direction_positive']}/12)")
    main['MDE_K4_CA'] = main['K4_C_minus_A']['MDE']; out['main'] = main
    ca = main['K4_C_minus_A']; h1 = 'detectable' if (ca['direction_positive'] >= 10 and ca['effect'] >= ca['MDE']) else ('detectable_negative' if (ca['direction_positive'] <= 2 and -ca['effect'] >= ca['MDE']) else 'unresolved')
    out['H']['Z-H1'] = h1
    # in-domain SD and amplification (Z-H5)
    IN = {t: np.array([L(R / 'stage1_train' / f'{t}_seed{s}.results.json')['test_metrics']['iou'] for s in SEEDS]) for t in TAGS}
    Cm = np.array(main['K4_arms']['C']['per_seed'])
    amp = {'A': float(A.std(ddof=1) / IN['Aplain'].std(ddof=1)), 'B': float(B.std(ddof=1) / IN['Bplain'].std(ddof=1)), 'C': float(Cm.std(ddof=1) / IN['Cconfidence'].std(ddof=1))}
    out['H']['Z-H5'] = {'amp': amp, 'pass': all(v >= 3 for v in amp.values())}
    # stratification (K4_r0, primary band width)
    def strata_counts(tag_hi, tag_lo):
        dS1 = []; dS4 = []; dE = []; dI = []; dN = []; dF = []
        for s in SEEDS:
            rh, rl = res[(tag_hi, s, KEY)]['rows'], res[(tag_lo, s, KEY)]['rows']
            dS1.append(rec(rh, S1) - rec(rl, S1)); dS4.append(rec(rh, S4) - rec(rl, S4)); dE.append(rec(rh, BIG_EDGE) - rec(rl, BIG_EDGE)); dI.append(rec(rh, BIG_INT) - rec(rl, BIG_INT))
            dN.append(rec(rh, [NEAR]) - rec(rl, [NEAR])); dF.append(rec(rh, [FAR]) - rec(rl, [FAR]))
        dS1, dS4, dE, dI, dN, dF = map(np.array, (dS1, dS4, dE, dI, dN, dF))
        return {'dR_S1': float(dS1.mean()), 'dR_S4': float(dS4.mean()), 'count_S4_lt_S1': int((dS4 < dS1).sum()), 'dR_big_edge': float(dE.mean()), 'dR_big_int': float(dI.mean()), 'count_int_lt_edge': int((dI < dE).sum()),
                'dFPR_near': float(dN.mean()), 'count_near_gt0': int((dN > 0).sum()), 'dFPR_far': float(dF.mean()), 'count_far_gt0': int((dF > 0).sum())}
    st = {'C_minus_A': strata_counts('Cconfidence', 'Aplain'), 'B_minus_A': strata_counts('Bplain', 'Aplain')}
    dgm = float(np.mean([np.median([r['dgm_rel'] for r in res[('Cconfidence', s, KEY)]['rows']]) for s in SEEDS]))
    out['H']['Z-E1'] = {'strata': st, 'dgm_rel_median_mean': dgm}
    # operating-point diagnostic (if already generated)
    td = R / f'thrdiag_z_{a.domain}'
    if all((td / f'{t}_seed{s}.json').exists() for t in TAGS for s in SEEDS):
        T = {(t, s): L(td / f'{t}_seed{s}.json') for t in TAGS for s in SEEDS}
        below = {SHORT[t]: int(sum(T[(t, s)]['oracle_threshold'] < T[(t, s)]['frozen_threshold'] for s in SEEDS)) for t in TAGS}
        share = {SHORT[t]: float(np.mean([1 - T[(t, s)]['iou_frozen'] / T[(t, s)]['iou_oracle_wide'] for s in SEEDS])) for t in TAGS}
        ca_o = paired(np.array([T[('Cconfidence', s)]['iou_oracle_wide'] for s in SEEDS]) - np.array([T[('Aplain', s)]['iou_oracle_wide'] for s in SEEDS]))
        ca_f = paired(np.array([T[('Cconfidence', s)]['iou_frozen'] for s in SEEDS]) - np.array([T[('Aplain', s)]['iou_frozen'] for s in SEEDS]))
        op_share = 1 - ca_o['effect'] / ca_f['effect'] if abs(ca_f['effect']) > 1e-9 else float('nan')
        out['H']['Z-H2'] = {'oracle_below_frozen': below, 'share': share, 'pass': below['A'] >= 9, 'oracle_median': {SHORT[t]: float(np.median([T[(t, s)]['oracle_threshold'] for s in SEEDS])) for t in TAGS}}
        out['H']['Z-H3'] = {'CA_oracle': ca_o, 'operating_point_share': op_share, 'applies': h1 == 'detectable', 'pass': (ca_o['effect'] < ca['MDE'] and op_share >= 0.5) if h1 == 'detectable' else None}
    else:
        out['H']['Z-H2'] = None; out['H']['Z-H3'] = None
    # Z-H4: seed ranking of arm A across the four target domains (Inria, Mass, sat1, sat2 when available)
    doms = {'inria': np.array([L(R / 'stage1' / f'Aplain_seed{s}_{KEY}.json')['target']['iou'] for s in SEEDS]), 'mass': np.array([L(R / 'mass' / f'Aplain_seed{s}_{KEY}.json')['overall']['iou'] for s in SEEDS]), a.domain: A}
    other = 'sat2' if a.domain == 'sat1' else 'sat1'
    if all((R / f'z_{other}' / f'Aplain_seed{s}_{KEY}.json').exists() for s in SEEDS):
        doms[other] = np.array([L(R / f'z_{other}' / f'Aplain_seed{s}_{KEY}.json')['overall']['iou'] for s in SEEDS])
    names = list(doms); rho = {f'{p}-{q}': float(stats.spearmanr(doms[p], doms[q]).correlation) for i, p in enumerate(names) for q in names[i + 1:]}
    out['H']['Z-H4'] = {'rho_A': rho, 'pass': all(v < 0.5 for v in rho.values())}
    print('\n' + '=' * 78); print('[verdicts]'); print('=' * 78)
    print(f"  Z-H1 measurability: C-A (K=4) = {ca['effect']:+.4f} vs MDE {ca['MDE']:.4f}, {ca['direction_positive']}/12 -> {h1}")
    if out['H']['Z-H2']: print(f"  Z-H2 baseline drift: A oracle<frozen {out['H']['Z-H2']['oracle_below_frozen']['A']}/12 (share A {out['H']['Z-H2']['share']['A']:.3f} B {out['H']['Z-H2']['share']['B']:.3f} C {out['H']['Z-H2']['share']['C']:.3f}; oracle median {out['H']['Z-H2']['oracle_median']}) -> {'holds' if out['H']['Z-H2']['pass'] else 'does not hold'}")
    if out['H']['Z-H3']: print(f"  Z-H3 calibration transfer: C-A at oracle {out['H']['Z-H3']['CA_oracle']['effect']:+.4f} ({out['H']['Z-H3']['CA_oracle']['direction_positive']}/12), share {out['H']['Z-H3']['operating_point_share']:.3f} -> {out['H']['Z-H3']['pass'] if out['H']['Z-H3']['applies'] else 'not applicable (undetectable at the frozen threshold)'}")
    print(f"  Z-H4 seed ranking (A): { {k: round(v, 3) for k, v in rho.items()} } -> {'holds' if out['H']['Z-H4']['pass'] else 'does not hold'}")
    print(f"  Z-H5 variance amplification: A {amp['A']:.1f}x B {amp['B']:.1f}x C {amp['C']:.1f}x -> {'holds' if out['H']['Z-H5']['pass'] else 'does not hold'}")
    print(f"  Z-E1 strata C-A: S4<S1 {st['C_minus_A']['count_S4_lt_S1']}/12 (dR S1 {st['C_minus_A']['dR_S1']:+.3f} S4 {st['C_minus_A']['dR_S4']:+.3f}), interior<edge {st['C_minus_A']['count_int_lt_edge']}/12, near-band dFPR {st['C_minus_A']['dFPR_near']:+.3f} ({st['C_minus_A']['count_near_gt0']}/12); B-A S4<S1 {st['B_minus_A']['count_S4_lt_S1']}/12; DGM residual {dgm:.3f}")
    p = R / f'z_{a.domain}_summary.json'; p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8'); print(f'\n  -> {p.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
