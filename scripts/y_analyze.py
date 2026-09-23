# -*- coding: utf-8 -*-
"""
Appendix Y analysis: gates, stop-loss and Y-H1..H5 for the three arms x 12 seeds trained on Inria and evaluated on the WHU and Massachusetts targets.
**Committed before any Y number existed.** No significance claims; CIs descriptive; the MDE uses the same algorithm as mass_analyze.
Inputs: results/y_train (copies of the training results.json), results/y_whu (L1), results/y_whu_strata, results/y_mass, results/thrdiag_y_whu,
        results/thrdiag_y_mass, results/y_calib (GPU0 vs GPU1), results/y_ckpt_sha256.txt.
Writes results/y_summary.json and results/y_{whu,mass}_summary.json (main.MDE_K4_CA, used by mass_thrdiag_analyze --mass-dir).
"""
import glob, json, sys, pathlib
import numpy as np
from scipy import stats
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
SEEDS = list(range(100, 112)); TAGS = ('Aplain', 'Bplain', 'Cconfidence'); SHORT = {'Aplain': 'A', 'Bplain': 'B', 'Cconfidence': 'C'}
KEYS_C = [f'fmin0.01_K{k}_r{r}' for k in (1, 4, 8) for r in range(5)]; KEY = 'fmin0.01_K4_r0'; E1 = 0.0005
S1, S4, BIG_EDGE, BIG_INT, NEAR = [0, 1], [6, 7], [4, 6], [5, 7], 8
GRID = [round(0.05 + 0.01 * i, 2) for i in range(91)]


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


def verdict(pr):
    if abs(pr['effect']) >= pr['MDE'] and (pr['direction_positive'] >= 10 or pr['direction_positive'] <= 2): return 'pos' if pr['effect'] > 0 else 'neg'
    return 'unresolved'


def rec(rows, idx):
    n = np.array([r['n'] for r in rows]); p1 = np.array([r['p1'] for r in rows]); return p1[:, idx].sum() / max(n[:, idx].sum(), 1)


def main():
    R = ROOT / 'results'
    out = {'seeds': SEEDS, 'note': 'appendix Y; descriptive; no significance claims', 'gates': {}, 'stop_loss': {}, 'H': {}, 'targets': {}}
    TR = {t: {s: L(R / 'y_train' / f'{t}_seed{s}.results.json') for s in SEEDS} for t in TAGS}
    L1 = {t: {s: L(R / 'y_whu' / f'{t}_seed{s}_{KEY}.json') for s in SEEDS} for t in TAGS}
    ST = {'whu': {}, 'mass': {}}
    for t in TAGS:
        for s in SEEDS:
            for k in (KEYS_C if t == 'Cconfidence' else [KEY]):
                ST['whu'][(t, s, k)] = L(R / 'y_whu_strata' / f'{t}_seed{s}_{k}.json'); ST['mass'][(t, s, k)] = L(R / 'y_mass' / f'{t}_seed{s}_{k}.json')
    TH = {'whu': {(t, s): L(R / 'thrdiag_y_whu' / f'{t}_seed{s}.json') for t in TAGS for s in SEEDS}, 'mass': {(t, s): L(R / 'thrdiag_y_mass' / f'{t}_seed{s}.json') for t in TAGS for s in SEEDS}}
    mans = {'whu': L(R / 'support_manifests' / 'whu_support_manifests.json'), 'mass': L(R / 'support_manifests' / 'mass_support_manifests.json')}

    print('=' * 78); print('[gate Y.3.1] provenance: sha256 = training-machine digest list; frozen threshold = L1; support list = manifest'); print('=' * 78)
    sha = {}
    for line in (R / 'y_ckpt_sha256.txt').read_text(encoding='utf-8').splitlines():
        if line.strip():
            h, p = line.split(maxsplit=1); sha[p.strip().lstrip('*')] = h
    bad = []
    for t in TAGS:
        for s in SEEDS:
            ref = sha.get(f'y_{t}_seed{s}.pth') or sha.get(f'outputs/y_{t}_seed{s}/best_gaplsegnet_v5.pth'); ft = L1[t][s]['source_val_threshold']
            if L1[t][s]['checkpoint_sha256'] != ref: bad.append((t, s, 'l1 sha'))
            for dom in ('whu', 'mass'):
                for k in (KEYS_C if t == 'Cconfidence' else [KEY]):
                    d = ST[dom][(t, s, k)]
                    if d['checkpoint_sha256'] != ref: bad.append((t, s, k, dom, 'sha'))
                    if abs(d['threshold'] - ft) > 1e-9: bad.append((t, s, k, dom, 'thr'))
                    if d['support_files'] != mans[dom][k]['files']: bad.append((t, s, k, dom, 'support'))
                if abs(TH[dom][(t, s)]['frozen_threshold'] - ft) > 1e-9: bad.append((t, s, dom, 'thrdiag thr'))
    if bad: print('  mismatch:', bad[:8]); sys.exit(2)
    print('  all passed'); out['gates']['provenance'] = 'pass'

    print('\n' + '=' * 78); print('[gate Y.3.2] cross-device calibration: A seed100 K4_r0 on the WHU target, GPU0 vs GPU1 < 0.0005'); print('=' * 78)
    calib = sorted(c for c in glob.glob(str(R / 'y_calib' / '*.json')) if not c.endswith('.per_sample.json')); vals = {pathlib.Path(c).name: L(c)['target']['iou'] for c in calib}
    for k, v in vals.items(): print(f'  {k}: {v:.6f}')
    worst = (max(vals.values()) - min(vals.values())) if len(vals) >= 2 else float('nan')
    out['gates']['calib_max_abs'] = worst; out['gates']['calib_pass'] = len(vals) >= 2 and worst < E1
    print(f"  => {'pass' if out['gates']['calib_pass'] else 'fail / missing'} (d = {worst:.2e})")

    print('\n' + '=' * 78); print('[gate Y.3.3] label consistency: stratum totals bit-identical'); print('=' * 78)
    for dom in ('whu', 'mass'):
        tots = {tuple(d['strata_n_total']) for d in ST[dom].values()}
        if len(tots) != 1: print(f'  {dom} mismatch => stop'); sys.exit(2)
    if ST['mass'][('Aplain', 100, KEY)]['strata_n_total'] != L(R / 'mass' / f'Aplain_seed100_{KEY}.json')['strata_n_total']: print('  mass differs from results/mass => stop'); sys.exit(2)
    print('  pass'); out['gates']['labels'] = 'pass'

    print('\n' + '=' * 78); print('[stop-loss Y.4] in-domain (Inria test, 0.5) mean >= 0.60; target arm A mean >= 0.15'); print('=' * 78)
    IN = {t: np.array([TR[t][s]['test_metrics']['iou'] for s in SEEDS]) for t in TAGS}
    for t in TAGS: print(f"  {SHORT[t]} in-domain {IN[t].mean():.4f} (SD {IN[t].std(ddof=1):.4f})  median best epoch {np.median([TR[t][s]['best_epoch'] for s in SEEDS]):.0f}")
    live = all(IN[t].mean() >= 0.60 for t in TAGS)
    out['stop_loss']['indomain'] = {SHORT[t]: {'mean': float(IN[t].mean()), 'sd': float(IN[t].std(ddof=1)), 'per_seed': IN[t].tolist()} for t in TAGS}

    def iou(dom, t, s, k): return ST[dom][(t, s, k)]['overall']['iou']
    arms = {}
    for dom in ('whu', 'mass'):
        A = np.array([iou(dom, 'Aplain', s, KEY) for s in SEEDS]); B = np.array([iou(dom, 'Bplain', s, KEY) for s in SEEDS])
        live = live and A.mean() >= 0.15
        main = {}
        print('\n' + '=' * 78); print(f'[{dom}] primary criterion (frozen threshold; arm C averaged over R=5 draws)  A mean {A.mean():.4f} -> stop-loss {"triggered" if A.mean() < 0.15 else "not triggered"}'); print('=' * 78)
        for K in (1, 4, 8):
            C = np.array([np.mean([iou(dom, 'Cconfidence', s, f'fmin0.01_K{K}_r{r}') for r in range(5)]) for s in SEEDS])
            main[f'K{K}_C_minus_A'], main[f'K{K}_C_minus_B'], main[f'K{K}_B_minus_A'] = paired(C - A), paired(C - B), paired(B - A)
            main[f'K{K}_arms'] = {'A': {'mean': float(A.mean()), 'sd': float(A.std(ddof=1)), 'per_seed': A.tolist()}, 'B': {'mean': float(B.mean()), 'sd': float(B.std(ddof=1)), 'per_seed': B.tolist()}, 'C': {'mean': float(C.mean()), 'sd': float(C.std(ddof=1)), 'per_seed': C.tolist()}}
            ca = main[f'K{K}_C_minus_A']
            print(f"  K={K}: A {A.mean():.4f} ({A.std(ddof=1):.4f})  B {B.mean():.4f} ({B.std(ddof=1):.4f})  C {C.mean():.4f} ({C.std(ddof=1):.4f})  C-A {ca['effect']:+.4f} [{ca['ci95_descriptive'][0]:+.4f}, {ca['ci95_descriptive'][1]:+.4f}] {ca['direction_positive']}/12 MDE {ca['MDE']:.4f}  C-B {main[f'K{K}_C_minus_B']['effect']:+.4f} ({main[f'K{K}_C_minus_B']['direction_positive']}/12)  B-A {main[f'K{K}_B_minus_A']['effect']:+.4f} ({main[f'K{K}_B_minus_A']['direction_positive']}/12)")
        main['MDE_K4_CA'] = main['K4_C_minus_A']['MDE']
        (R / f'y_{dom}_summary.json').write_text(json.dumps({'main': main, 'note': 'appendix Y'}, ensure_ascii=False, indent=2), encoding='utf-8')
        arms[dom] = {'A': A, 'B': B, 'C': np.array(main['K4_arms']['C']['per_seed']), 'main': main}
        # stratification and residual (K4_r0)
        def sc(hi, lo):
            d = {k: [] for k in ('S1', 'S4', 'E', 'I', 'N')}
            for s in SEEDS:
                rh, rl = ST[dom][(hi, s, KEY)]['rows'], ST[dom][(lo, s, KEY)]['rows']
                d['S1'].append(rec(rh, S1) - rec(rl, S1)); d['S4'].append(rec(rh, S4) - rec(rl, S4)); d['E'].append(rec(rh, BIG_EDGE) - rec(rl, BIG_EDGE)); d['I'].append(rec(rh, BIG_INT) - rec(rl, BIG_INT)); d['N'].append(rec(rh, [NEAR]) - rec(rl, [NEAR]))
            d = {k: np.array(v) for k, v in d.items()}
            return {'dR_S1': float(d['S1'].mean()), 'dR_S4': float(d['S4'].mean()), 'count_S4_lt_S1': int((d['S4'] < d['S1']).sum()), 'dR_big_edge': float(d['E'].mean()), 'dR_big_int': float(d['I'].mean()), 'count_int_lt_edge': int((d['I'] < d['E']).sum()), 'dFPR_near': float(d['N'].mean()), 'count_near_gt0': int((d['N'] > 0).sum())}
        arms[dom]['strata'] = {'C_minus_A': sc('Cconfidence', 'Aplain'), 'B_minus_A': sc('Bplain', 'Aplain')}
        arms[dom]['dgm'] = float(np.mean([np.median([r['dgm_rel'] for r in ST[dom][('Cconfidence', s, KEY)]['rows']]) for s in SEEDS]))
        # oracle
        T = TH[dom]
        arms[dom]['oracle'] = {SHORT[t]: np.array([T[(t, s)]['iou_oracle_wide'] for s in SEEDS]) for t in TAGS}
        arms[dom]['frozen_thr'] = {SHORT[t]: np.array([T[(t, s)]['frozen_threshold'] for s in SEEDS]) for t in TAGS}
        arms[dom]['oracle_thr'] = {SHORT[t]: np.array([T[(t, s)]['oracle_threshold'] for s in SEEDS]) for t in TAGS}
        arms[dom]['boundary'] = {SHORT[t]: int(sum(T[(t, s)]['oracle_at_boundary'] for s in SEEDS)) for t in TAGS}
        arms[dom]['curves'] = {SHORT[t]: np.mean([[T[(t, s)]['iou_by_threshold'][str(g)] for g in GRID] for s in SEEDS], 0) for t in TAGS}

    print('\n' + '=' * 78); print('[Y-H1] measurability 2x2: C-A (K=4) over {WHU, Mass} x {frozen, oracle}'); print('=' * 78)
    D = {}
    for dom in ('whu', 'mass'):
        D[f'{dom}_frozen'] = arms[dom]['main']['K4_C_minus_A']
        D[f'{dom}_oracle'] = paired(arms[dom]['oracle']['C'] - arms[dom]['oracle']['A'])
    V = {k: verdict(v) for k, v in D.items()}
    for k in ('whu_frozen', 'whu_oracle', 'mass_frozen', 'mass_oracle'):
        v = D[k]; print(f"  {k:12s} effect {v['effect']:+.4f}  CI [{v['ci95_descriptive'][0]:+.4f}, {v['ci95_descriptive'][1]:+.4f}]  MDE {v['MDE']:.4f}  positive {v['direction_positive']}/12 -> {V[k]}")
    pairs = [('whu_frozen', 'mass_frozen'), ('whu_oracle', 'mass_oracle'), ('whu_frozen', 'whu_oracle'), ('mass_frozen', 'mass_oracle')]
    diff = [(p, q) for p, q in pairs if V[p] != V[q]]
    out['H']['Y-H1'] = {'D': D, 'verdicts': V, 'differing_pairs': diff, 'pass': bool(diff) if live else None,
                        'WHU_source_reference': {'inria_frozen': 'unresolved', 'mass_frozen': 'pos', 'mass_oracle': 'unresolved'}}
    print(f"  adjacent cells with different verdicts: {diff if diff else 'none'} -> {'holds' if diff else 'R6 not observed on the Inria source'} (WHU source: Inria frozen unresolved; Mass frozen pos, oracle unresolved)")

    print('\n' + '=' * 78); print('[Y-H2] variance amplification: cross-domain SD / in-domain SD >= 3 (three arms, two targets)'); print('=' * 78)
    amp = {}
    for dom in ('whu', 'mass'):
        amp[dom] = {a: float(arms[dom][a].std(ddof=1) / IN[t].std(ddof=1)) for a, t in (('A', 'Aplain'), ('B', 'Bplain'), ('C', 'Cconfidence'))}
        print(f"  {dom}: A {amp[dom]['A']:.1f}x B {amp[dom]['B']:.1f}x C {amp[dom]['C']:.1f}x; SD_B/SD_A {arms[dom]['B'].std(ddof=1)/arms[dom]['A'].std(ddof=1):.2f} SD_C/SD_A {arms[dom]['C'].std(ddof=1)/arms[dom]['A'].std(ddof=1):.2f}")
    out['H']['Y-H2'] = {'amp': amp, 'pass': all(v >= 3 for d in amp.values() for v in d.values()) if live else None}

    print('\n' + '=' * 78); print('[Y-H3] baseline operating-point drift: A oracle<frozen >= 9/12 on both domains; share Mass > WHU >= 9/12'); print('=' * 78)
    below = {dom: int((arms[dom]['oracle_thr']['A'] < arms[dom]['frozen_thr']['A']).sum()) for dom in ('whu', 'mass')}
    share = {dom: 1 - arms[dom]['A'] / arms[dom]['oracle']['A'] for dom in ('whu', 'mass')}
    b = int((share['mass'] > share['whu']).sum())
    print(f"  A oracle<frozen: WHU {below['whu']}/12, Mass {below['mass']}/12; share WHU {share['whu'].mean():.3f} Mass {share['mass'].mean():.3f}, Mass>WHU {b}/12; oracle median WHU {np.median(arms['whu']['oracle_thr']['A']):.2f} Mass {np.median(arms['mass']['oracle_thr']['A']):.2f}; at boundary {arms['whu']['boundary']['A']}/{arms['mass']['boundary']['A']}")
    for dom in ('whu', 'mass'):
        print(f"  {dom} share per arm: A {np.mean(1 - arms[dom]['A']/arms[dom]['oracle']['A']):.3f} B {np.mean(1 - arms[dom]['B']/arms[dom]['oracle']['B']):.3f} C {np.mean(1 - arms[dom]['C']/arms[dom]['oracle']['C']):.3f} (C: K4 R=5 mean vs r0 oracle, approximate)")
    out['H']['Y-H3'] = {'oracle_below_frozen': below, 'share_mean': {d: float(share[d].mean()) for d in share}, 'share_mass_gt_whu': b, 'pass': (below['whu'] >= 9 and below['mass'] >= 9 and b >= 9) if live else None}

    print('\n' + '=' * 78); print('[Y-H4] seed ranking not preserved across targets: Spearman(WHU, Mass) < 0.5 (three arms)'); print('=' * 78)
    rho = {a: float(stats.spearmanr(arms['whu'][a], arms['mass'][a]).correlation) for a in 'ABC'}
    print('  ' + '  '.join(f"{a} {rho[a]:+.3f}" for a in 'ABC'))
    out['H']['Y-H4'] = {'rho': rho, 'pass': all(v < 0.5 for v in rho.values()) if live else None}

    print('\n' + '=' * 78); print('[Y-H5] calibration-transfer reading (targets where C-A is resolvably positive at the frozen threshold)'); print('=' * 78)
    h5 = {}
    for dom in ('whu', 'mass'):
        if V[f'{dom}_frozen'] == 'pos':
            ca_o = D[f'{dom}_oracle']; sh = 1 - ca_o['effect'] / D[f'{dom}_frozen']['effect']
            h5[dom] = {'CA_oracle': ca_o['effect'], 'share': sh, 'pass': bool(ca_o['effect'] < D[f'{dom}_frozen']['MDE'] and sh >= 0.5)}
            print(f"  {dom}: C-A at oracle {ca_o['effect']:+.4f} (MDE {D[f'{dom}_frozen']['MDE']:.4f}), operating-point share {sh:.3f} -> {'still calibration transfer' if h5[dom]['pass'] else 'gain partly independent of the operating point'}")
        else:
            h5[dom] = {'applies': False}; print(f"  {dom}: not resolvably positive at the frozen threshold; not applicable")
    out['H']['Y-H5'] = h5

    print('\n' + '=' * 78); print('[Y-E1] stratification (K4_r0) and DGM residual; [Y-E2] 0.40-0.60 range and C-A sign flips'); print('=' * 78)
    for dom in ('whu', 'mass'):
        st = arms[dom]['strata']
        print(f"  {dom} C-A: S4<S1 {st['C_minus_A']['count_S4_lt_S1']}/12 (dR S1 {st['C_minus_A']['dR_S1']:+.3f} S4 {st['C_minus_A']['dR_S4']:+.3f}), interior<edge {st['C_minus_A']['count_int_lt_edge']}/12, near-band dFPR {st['C_minus_A']['dFPR_near']:+.3f} ({st['C_minus_A']['count_near_gt0']}/12); B-A S4<S1 {st['B_minus_A']['count_S4_lt_S1']}/12; DGM {arms[dom]['dgm']:.3f}")
        rng_ = np.array([max(v for k, v in TH[dom][('Cconfidence', s)]['iou_by_threshold'].items() if 0.40 <= float(k) <= 0.60) - min(v for k, v in TH[dom][('Cconfidence', s)]['iou_by_threshold'].items() if 0.40 <= float(k) <= 0.60) for s in SEEDS])
        d = arms[dom]['curves']['C'] - arms[dom]['curves']['A']; flips = [GRID[i] for i in range(1, len(GRID)) if np.sign(d[i]) != np.sign(d[i - 1])]
        print(f"  {dom} arm C 0.40-0.60 range {rng_.mean():.4f}; mean C-A curve at t=0.1/0.3/0.5/0.7 = " + ' '.join(f"{d[GRID.index(t)]:+.4f}" for t in (0.1, 0.3, 0.5, 0.7)) + f"; sign flips {flips if flips else 'none'}")
        out['targets'][dom] = {'strata': st, 'dgm': arms[dom]['dgm'], 'range_0.40-0.60_C': rng_.tolist(), 'curve_D': d.tolist(), 'flips': flips,
                               'oracle_means': {a: float(arms[dom]['oracle'][a].mean()) for a in 'ABC'}, 'boundary': arms[dom]['boundary']}
    out['live'] = live
    print('\n' + '=' * 78); print('[verdicts]'); print('=' * 78)
    for k in ('Y-H1', 'Y-H2', 'Y-H3', 'Y-H4'):
        v = out['H'][k]['pass']; print(f"  {k} -> {'holds' if v else ('does not hold' if v is not None else 'stop-loss, not judged')}")
    (R / 'y_summary.json').write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding='utf-8'); print('\n  -> results/y_summary.json')


if __name__ == '__main__':
    main()
