# -*- coding: utf-8 -*-
"""
Appendix W analysis: whether R1/R2/R4/R6 hold on the second test bed (deeplabv3_r50 / segformer_b1 x 12 seeds).
**Committed before any W number existed.** No significance claims; "holds / does not hold" follows the count rules of W.4;
CIs are descriptive; the MDE uses the same algorithm as mass_analyze.
Inputs: results/w (L1-style), results/w_train (copies of the training results.json), results/w_l2 (Inria f=1/2/4/8 stratified),
        results/w_mass, results/thrdiag_w, results/thrdiag_w_mass, results/w_calib (W.3.2), results/w_ckpt_sha256.txt (W.3.1).
GAPL reference: results/stage1 (arm A, Inria), results/mass (arm A, Mass), results/stage1_train (arm A, in-domain),
        stratum totals of results/rq5 / results/mass (W.3.3).
"""
import argparse, glob, json, sys, pathlib
import numpy as np
from scipy import stats
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
SEEDS = list(range(100, 112)); ARCHS = ('deeplabv3_r50', 'segformer_b1'); SHORT = {'deeplabv3_r50': 'X1', 'segformer_b1': 'X2'}
KEY = 'fmin0.01_K4_r0'; E1 = 0.0005; STOP = 0.80
S1, S4, BIG_EDGE, BIG_INT = [0, 1], [6, 7], [4, 6], [5, 7]


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
    d = np.asarray(d, float); n = len(d); m = float(d.mean()); sd = float(d.std(ddof=1))
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return {'n': n, 'effect': m, 'sd_d': sd, 'MDE': mde(sd, n), 'ci95_descriptive': [m - half, m + half],
            'direction_positive': int((d > 0).sum()), 'per_seed': d.tolist()}


def verdict(pr):
    """Resolvable = direction >= 10/12 (or <= 2/12) and |effect| >= MDE; returns 'pos' / 'neg' / 'unresolved'."""
    if abs(pr['effect']) >= pr['MDE'] and (pr['direction_positive'] >= 10 or pr['direction_positive'] <= 2):
        return 'pos' if pr['effect'] > 0 else 'neg'
    return 'unresolved'


def recall_rows(rows, idx):
    n = np.array([r['n'] for r in rows]); p1 = np.array([r['p1'] for r in rows])
    return p1[:, idx].sum() / max(n[:, idx].sum(), 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='results/w_summary.json'); a = ap.parse_args()
    R = ROOT / 'results'
    out = {'seeds': SEEDS, 'archs': list(ARCHS), 'note': 'appendix W; descriptive; no significance claims', 'gates': {}, 'stop_loss': {}, 'H': {}}
    W = {ar: {s: L(R / 'w' / f'{ar}_seed{s}_{KEY}.json') for s in SEEDS} for ar in ARCHS}
    TR = {ar: {s: L(R / 'w_train' / f'{ar}_seed{s}.results.json') for s in SEEDS} for ar in ARCHS}
    L2 = {ar: {(s, f): L(R / 'w_l2' / f'{ar}_seed{s}_gsd{f}.json') for s in SEEDS for f in (1, 2, 4, 8)} for ar in ARCHS}
    MS = {ar: {s: L(R / 'w_mass' / f'{ar}_seed{s}_{KEY}.json') for s in SEEDS} for ar in ARCHS}
    TI = {ar: {s: L(R / 'thrdiag_w' / f'{ar}_seed{s}_gsd1.json') for s in SEEDS} for ar in ARCHS}
    TM = {ar: {s: L(R / 'thrdiag_w_mass' / f'{ar}_seed{s}.json') for s in SEEDS} for ar in ARCHS}
    GA_I = {s: L(R / 'stage1' / f'Aplain_seed{s}_{KEY}.json')['target']['iou'] for s in SEEDS}
    GA_M = {s: L(R / 'mass' / f'Aplain_seed{s}_{KEY}.json')['overall']['iou'] for s in SEEDS}
    GA_IN = {s: L(R / 'stage1_train' / f'Aplain_seed{s}.results.json')['test_metrics']['iou'] for s in SEEDS}

    print('=' * 78); print('[gate W.3.1] provenance: checkpoint sha256 in results = training-machine digest list; frozen threshold = L1; arch key'); print('=' * 78)
    sha = {}
    for line in (R / 'w_ckpt_sha256.txt').read_text(encoding='utf-8').splitlines():
        if line.strip():
            h, p = line.split(maxsplit=1); sha[pathlib.Path(p.strip().lstrip('*')).name] = h
    bad = []
    for ar in ARCHS:
        for s in SEEDS:
            ref = sha.get(f'{ar}_seed{s}.pth'); ft = W[ar][s]['source_val_threshold']
            for d, tag in [(W[ar][s], 'w'), (MS[ar][s], 'mass')] + [(L2[ar][(s, f)], f'gsd{f}') for f in (1, 2, 4, 8)]:
                if d['checkpoint_sha256'] != ref: bad.append((ar, s, tag, 'sha'))
                if d.get('arch') != ar: bad.append((ar, s, tag, 'arch'))
                if 'threshold' in d and abs(d['threshold'] - ft) > 1e-9: bad.append((ar, s, tag, 'thr'))
            for d in (TI[ar][s], TM[ar][s]):
                if abs(d['frozen_threshold'] - ft) > 1e-9: bad.append((ar, s, 'thrdiag', 'thr'))
    if bad: print('  mismatch:', bad[:10]); sys.exit(2)
    print('  all passed'); out['gates']['provenance'] = 'pass'

    print('\n' + '=' * 78); print('[gate W.3.2] cross-machine calibration: X1 seed100 Inria f=1 (frozen threshold), dyt vs 4090d, difference < 0.0005'); print('=' * 78)
    calib = sorted(glob.glob(str(R / 'w_calib' / '*.json')))
    worst = 0.0
    for c in calib:
        d = L(c); ar, s = d['arch'], d['train_args']['seed']
        ref = W[ar][s]['target']['iou'] if 'target' in d else MS[ar][s]['overall']['iou']
        v = d['target']['iou'] if 'target' in d else d['overall']['iou']
        print(f"  {pathlib.Path(c).name}: {v:.6f} vs main evaluation machine {ref:.6f}  d = {v-ref:+.8f}"); worst = max(worst, abs(v - ref))
    if not calib: print('  (no calibration file -- must be disclosed)')
    out['gates']['calib_max_abs'] = worst; out['gates']['calib_pass'] = bool(calib) and worst < E1
    print(f"  => {'pass' if out['gates']['calib_pass'] else 'fail / missing'} (max|d| = {worst:.2e})")

    print('\n' + '=' * 78); print('[gate W.3.3] label consistency: stratum totals of every view bit-identical to the GAPL evaluation of the same view'); print('=' * 78)
    ok = True
    for f in (1, 2, 4, 8):
        ref = L(R / 'rq5' / f'Aplain_seed100_gsd{f}.json')['strata_n_total']
        for ar in ARCHS:
            for s in SEEDS:
                if L2[ar][(s, f)]['strata_n_total'] != ref: ok = False; print(f'  gsd{f} {ar} seed{s} stratum totals differ')
    refm = L(R / 'mass' / f'Aplain_seed100_{KEY}.json')
    for ar in ARCHS:
        for s in SEEDS:
            if MS[ar][s]['strata_n_total'] != refm['strata_n_total'] or MS[ar][s]['strata_n_total_alt'] != refm['strata_n_total_alt']:
                ok = False; print(f'  mass {ar} seed{s} stratum totals differ')
    if not ok: print('  => stop'); sys.exit(2)
    print('  pass'); out['gates']['labels'] = 'pass'

    print('\n' + '=' * 78); print('[stop-loss W.4] in-domain test IoU (0.5) mean >= 0.80 (GAPL A 0.890)'); print('=' * 78)
    live = []
    for ar in ARCHS:
        v = np.array([TR[ar][s]['test_metrics']['iou'] for s in SEEDS]); m = float(v.mean())
        out['stop_loss'][ar] = {'indomain_mean': m, 'indomain_sd': float(v.std(ddof=1)), 'triggered': m < STOP, 'per_seed': v.tolist(),
                                'best_epoch': [TR[ar][s]['best_epoch'] for s in SEEDS], 'elapsed_min': float(np.mean([TR[ar][s]['elapsed_seconds'] for s in SEEDS]) / 60)}
        print(f"  {SHORT[ar]} {ar}: in-domain {m:.4f} (SD {v.std(ddof=1):.4f}) [{v.min():.3f}, {v.max():.3f}]  median best epoch {np.median(out['stop_loss'][ar]['best_epoch']):.0f}  -> {'stop-loss triggered: descriptive only' if m < STOP else 'not triggered'}")
        if m >= STOP: live.append(ar)
    print(f"  GAPL A in-domain {np.mean(list(GA_IN.values())):.4f} (SD {np.std(list(GA_IN.values()), ddof=1):.4f})")

    # per-arm, per-domain, per-seed values
    I = {ar: np.array([W[ar][s]['target']['iou'] for s in SEEDS]) for ar in ARCHS}
    Mm = {ar: np.array([MS[ar][s]['overall']['iou'] for s in SEEDS]) for ar in ARCHS}
    IN = {ar: np.array([TR[ar][s]['test_metrics']['iou'] for s in SEEDS]) for ar in ARCHS}
    IO = {ar: np.array([TI[ar][s]['iou_oracle_wide'] for s in SEEDS]) for ar in ARCHS}
    MO = {ar: np.array([TM[ar][s]['iou_oracle_wide'] for s in SEEDS]) for ar in ARCHS}
    out['per_arch'] = {ar: {'indomain': IN[ar].tolist(), 'inria_frozen': I[ar].tolist(), 'mass_frozen': Mm[ar].tolist(),
                            'inria_oracle': IO[ar].tolist(), 'mass_oracle': MO[ar].tolist(),
                            'frozen_threshold': [W[ar][s]['source_val_threshold'] for s in SEEDS],
                            'inria_oracle_threshold': [TI[ar][s]['oracle_threshold'] for s in SEEDS],
                            'mass_oracle_threshold': [TM[ar][s]['oracle_threshold'] for s in SEEDS]} for ar in ARCHS}
    print('\n' + '=' * 78); print('[arms] IoU mean (seed-SD) at the frozen threshold: in-domain / Inria f=1 / Massachusetts; oracle means'); print('=' * 78)
    for ar in ARCHS:
        print(f"  {SHORT[ar]} {ar:14s} in-domain {IN[ar].mean():.4f} ({IN[ar].std(ddof=1):.4f})  Inria {I[ar].mean():.4f} ({I[ar].std(ddof=1):.4f})  Mass {Mm[ar].mean():.4f} ({Mm[ar].std(ddof=1):.4f})  | oracle Inria {IO[ar].mean():.4f} Mass {MO[ar].mean():.4f}")
    gi = np.array([GA_I[s] for s in SEEDS]); gm = np.array([GA_M[s] for s in SEEDS]); gin = np.array([GA_IN[s] for s in SEEDS])
    print(f"  GAPL A (reference)   in-domain {gin.mean():.4f} ({gin.std(ddof=1):.4f})  Inria {gi.mean():.4f} ({gi.std(ddof=1):.4f})  Mass {gm.mean():.4f} ({gm.std(ddof=1):.4f})")

    print('\n' + '=' * 78); print('[W-H1] variance amplification: cross-domain seed-SD / in-domain seed-SD >= 3 (both Inria and Mass)'); print('=' * 78)
    for ar in ARCHS:
        ri = I[ar].std(ddof=1) / IN[ar].std(ddof=1); rm = Mm[ar].std(ddof=1) / IN[ar].std(ddof=1)
        h = bool(ri >= 3 and rm >= 3) if ar in live else None
        out['H'][f'W-H1_{ar}'] = {'ratio_inria': float(ri), 'ratio_mass': float(rm), 'pass': h}
        print(f"  {SHORT[ar]}: Inria {ri:.1f}x  Mass {rm:.1f}x  (GAPL A 4.2x / 9.6x) -> {'holds' if h else ('does not hold' if h is not None else 'stop-loss, not judged')}")

    print('\n' + '=' * 78); print('[W-H2] baseline operating-point drift: (a) oracle < frozen >= 9/12 on both domains; (b) share Mass > Inria >= 9/12'); print('=' * 78)
    for ar in ARCHS:
        ft = np.array(out['per_arch'][ar]['frozen_threshold']); ti = np.array(out['per_arch'][ar]['inria_oracle_threshold']); tm = np.array(out['per_arch'][ar]['mass_oracle_threshold'])
        a_i = int((ti < ft).sum()); a_m = int((tm < ft).sum())
        sh_i = 1 - I[ar] / IO[ar]; sh_m = 1 - Mm[ar] / MO[ar]; b = int((sh_m > sh_i).sum())
        nb_i = sum(int(TI[ar][s]['oracle_at_boundary']) for s in SEEDS); nb_m = sum(int(TM[ar][s]['oracle_at_boundary']) for s in SEEDS)
        h = bool(a_i >= 9 and a_m >= 9 and b >= 9) if ar in live else None
        out['H'][f'W-H2_{ar}'] = {'oracle_below_frozen_inria': a_i, 'oracle_below_frozen_mass': a_m, 'share_inria_mean': float(sh_i.mean()), 'share_mass_mean': float(sh_m.mean()),
                                  'share_mass_gt_inria': b, 'at_boundary_inria': nb_i, 'at_boundary_mass': nb_m, 'pass': h,
                                  'frozen_median': float(np.median(ft)), 'oracle_median_inria': float(np.median(ti)), 'oracle_median_mass': float(np.median(tm))}
        print(f"  {SHORT[ar]}: oracle<frozen Inria {a_i}/12, Mass {a_m}/12; share Inria {sh_i.mean():.3f} Mass {sh_m.mean():.3f}, Mass>Inria {b}/12; frozen median {np.median(ft):.2f}, oracle median Inria {np.median(ti):.2f} Mass {np.median(tm):.2f}; at boundary {nb_i}/{nb_m}  (GAPL A: 12/12, 12/12; 0.037 vs 0.170) -> {'holds' if h else ('does not hold' if h is not None else 'stop-loss, not judged')}")

    print('\n' + '=' * 78); print('[W-H3] seed ranking not preserved across target domains: Spearman(Inria, Mass) < 0.5'); print('=' * 78)
    for ar in ARCHS:
        rho = float(stats.spearmanr(I[ar], Mm[ar]).correlation)
        r_gi = float(stats.spearmanr(I[ar], gi).correlation); r_gm = float(stats.spearmanr(Mm[ar], gm).correlation); r_in = float(stats.spearmanr(IN[ar], I[ar]).correlation)
        h = bool(rho < 0.5) if ar in live else None
        out['H'][f'W-H3_{ar}'] = {'rho_inria_mass': rho, 'rho_with_gaplA_inria': r_gi, 'rho_with_gaplA_mass': r_gm, 'rho_indomain_inria': r_in, 'pass': h}
        print(f"  {SHORT[ar]}: rho(Inria, Mass) = {rho:+.3f} (GAPL A +0.10); same seeds vs GAPL A: Inria {r_gi:+.3f} Mass {r_gm:+.3f}; in-domain vs Inria {r_in:+.3f} -> {'holds' if h else ('does not hold' if h is not None else 'stop-loss, not judged')}")
    rho_g = float(stats.spearmanr(gi, gm).correlation); out['H']['gaplA_rho_inria_mass'] = rho_g

    print('\n' + '=' * 78); print('[W-H4] measurability set jointly by target domain and operating point: the 2x2 (domain x operating point) of D = X2 - X1'); print('=' * 78)
    D = {'inria_frozen': paired(I['segformer_b1'] - I['deeplabv3_r50']), 'mass_frozen': paired(Mm['segformer_b1'] - Mm['deeplabv3_r50']),
         'inria_oracle': paired(IO['segformer_b1'] - IO['deeplabv3_r50']), 'mass_oracle': paired(MO['segformer_b1'] - MO['deeplabv3_r50'])}
    V = {k: verdict(v) for k, v in D.items()}
    for k in ('inria_frozen', 'inria_oracle', 'mass_frozen', 'mass_oracle'):
        v = D[k]; print(f"  {k:13s} effect {v['effect']:+.4f}  CI [{v['ci95_descriptive'][0]:+.4f}, {v['ci95_descriptive'][1]:+.4f}]  s_d {v['sd_d']:.4f}  MDE {v['MDE']:.4f}  positive {v['direction_positive']}/12  -> {V[k]}")
    pairs = [('inria_frozen', 'mass_frozen'), ('inria_oracle', 'mass_oracle'), ('inria_frozen', 'inria_oracle'), ('mass_frozen', 'mass_oracle')]
    diff = [(p, q) for p, q in pairs if V[p] != V[q]]
    h4 = bool(diff) if len(live) == 2 else None
    out['H']['W-H4'] = {'D': D, 'verdicts': V, 'differing_pairs': diff, 'pass': h4}
    print(f"  adjacent cells with different verdicts: {diff if diff else 'none'} -> {'holds (weak criterion: one differing pair suffices)' if h4 else ('R6 not observed on this bed' if h4 is not None else 'stop-loss, not judged')}")

    print('\n' + '=' * 78); print('[W-E1] synthetic blur: monotone-decrease count of IoU(f); distribution of the f=2 vs f=1 recall loss (1 px band = default 4 px)'); print('=' * 78)
    for ar in ARCHS:
        io = {f: np.array([L2[ar][(s, f)]['overall']['iou'] for s in SEEDS]) for f in (1, 2, 4, 8)}
        mono = int(sum((io[1][i] > io[2][i] > io[4][i] > io[8][i]) for i in range(12)))
        dS1 = []; dS4 = []; dE = []; dI = []
        for s in SEEDS:
            r1, r2 = L2[ar][(s, 1)]['rows'], L2[ar][(s, 2)]['rows']
            dS1.append(recall_rows(r2, S1) - recall_rows(r1, S1)); dS4.append(recall_rows(r2, S4) - recall_rows(r1, S4))
            dE.append(recall_rows(r2, BIG_EDGE) - recall_rows(r1, BIG_EDGE)); dI.append(recall_rows(r2, BIG_INT) - recall_rows(r1, BIG_INT))
        dS1, dS4, dE, dI = map(np.array, (dS1, dS4, dE, dI))
        out['H'][f'W-E1_{ar}'] = {'iou_by_f': {str(f): io[f].tolist() for f in io}, 'monotone_seeds': mono, 'dR_S1': dS1.tolist(), 'dR_S4': dS4.tolist(), 'dR_big_edge': dE.tolist(), 'dR_big_int': dI.tolist(),
                                  'count_S4_loss_gt_S1': int((dS4 < dS1).sum()), 'count_int_loss_gt_edge': int((dI < dE).sum())}
        print(f"  {SHORT[ar]}: IoU f1/2/4/8 = {io[1].mean():.4f}/{io[2].mean():.4f}/{io[4].mean():.4f}/{io[8].mean():.4f}, monotone {mono}/12; f2-f1 dR S1 {dS1.mean():+.3f} S4 {dS4.mean():+.3f} [larger loss in S4 {(dS4 < dS1).sum()}/12]; large buildings edge {dE.mean():+.3f} interior {dI.mean():+.3f} [larger loss in interior {(dI < dE).sum()}/12]")

    print('\n' + '=' * 78); print('[W-E2] seed-paired difference X - A against the GAPL arm A (same 1000-image subsets)'); print('=' * 78)
    for ar in ARCHS:
        pi = paired(I[ar] - gi); pm = paired(Mm[ar] - gm); pin = paired(IN[ar] - gin)
        out['H'][f'W-E2_{ar}'] = {'indomain': pin, 'inria': pi, 'mass': pm}
        print(f"  {SHORT[ar]} - A: in-domain {pin['effect']:+.4f} ({pin['direction_positive']}/12)  Inria {pi['effect']:+.4f} [{pi['ci95_descriptive'][0]:+.4f}, {pi['ci95_descriptive'][1]:+.4f}] ({pi['direction_positive']}/12)  Mass {pm['effect']:+.4f} [{pm['ci95_descriptive'][0]:+.4f}, {pm['ci95_descriptive'][1]:+.4f}] ({pm['direction_positive']}/12)")

    print('\n' + '=' * 78); print('[W-E3] threshold sensitivity (0.40-0.60 range) and the sign of D along the threshold (0.05-0.95 mean curves)'); print('=' * 78)
    grid = [round(0.05 + 0.01 * i, 2) for i in range(91)]
    for ar in ARCHS:
        rng_i = np.array([max(W[ar][s]['target_threshold_sensitivity'].values()) - min(W[ar][s]['target_threshold_sensitivity'].values()) for s in SEEDS])
        rng_m = np.array([max(v for k, v in TM[ar][s]['iou_by_threshold'].items() if 0.40 <= float(k) <= 0.60) - min(v for k, v in TM[ar][s]['iou_by_threshold'].items() if 0.40 <= float(k) <= 0.60) for s in SEEDS])
        out['H'][f'W-E3_{ar}'] = {'range_0.40-0.60_inria': rng_i.tolist(), 'range_0.40-0.60_mass': rng_m.tolist()}
        print(f"  {SHORT[ar]}: 0.40-0.60 range Inria {rng_i.mean():.4f} (max {rng_i.max():.4f}) Mass {rng_m.mean():.4f} (max {rng_m.max():.4f})")
    curves = {}
    for dom, T in (('inria', TI), ('mass', TM)):
        c = {ar: np.mean([[T[ar][s]['iou_by_threshold'][str(g)] for g in grid] for s in SEEDS], 0) for ar in ARCHS}
        d = c['segformer_b1'] - c['deeplabv3_r50']; curves[dom] = {'grid': grid, 'X1': c['deeplabv3_r50'].tolist(), 'X2': c['segformer_b1'].tolist(), 'D': d.tolist()}
        flips = [grid[i] for i in range(1, len(grid)) if np.sign(d[i]) != np.sign(d[i - 1])]
        print(f"  {dom}: mean D curve at t=0.1/0.3/0.5/0.7/0.9 = " + ' '.join(f"{d[grid.index(t)]:+.4f}" for t in (0.1, 0.3, 0.5, 0.7, 0.9)) + f"; sign flips {flips if flips else 'none'}")
    out['H']['W-E3_curves'] = curves

    print('\n' + '=' * 78); print('[verdicts]'); print('=' * 78)
    for k, v in out['H'].items():
        if isinstance(v, dict) and 'pass' in v and k.startswith('W-H'):
            print(f"  {k:22s} -> {'holds' if v['pass'] else ('does not hold' if v['pass'] is not None else 'stop-loss, not judged')}")
    (ROOT / a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8'); print(f'\n  -> {a.out}')


if __name__ == '__main__':
    main()
