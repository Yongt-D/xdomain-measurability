# -*- coding: utf-8 -*-
"""
Appendix U analysis: gates U.3 -> stop-loss U.4 -> hypotheses U-H1..H5 and exploratory U-E1/E2. **Committed before any of its numbers existed.**
No significance claims; "supported" = the pre-registered >= 9/12-seed count rule; CIs are descriptive; the MDE uses the same algorithm as mde_at_n.
"""
import argparse, json, sys, pathlib, collections
import numpy as np
from scipy import stats
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
SEEDS = list(range(100, 112))
TAGS = ('Aplain', 'Bplain', 'Cconfidence')
SHORT = {'Aplain': 'A', 'Bplain': 'B', 'Cconfidence': 'C'}
KS = (1, 4, 8); R = 5
E1 = 0.0005; STOP_LOSS = 0.15
S1 = [0, 1]; S4 = [6, 7]; BIG_EDGE = [4, 6]; BIG_INT = [5, 7]; NEG_NEAR = 8; NEG_FAR = 9


def L(p):
    p = pathlib.Path(p)
    if not p.exists():
        print(f'  missing {p}'); sys.exit(1)
    return json.loads(p.read_text(encoding='utf-8'))


def mde(s_d, n=12):
    lo, hi = 1e-5, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        ncp = mid / (s_d / np.sqrt(n)); crit = stats.t.ppf(0.975, n - 1)
        p = stats.nct.sf(crit, n - 1, ncp) + stats.nct.cdf(-crit, n - 1, ncp)
        if p >= 0.80: hi = mid
        else: lo = mid
    return float(hi)


def counts(rows, band):
    """Return the two 10-dimensional arrays (n, p1), taken from n/p1 or from alt[band] according to the band width."""
    if band is None:
        n = np.array([r['n'] for r in rows]); p1 = np.array([r['p1'] for r in rows])
    else:
        n = np.array([r['alt'][band]['n'] for r in rows]); p1 = np.array([r['alt'][band]['p1'] for r in rows])
    return n, p1


def recall(n, p1, strata):
    return p1[:, strata].sum() / max(n[:, strata].sum(), 1)


def fpr(n, p1, k):
    return p1[:, k].sum() / max(n[:, k].sum(), 1)


def paired(d):
    d = np.asarray(d); n = len(d); m = float(d.mean()); sd = float(d.std(ddof=1))
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return {'n': n, 'effect': m, 'sd_d': sd, 'ci95_descriptive': [m - half, m + half],
            'direction_positive': int((d > 0).sum()), 'per_seed': d.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='mass')
    ap.add_argument('--calib-dir', default='mass_calib')
    a = ap.parse_args()
    D = ROOT / 'results' / a.dir
    shas = {}
    for line in (ROOT / 'results' / 'stage1_ckpt_sha256.txt').read_text(encoding='utf-8').splitlines():
        h, name = line.split(); shas[name.strip('*')] = h
    man = L(ROOT / 'results' / 'support_manifests' / 'mass_support_manifests.json')
    keys_C = [f'fmin0.01_K{k}_r{r}' for k in KS for r in range(R)]
    R_ = {}
    for s in SEEDS:
        for t in TAGS:
            for key in (keys_C if t == 'Cconfidence' else ['fmin0.01_K4_r0']):
                R_[(t, s, key)] = L(D / f'{t}_seed{s}_{key}.json')
    out = {'seeds': SEEDS, 'note': 'appendix U; descriptive; no significance claims', 'gates': {}, 'H': {}}
    print(f'loaded {len(R_)} results')

    print('=' * 76); print('[gate U.3.1] provenance: sha256 / frozen threshold / support list'); print('=' * 76)
    bad = []
    for (t, s, key), d in R_.items():
        l1 = L(ROOT / 'results' / 'stage1' / f'{t}_seed{s}_fmin0.01_K4_r0.json')
        if d['checkpoint_sha256'] != shas[f'{t}_seed{s}.pth']: bad.append((t, s, key, 'sha256'))
        if abs(d['threshold'] - l1['source_val_threshold']) > 1e-9: bad.append((t, s, key, 'threshold'))
        if d['support_files'] != man[key]['files']: bad.append((t, s, key, 'support'))
        if d['size_bins'] != [23, 92, 369] or d.get('bands_px') != [1.0, 4.0]: bad.append((t, s, key, 'strata-def'))
    if bad:
        print('  mismatch:', bad[:10]); print('  => stop'); sys.exit(2)
    print('  all passed')

    print('\n' + '=' * 76); print(f'[gate U.3.2] cross-machine calibration: A seed100 K4_r0 image-level IoU on two machines differs by < {E1}'); print('=' * 76)
    hosts = collections.Counter(d['hostname'] for d in R_.values())
    print('  main evaluation hosts:', dict(hosts))
    cal_dir = ROOT / 'results' / a.calib_dir
    cals = sorted(cal_dir.glob('*.json')) if cal_dir.exists() else []
    main_ref = R_[('Aplain', 100, 'fmin0.01_K4_r0')]
    if not cals:
        print('  **no calibration file** (results/mass_calib/); may be waived only if the main evaluation ran on a single host, otherwise stop')
        if len(hosts) > 1: sys.exit(2)
        out['gates']['U332'] = 'single-host, no calib'
    else:
        worst = 0.0
        for c in cals:
            dc = L(c)
            dv = dc['overall']['iou'] - main_ref['overall']['iou']
            print(f"  {c.name}: {dc['hostname']} IoU {dc['overall']['iou']:.6f}  vs main ({main_ref['hostname']}) {main_ref['overall']['iou']:.6f}  d = {dv:+.8f}")
            worst = max(worst, abs(dv))
            if dc['strata_n_total'] != main_ref['strata_n_total']: print('  **stratum totals differ**'); sys.exit(2)
        out['gates']['U332'] = {'max_abs': worst}
        if worst >= E1: print('  => **FAIL**, stop'); sys.exit(2)
        print(f'  => pass (max|d| = {worst:.8f})')

    print('\n' + '=' * 76); print('[gate U.3.3] label consistency: per-stratum pixel totals bit-identical (1 px and 4 px bands)'); print('=' * 76)
    tot1 = {tuple(d['strata_n_total']) for d in R_.values()}
    tot4 = {tuple(d['strata_n_total_alt']['4.0']) for d in R_.values()}
    if len(tot1) != 1 or len(tot4) != 1:
        print(f'  mismatch: {len(tot1)} / {len(tot4)} variants; stop'); sys.exit(2)
    names = next(iter(R_.values()))['strata_names']
    for lab, tot in (('band 1 px', next(iter(tot1))), ('band 4 px', next(iter(tot4)))):
        posn = sum(tot[:8]); negn = sum(tot[8:])
        print(f'  pass. {lab}: ' + '  '.join(f'{names[k]} {tot[k] / (posn if k < 8 else negn) * 100:.1f}%' for k in range(10)))
    out['gates']['strata_n_total_b1'] = list(next(iter(tot1))); out['gates']['strata_n_total_b4'] = list(next(iter(tot4)))

    # ---------- stop-loss U.4 ----------
    def iou(t, s, key): return R_[(t, s, key)]['overall']['iou']
    a_vals = np.array([iou('Aplain', s, 'fmin0.01_K4_r0') for s in SEEDS])
    print('\n' + '=' * 76); print(f'[stop-loss U.4] arm A K4_r0 frozen-threshold IoU mean >= {STOP_LOSS}'); print('=' * 76)
    print(f'  A mean {a_vals.mean():.6f}  SD {a_vals.std(ddof=1):.6f}  [{a_vals.min():.4f}, {a_vals.max():.4f}]')
    deep = a_vals.mean() < STOP_LOSS
    out['stop_loss'] = {'A_mean': float(a_vals.mean()), 'triggered': bool(deep)}
    print('  => ' + ('**triggered**: this domain is reported descriptively as a deep-degradation point only; U.5 is not judged' if deep else 'not triggered'))

    # ---------- primary criterion (same reading as L1) ----------
    print('\n' + '=' * 76); print('[primary] frozen threshold; arm C averaged over R=5 draws; A/B structurally identical across K'); print('=' * 76)
    vals = {}
    for k in KS:
        vals[('Aplain', k)] = a_vals
        vals[('Bplain', k)] = np.array([iou('Bplain', s, 'fmin0.01_K4_r0') for s in SEEDS])
        vals[('Cconfidence', k)] = np.array([np.mean([iou('Cconfidence', s, f'fmin0.01_K{k}_r{r}') for r in range(R)]) for s in SEEDS])
    out['main'] = {}
    for k in KS:
        print(f'--- K = {k} ---')
        for t in TAGS:
            v = vals[(t, k)]; print(f'  {SHORT[t]}  mean {v.mean():.6f}  seed-SD {v.std(ddof=1):.6f}  [{v.min():.4f}, {v.max():.4f}]')
        for lhs, rhs in (('Cconfidence', 'Aplain'), ('Cconfidence', 'Bplain'), ('Bplain', 'Aplain')):
            pr = paired(vals[(lhs, k)] - vals[(rhs, k)])
            print(f"  {SHORT[lhs]}-{SHORT[rhs]}: effect {pr['effect']:+.6f}  CI [{pr['ci95_descriptive'][0]:+.6f}, {pr['ci95_descriptive'][1]:+.6f}]  s_d {pr['sd_d']:.6f}  positive {pr['direction_positive']}/12")
            out['main'][f'K{k}_{SHORT[lhs]}_minus_{SHORT[rhs]}'] = pr
        out['main'][f'K{k}_arms'] = {SHORT[t]: {'mean': float(vals[(t, k)].mean()), 'sd': float(vals[(t, k)].std(ddof=1)), 'per_seed': vals[(t, k)].tolist()} for t in TAGS}
    ca = out['main']['K4_C_minus_A']; m4 = mde(ca['sd_d'])
    out['main']['MDE_K4_CA'] = m4
    print(f'  this domain, K=4 C-A: s_d = {ca["sd_d"]:.6f} -> MDE(n=12) = {m4:.4f}')

    # ---------- RQ1-type: variance amplification ----------
    indom = L(ROOT / 'results' / 'stage1_train_indomain_summary.json')['arms']
    print('\n' + '=' * 76); print('[U-H2] variance amplification: cross-domain seed-SD / in-domain seed-SD (K=4)'); print('=' * 76)
    amp = {}
    for t in TAGS:
        sd_x = float(vals[(t, 4)].std(ddof=1)); sd_in = float(indom[t]['sd'])
        amp[SHORT[t]] = {'sd_cross': sd_x, 'sd_indomain': sd_in, 'ratio': sd_x / sd_in}
        print(f'  {SHORT[t]}: cross-domain {sd_x:.6f} / in-domain {sd_in:.6f} = {sd_x / sd_in:.2f}x')
    rBA = amp['B']['sd_cross'] / amp['A']['sd_cross']; rCA = amp['C']['sd_cross'] / amp['A']['sd_cross']
    print(f'  SD_B/SD_A = {rBA:.2f}   SD_C/SD_A = {rCA:.2f}')
    out['H']['U-H2'] = {'amp': amp, 'SD_B_over_A': rBA, 'SD_C_over_A': rCA}

    # ---------- RQ5-type: stratification (K4_r0) ----------
    print('\n' + '=' * 76); print('[U-H3/H4/H5, U-E1] stratification (K4_r0; two band widths)'); print('=' * 76)
    strat = {}
    for band, lab in ((None, '1px'), ('4.0', '4px')):
        for t in ('Cconfidence', 'Bplain'):
            dS1, dS4, dE, dI, dFn, dFf = [], [], [], [], [], []
            for s in SEEDS:
                nt, pt = counts(R_[(t, s, 'fmin0.01_K4_r0')]['rows'], band)
                na, pa = counts(R_[('Aplain', s, 'fmin0.01_K4_r0')]['rows'], band)
                dS1.append(recall(nt, pt, S1) - recall(na, pa, S1)); dS4.append(recall(nt, pt, S4) - recall(na, pa, S4))
                dE.append(recall(nt, pt, BIG_EDGE) - recall(na, pa, BIG_EDGE)); dI.append(recall(nt, pt, BIG_INT) - recall(na, pa, BIG_INT))
                dFn.append(fpr(nt, pt, NEG_NEAR) - fpr(na, pa, NEG_NEAR)); dFf.append(fpr(nt, pt, NEG_FAR) - fpr(na, pa, NEG_FAR))
            c_size = int(sum(1 for x, y in zip(dS4, dS1) if x < y)); c_int = int(sum(1 for x, y in zip(dI, dE) if x < y))
            c_far = int(sum(1 for v in dFf if v > 0)); c_near = int(sum(1 for v in dFn if v > 0))
            print(f'  [{lab}] {SHORT[t]}-A: dR S1 {np.mean(dS1):+.4f} S4 {np.mean(dS4):+.4f} [S4<S1: {c_size}/12]  '
                  f'large buildings edge {np.mean(dE):+.4f} interior {np.mean(dI):+.4f} [interior<edge: {c_int}/12]  '
                  f'dFPR near {np.mean(dFn):+.5f} [>0: {c_near}/12] far {np.mean(dFf):+.5f} [>0: {c_far}/12]')
            strat[f'{SHORT[t]}_{lab}'] = {'dR_S1': dS1, 'dR_S4': dS4, 'dR_big_edge': dE, 'dR_big_int': dI, 'dFPR_near': dFn, 'dFPR_far': dFf,
                                          'count_S4_lt_S1': c_size, 'count_int_lt_edge': c_int, 'count_far_gt0': c_far, 'count_near_gt0': c_near}
    out['H']['strata'] = strat
    r_med = [float(np.median([r['dgm_rel'] for r in R_[('Cconfidence', s, 'fmin0.01_K4_r0')]['rows']])) for s in SEEDS]
    out['H']['U-E2'] = {'dgm_median_per_seed': r_med}
    print(f'  [U-E2] arm C DGM residual relative magnitude, per-image median (mean over 12 seeds) = {np.mean(r_med):.4f}')

    # ---------- Inria reference (read-only committed summaries) ----------
    ref = {}
    try:
        rq5 = L(ROOT / 'results' / 'rq5_summary.json')['H']
        st1 = L(ROOT / 'results' / 'stage1_summary.json')['summary']
        ref = {'inria_K4_C_minus_A': st1['K4_C_minus_A']['effect'], 'inria_K4_dir': st1['K4_C_minus_A']['direction_positive'],
               'inria_f2': {k: rq5['C_f2'][k] for k in ('count_S4_lt_S1', 'count_int_lt_edge', 'count_far_gt0')},
               'inria_f4': {k: rq5['C_f4'][k] for k in ('count_S4_lt_S1', 'count_int_lt_edge', 'count_far_gt0')},
               'inria_B_f2_S4_lt_S1': rq5['B_f2']['count_S4_lt_S1'],
               'inria_dgm_median_f': {f: float(np.mean(rq5[f'dgm_f{f}']['median_rel_per_seed'])) for f in (1, 2, 4)}}
    except Exception as e:
        print('  (failed to read the Inria reference:', e, ')')
    out['inria_reference'] = ref

    # ---------- verdicts ----------
    print('\n' + '=' * 76); print('[verdicts] (>= 9/12 = supported; descriptive only if the stop-loss triggered)'); print('=' * 76)
    if deep:
        print('  stop-loss triggered: U-H1..H5 are not judged; the numbers above describe a deep-degradation point only')
        out['verdict'] = {'stop_loss': True}
    else:
        dirc = ca['direction_positive']; eff = ca['effect']
        h1 = 'replicates-undetectable' if (abs(eff) < m4 and 3 <= dirc <= 9) else ('detectable' if (abs(eff) >= m4 and (dirc >= 10 or dirc <= 2)) else 'mixed')
        h2 = bool(rBA >= 2 and rCA < rBA)
        h3 = strat['C_1px']['count_S4_lt_S1'] >= 9
        h4a = strat['C_1px']['count_int_lt_edge'] >= 9; h4b = strat['C_4px']['count_int_lt_edge'] >= 9
        h5 = strat['B_1px']['count_S4_lt_S1'] >= 9
        print(f"  U-H1 primary replication   C-A effect {eff:+.4f} vs MDE {m4:.4f}, direction {dirc}/12 -> {h1}")
        print(f"  U-H2 variance amplification SD_B/SD_A {rBA:.2f} (>=2?) SD_C/SD_A {rCA:.2f} (<B/A?) -> {'supported' if h2 else 'not supported'}")
        print(f"  U-H3 size ordering         C-A S4<S1 {strat['C_1px']['count_S4_lt_S1']}/12 -> {'supported' if h3 else 'not supported'}   (Inria f=2 {ref.get('inria_f2', {}).get('count_S4_lt_S1', '?')}/12, f=4 {ref.get('inria_f4', {}).get('count_S4_lt_S1', '?')}/12)")
        print(f"  U-H4 interior ordering     1px {strat['C_1px']['count_int_lt_edge']}/12, 4px {strat['C_4px']['count_int_lt_edge']}/12 -> {'supported' if (h4a and h4b) else ('depends on band definition' if (h4a or h4b) else 'not supported')}")
        print(f"  U-H5 attribution           B-A S4<S1 {strat['B_1px']['count_S4_lt_S1']}/12 -> {'geometry module' if (h3 and h5) else ('prototype/gating side' if h3 else 'H3 fails, no attribution')}")
        print(f"  U-E1 false positives (exploratory)  C-A far>0 {strat['C_1px']['count_far_gt0']}/12, near>0 {strat['C_1px']['count_near_gt0']}/12")
        print(f"  U-E2 residual magnitude (exploratory) this domain {np.mean(r_med):.4f} vs Inria f=1/2/4 " + ' / '.join(f"{v:.4f}" for v in ref.get('inria_dgm_median_f', {}).values()))
        out['verdict'] = {'stop_loss': False, 'U-H1': h1, 'U-H2': h2, 'U-H3': bool(h3), 'U-H4_1px': bool(h4a), 'U-H4_4px': bool(h4b),
                          'U-H5': ('geometry' if (h3 and h5) else ('prototype' if h3 else 'n/a'))}
    outp = ROOT / 'results' / f'{a.dir}_summary.json'
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n  -> {outp}')


if __name__ == '__main__':
    main()
