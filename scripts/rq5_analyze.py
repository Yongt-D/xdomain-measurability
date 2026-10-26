# -*- coding: utf-8 -*-
"""
RQ5 (Appendix T) analysis: gates T.3 -> hypotheses H5a-H5e. **Committed before any of its numbers existed.**
No significance claims; "supported" = the pre-registered >= 9/12-seed count rule; correlations are descriptive.
"""
import argparse, json, sys, pathlib, collections
import numpy as np
from scipy import stats
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
SEEDS = list(range(100, 112))
TAGS = ('Aplain', 'Bplain', 'Cconfidence')
SHORT = {'Aplain': 'A', 'Bplain': 'B', 'Cconfidence': 'C'}
FACTORS = (1, 2, 4, 8)
E1 = 0.0005
POS = list(range(8)); S1 = [0, 1]; S4 = [6, 7]; BIG_EDGE = [4, 6]; BIG_INT = [5, 7]; NEG_NEAR = 8; NEG_FAR = 9


def L(p):
    p = pathlib.Path(p)
    if not p.exists():
        print(f'  missing {p}'); sys.exit(1)
    return json.loads(p.read_text(encoding='utf-8'))


def recall(rows, strata):
    tp = sum(r['p1'][k] for r in rows for k in strata); n = sum(r['n'][k] for r in rows for k in strata)
    return tp / max(n, 1)


def fpr(rows, k):
    return sum(r['p1'][k] for r in rows) / max(sum(r['n'][k] for r in rows), 1)


def count_support(vals, cond):
    return int(sum(1 for v in vals if cond(v)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='rq5')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    D = ROOT / 'results' / a.dir
    shas = {}
    for line in (ROOT / 'results' / 'stage1_ckpt_sha256.txt').read_text(encoding='utf-8').splitlines():
        h, name = line.split(); shas[name.strip('*')] = h
    R = {(t, s, f): L(D / f'{t}_seed{s}_gsd{f}.json') for t in TAGS for s in SEEDS for f in FACTORS}
    out = {'seeds': SEEDS, 'note': 'appendix T; descriptive; no significance claims', 'gates': {}, 'H': {}}

    print('=' * 76); print('[gate T.3.1] provenance: sha256 / frozen threshold / manifest'); print('=' * 76)
    bad = []
    for (t, s, f), d in R.items():
        l1 = L(ROOT / 'results' / 'stage1' / f'{t}_seed{s}_fmin0.01_K4_r0.json')
        if d['checkpoint_sha256'] != shas[f'{t}_seed{s}.pth']: bad.append((t, s, f, 'sha256'))
        if abs(d['threshold'] - l1['source_val_threshold']) > 1e-9: bad.append((t, s, f, 'threshold'))
        if d['manifest_key'] != 'fmin0.01_K4_r0': bad.append((t, s, f, 'manifest'))
    if bad:
        print('  mismatch:', bad[:10]); print('  => stop'); sys.exit(2)
    print('  all passed')

    print('\n' + '=' * 76); print(f'[gate T.3.2] cross-machine calibration: image-level IoU vs the same key in results/l2 differs by < {E1} (per machine)'); print('=' * 76)
    by_m = collections.defaultdict(list); worst = (0, None)
    for (t, s, f), d in R.items():
        ref = L(ROOT / 'results' / 'l2' / f'{t}_seed{s}_gsd{f}.json')['target']['iou']
        dv = d['overall']['iou'] - ref; by_m[d['hostname']].append(dv)
        if abs(dv) > worst[0]: worst = (abs(dv), (t, s, f, d['hostname']))
    for m, v in sorted(by_m.items()):
        v = np.abs(np.asarray(v)); print(f'  {m:24s} n={v.size:3d}  max|d| = {v.max():.8f}  mean|d| = {v.mean():.8f}')
    out['gates']['T332'] = {m: {'n': len(v), 'max_abs': float(np.abs(v).max())} for m, v in by_m.items()}
    if worst[0] >= E1:
        print(f'  => **FAIL** global max|d| = {worst[0]:.8f} @ {worst[1]}; results of that machine must not be pooled; stop'); sys.exit(2)
    print(f'  => pass (global max|d| = {worst[0]:.8f} @ {worst[1]})')

    print('\n' + '=' * 76); print('[gate T.3.3] label consistency: per-stratum pixel totals bit-identical across all results'); print('=' * 76)
    totals = {tuple(d['strata_n_total']) for d in R.values()}
    if len(totals) != 1:
        print(f'  mismatch: {len(totals)} variants; stop'); sys.exit(2)
    tot = next(iter(totals)); names = next(iter(R.values()))['strata_names']
    print('  pass. pixel share per stratum (within positives / within negatives):')
    posn = sum(tot[:8]); negn = sum(tot[8:])
    for k in range(10):
        print(f'    {names[k]:9s} {tot[k]:>12d}  {tot[k] / (posn if k < 8 else negn) * 100:6.2f}%')
    out['gates']['strata_n_total'] = list(tot)

    # ---------- hypotheses ----------
    def dR(t, s, f, strata):
        return recall(R[(t, s, f)]['rows'], strata) - recall(R[('Aplain', s, f)]['rows'], strata)

    print('\n' + '=' * 76); print('[H5a-H5e]'); print('=' * 76)
    for f in FACTORS:
        print(f'--- f={f} (GSD {0.3 * f:.1f} m) ---')
        for t in ('Cconfidence', 'Bplain'):
            d_s1 = [dR(t, s, f, S1) for s in SEEDS]; d_s4 = [dR(t, s, f, S4) for s in SEEDS]
            d_edge = [dR(t, s, f, BIG_EDGE) for s in SEEDS]; d_int = [dR(t, s, f, BIG_INT) for s in SEEDS]
            c_size = count_support(zip(d_s4, d_s1), lambda v: v[0] < v[1])
            c_int = count_support(zip(d_int, d_edge), lambda v: v[0] < v[1])
            d_far = [fpr(R[(t, s, f)]['rows'], NEG_FAR) - fpr(R[('Aplain', s, f)]['rows'], NEG_FAR) for s in SEEDS]
            d_near = [fpr(R[(t, s, f)]['rows'], NEG_NEAR) - fpr(R[('Aplain', s, f)]['rows'], NEG_NEAR) for s in SEEDS]
            c_far = count_support(d_far, lambda v: v > 0)
            print(f'  {SHORT[t]}-A: dR S1 {np.mean(d_s1):+.4f}  S4 {np.mean(d_s4):+.4f}  [S4<S1: {c_size}/12]   '
                  f'large buildings dR edge {np.mean(d_edge):+.4f} interior {np.mean(d_int):+.4f}  [interior<edge: {c_int}/12]   '
                  f'dFPR near {np.mean(d_near):+.5f} far {np.mean(d_far):+.5f}  [far>0: {c_far}/12]')
            out['H'][f'{SHORT[t]}_f{f}'] = {'dR_S1': d_s1, 'dR_S4': d_s4, 'dR_big_edge': d_edge, 'dR_big_int': d_int,
                                          'dFPR_near': d_near, 'dFPR_far': d_far,
                                          'count_S4_lt_S1': c_size, 'count_int_lt_edge': c_int, 'count_far_gt0': c_far}
        # H5d
        r_med = [float(np.median([r['dgm_rel'] for r in R[('Cconfidence', s, f)]['rows']])) for s in SEEDS]
        rho = []
        for s in SEEDS:
            rc, ra = R[('Cconfidence', s, f)]['rows'], R[('Aplain', s, f)]['rows']
            x, y = [], []
            for i in range(len(rc)):
                n = sum(rc[i]['n'][k] for k in BIG_EDGE + BIG_INT)
                if n == 0: continue
                x.append(rc[i]['dgm_rel'])
                y.append(sum(rc[i]['p1'][k] for k in BIG_EDGE + BIG_INT) / n - sum(ra[i]['p1'][k] for k in BIG_EDGE + BIG_INT) / n)
            rho.append(float(stats.spearmanr(x, y).correlation) if len(x) > 10 else float('nan'))
        print(f'  relative DGM residual magnitude of C: mean per-image median {np.mean(r_med):.4f}; mean Spearman(r_i, large-building R_C-R_A) {np.nanmean(rho):+.3f}  [<0: {count_support(rho, lambda v: v < 0)}/12]')
        out['H'][f'dgm_f{f}'] = {'median_rel_per_seed': r_med, 'spearman_per_seed': rho}
    # verdict summary (f=2)
    h = out['H']
    print('\n' + '=' * 76); print('[verdicts] (f=2; >= 9/12 = supported)'); print('=' * 76)
    h5a = h['C_f2']['count_S4_lt_S1'] >= 9
    h5b = h['C_f2']['count_int_lt_edge'] >= 9
    b_size = h['B_f2']['count_S4_lt_S1'] >= 9
    up = count_support(zip(h['dgm_f1']['median_rel_per_seed'], h['dgm_f2']['median_rel_per_seed']), lambda v: v[1] > v[0])
    h5d1 = up >= 9
    h5d2 = count_support(h['dgm_f2']['spearman_per_seed'], lambda v: v < 0) >= 9
    print(f"  H5a size (C-A: S4<S1)               {h['C_f2']['count_S4_lt_S1']}/12 -> {'supported' if h5a else 'not supported'}")
    print(f"  H5b interior<edge (C-A, S3 u S4)    {h['C_f2']['count_int_lt_edge']}/12 -> {'supported' if h5b else 'not supported'}")
    print(f"  H5c attribution: S4<S1 of B-A       {h['B_f2']['count_S4_lt_S1']}/12 -> "
          f"{'geometry refinement module' if (h5a and b_size) else ('prototype/gating side' if (h5a and not b_size) else 'H5a fails, no attribution')}")
    print(f"  H5d(i) median r rises f1->f2         {up}/12 -> {'supported' if h5d1 else 'not supported'}")
    print(f"  H5d(ii) Spearman(r, dR_big)<0        {count_support(h['dgm_f2']['spearman_per_seed'], lambda v: v < 0)}/12 -> {'supported' if h5d2 else 'not supported'}")
    print(f"  H5e exploratory: dFPR_far>0 of C-A   {h['C_f2']['count_far_gt0']}/12 (no pre-set direction)")
    out['verdict_f2'] = {'H5a': bool(h5a), 'H5b': bool(h5b), 'H5c': ('geometry' if (h5a and b_size) else ('prototype' if h5a else 'n/a')),
                         'H5d_i': bool(h5d1), 'H5d_ii': bool(h5d2), 'H5e_count': h['C_f2']['count_far_gt0']}
    outp = pathlib.Path(a.out) if a.out else ROOT / 'results' / f'{a.dir}_summary.json'
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n  -> {outp}')


if __name__ == '__main__':
    main()
