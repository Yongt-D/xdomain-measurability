# -*- coding: utf-8 -*-
"""
Research-question analyses of Appendix S (revision 19): RQ1 variance amplification, RQ2 operating-point drift,
RQ3 pseudo-interaction, RQ4 sample-level origin of the degradation loss, E1 (exploratory).
**Committed before any new number it produces.** Reads committed result files only; no inference.
Criteria follow Appendix S.3; the script prints "supported / not supported" per item.
No significance claims; correlations and proportions are descriptive.

Usage: python scripts/rq_analyses.py --rq all|1|2|3|4|e1 [--gsd-available 1,2,4,8]
"""
import argparse, json, sys, pathlib, itertools
import numpy as np
from scipy import stats
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
SEEDS = list(range(100, 112))
TAGS = ('Aplain', 'Bplain', 'Cconfidence')
SHORT = {'Aplain': 'A', 'Bplain': 'B', 'Cconfidence': 'C'}
PIX = 512 * 512


def L(p):
    p = pathlib.Path(p)
    if not p.exists():
        print(f'  missing {p}'); sys.exit(1)
    return json.loads(p.read_text(encoding='utf-8'))


def cross(tag, s, k=4):
    rs = range(5) if tag == 'Cconfidence' else range(1)
    return float(np.mean([L(ROOT / 'results' / 'stage1' / f'{tag}_seed{s}_fmin0.01_K{k}_r{r}.json')['target']['iou'] for r in rs]))


def indomain(tag, s):
    return L(ROOT / 'results' / 'stage1_train' / f'{tag}_seed{s}.results.json')['test_metrics']['iou']


def per_sample_l1(tag, s, k=4, r=0):
    rows = L(ROOT / 'results' / 'stage1' / f'{tag}_seed{s}_fmin0.01_K{k}_r{r}.per_sample.json')
    return {x['name']: x for x in rows}


def per_sample_l2(tag, s, f):
    rows = L(ROOT / 'results' / 'l2' / f'{tag}_seed{s}_gsd{f}.per_sample.json')
    return {x['name']: x for x in rows}


def sd(x):
    return float(np.std(np.asarray(x, float), ddof=1))


# ---------------- RQ1 ----------------
def rq1(out):
    print('=' * 76); print('[RQ1] amplification of training randomness by domain shift (H1a/H1b; H1c exploratory)'); print('=' * 76)
    res = {}
    for t in TAGS:
        sin = sd([indomain(t, s) for s in SEEDS]); scr = sd([cross(t, s) for s in SEEDS])
        res[t] = {'sd_in': sin, 'sd_cross': scr, 'ratio': scr / sin}
        print(f'  {SHORT[t]}: in-domain SD {sin:.6f}  cross-domain SD {scr:.6f}  amplification {scr / sin:.2f}')
    h1a = {t: res[t]['ratio'] >= 2 for t in TAGS}
    print(f'  H1a (amplification >= 2 per arm): { {SHORT[t]: ("supported" if v else "not supported") for t, v in h1a.items()} }')
    rb_cross = res['Bplain']['sd_cross'] / res['Aplain']['sd_cross']; rb_in = res['Bplain']['sd_in'] / res['Aplain']['sd_in']
    h1b = rb_cross >= 2 and rb_in <= 1.5
    print(f'  H1b: SD_B/SD_A cross-domain {rb_cross:.2f} (>=2?)  in-domain {rb_in:.2f} (<=1.5?)  -> {"supported" if h1b else "not supported"}')
    # H1c per-image seed-SD (K=4, r0; C uses r0 to match A/B)
    h1c = {}
    for t in TAGS:
        P = [per_sample_l1(t, s) for s in SEEDS]
        names = sorted(P[0])
        M = np.array([[P[i][n]['iou'] for n in names] for i in range(len(SEEDS))])   # (12, 2997)
        fg = np.array([(P[0][n]['tp'] + P[0][n]['fn']) / PIX for n in names])
        psd = M.std(0, ddof=1)
        var = psd ** 2
        order = np.argsort(-var); cum = np.cumsum(var[order]) / var.sum()
        share50 = float((np.searchsorted(cum, 0.5) + 1) / len(names))
        nz = fg > 0
        rho = stats.spearmanr(fg[nz], psd[nz]).correlation
        h1c[t] = {'median_per_image_sd': float(np.median(psd)), 'mean_per_image_sd': float(psd.mean()),
                  'share_images_for_50pct_var': share50, 'spearman_fg_vs_sd_nonempty': float(rho),
                  'n_nonempty': int(nz.sum())}
        print(f'  H1c {SHORT[t]}: per-image seed-SD median {np.median(psd):.4f} mean {psd.mean():.4f};'
              f' {share50 * 100:.1f}% of the images account for 50% of the variance; Spearman(building fraction, SD | non-empty) = {rho:+.3f} (n={nz.sum()})')
    out['RQ1'] = {'arms': res, 'H1a': h1a, 'H1b': {'supported': bool(h1b), 'ratio_cross': rb_cross, 'ratio_in': rb_in}, 'H1c': h1c}


# ---------------- RQ2 ----------------
def rq2(out, factors):
    print('=' * 76); print('[RQ2] drift of the decision operating point with domain shift and degradation (H2a f=1; H2b/H2c f>=2)'); print('=' * 76)
    D = ROOT / 'results' / 'thrdiag_stage1'
    res = {}
    for f in factors:
        rows = {}
        for t in ('Aplain', 'Cconfidence'):
            for s in SEEDS:
                p = D / f'{t}_seed{s}_gsd{f}.json'
                if not p.exists():
                    print(f'  f={f}: missing {p.name}, skipping this f'); rows = None; break
                d = L(p)
                rows[(t, s)] = {'frozen_t': d['frozen_threshold'], 'oracle_t': d['oracle_threshold'],
                                'share': 1 - d['iou_frozen'] / max(d['iou_oracle_wide'], 1e-12),
                                'at_boundary': d['oracle_at_boundary']}
            if rows is None:
                break
        if rows is None:
            continue
        below = sum(int(v['oracle_t'] < v['frozen_t']) for v in rows.values())
        shareA = np.array([rows[('Aplain', s)]['share'] for s in SEEDS]); shareC = np.array([rows[('Cconfidence', s)]['share'] for s in SEEDS])
        c_gt_a = int((shareC > shareA).sum())
        res[f] = {'oracle_below_frozen': below, 'n': len(rows), 'share_A_mean': float(shareA.mean()), 'share_C_mean': float(shareC.mean()),
                  'C_share_gt_A_count': c_gt_a, 'oracle_t_median_A': float(np.median([rows[('Aplain', s)]['oracle_t'] for s in SEEDS])),
                  'oracle_t_median_C': float(np.median([rows[('Cconfidence', s)]['oracle_t'] for s in SEEDS])),
                  'n_at_boundary': int(sum(v['at_boundary'] for v in rows.values()))}
        print(f'  f={f}: oracle threshold < frozen threshold {below}/{len(rows)}; mean calibration share A {shareA.mean():.3f} C {shareC.mean():.3f};'
              f' C>A paired {c_gt_a}/12; oracle median A {res[f]["oracle_t_median_A"]:.2f} C {res[f]["oracle_t_median_C"]:.2f}; at boundary {res[f]["n_at_boundary"]}')
    if 1 in res:
        print(f'  H2a (f=1: all 24/24 below the frozen threshold): {"supported" if res[1]["oracle_below_frozen"] == res[1]["n"] else "not supported"}')
    fs = [f for f in (2, 4, 8) if f in res]
    if fs:
        h2b = all(res[f]['C_share_gt_A_count'] >= 9 for f in fs)
        print(f'  H2b (f in {fs}: C share > A paired >= 9/12 at every point): {"supported" if h2b else "not supported"}'
              f'{" (only part of the f values available)" if len(fs) < 3 else ""}')
        seq = [f for f in (1, 2, 4, 8) if f in res]
        monoA = all(res[a]['share_A_mean'] < res[b]['share_A_mean'] for a, b in zip(seq, seq[1:]))
        monoC = all(res[a]['share_C_mean'] < res[b]['share_C_mean'] for a, b in zip(seq, seq[1:]))
        print(f'  H2c (share increases monotonically with f): A {"supported" if monoA else "not supported"}  C {"supported" if monoC else "not supported"} (available f={seq})')
        res['H2b'] = bool(h2b); res['H2c'] = {'A': bool(monoA), 'C': bool(monoC)}
    else:
        print('  H2b/H2c: the widened diagnostics for f>=2 are not complete yet (M5 running); pending')
    out['RQ2'] = {str(k): v for k, v in res.items()}


# ---------------- RQ3 ----------------
def rq3(out):
    print('=' * 76); print('[RQ3] pseudo-interaction of an inert factor (H3; cross-bed order-of-magnitude analogy)'); print('=' * 76)
    d = np.array([cross('Cconfidence', s) - cross('Aplain', s) for s in SEEDS])
    I = np.array([d[i] - d[j] for i, j in itertools.combinations(range(len(SEEDS)), 2)])
    s_d = sd(d)
    frac = float((np.abs(I) >= 0.047).mean())
    dev = abs(sd(I) - np.sqrt(2) * s_d) / (np.sqrt(2) * s_d)
    h3 = frac >= 0.05 and dev < 0.3
    print(f'  s_d(C-A, K=4) = {s_d:.6f}; pseudo-interaction I_ij (66 pairs) SD = {sd(I):.6f}, sqrt(2)*s_d = {np.sqrt(2) * s_d:.6f} (relative deviation {dev * 100:.1f}%)')
    print(f'  fraction with |I| >= 0.047 = {frac * 100:.1f}% (criterion >= 5%); median |I| {np.median(np.abs(I)):.4f}, max {np.abs(I).max():.4f}')
    print(f'  H3: {"supported" if h3 else "not supported"} -- a single-run 2x2 "interaction" of order 0.047 is {"inside" if h3 else "outside"} the noise of this bed (order-of-magnitude analogy only)')
    out['RQ3'] = {'s_d': s_d, 'sd_I': sd(I), 'frac_abs_I_ge_0.047': frac, 'rel_dev_from_sqrt2_sd': dev, 'H3': bool(h3),
                  'median_abs_I': float(np.median(np.abs(I))), 'max_abs_I': float(np.abs(I).max())}


# ---------------- RQ4 ----------------
def rq4(out, factors):
    print('=' * 76); print('[RQ4] sample-level origin of the geometry mechanism\'s degradation loss (H4 at f=2; f=1 control)'); print('=' * 76)
    res = {}
    for f in factors:
        cnt = 0; rows = []
        cnt_ne = 0
        for s in SEEDS:
            PC, PA = per_sample_l2('Cconfidence', s, f), per_sample_l2('Aplain', s, f)
            names = sorted(PA)
            fg = np.array([(PA[n]['tp'] + PA[n]['fn']) / PIX for n in names])
            dif = np.array([PC[n]['iou'] - PA[n]['iou'] for n in names])
            med = np.median(fg)
            lo, hi = dif[fg <= med].mean(), dif[fg > med].mean()
            ne = fg > 0
            med2 = np.median(fg[ne]); lo2, hi2 = dif[ne & (fg <= med2)].mean(), dif[ne & (fg > med2)].mean()
            cnt += int(lo < hi); cnt_ne += int(lo2 < hi2)
            rows.append({'seed': s, 'delta_low': float(lo), 'delta_high': float(hi), 'delta_low_nonempty': float(lo2),
                         'delta_high_nonempty': float(hi2), 'median_fg': float(med)})
        res[f] = {'count_low_lt_high': cnt, 'count_low_lt_high_nonempty': cnt_ne, 'rows': rows,
                  'mean_delta_low': float(np.mean([r['delta_low'] for r in rows])), 'mean_delta_high': float(np.mean([r['delta_high'] for r in rows]))}
        print(f'  f={f}: delta_low < delta_high in {cnt}/12 seeds (excluding empty images {cnt_ne}/12); mean delta_low {res[f]["mean_delta_low"]:+.4f}  delta_high {res[f]["mean_delta_high"]:+.4f}')
    if 2 in res:
        h4 = res[2]['count_low_lt_high'] >= 9
        print(f'  H4 (>= 9/12 at f=2): {"supported" if h4 else "not supported"} -- the extra loss of the geometry mechanism {"is" if h4 else "is not"} concentrated in images with a small building fraction')
        res['H4'] = bool(h4)
    out['RQ4'] = {str(k): v for k, v in res.items()}


# ---------------- E1 ----------------
def e1(out):
    print('=' * 76); print('[E1] geometry reliance vs cross-domain performance (exploratory, no criterion; needs the F.1 ablations)'); print('=' * 76)
    D = ROOT / 'results' / 'ablations'
    g, c, ca = [], [], []
    for s in SEEDS:
        p = D / f'stage1_Cconfidence_seed{s}.json'
        if not p.exists():
            print(f'  missing {p.name} -- ablations incomplete, E1 pending'); return
        d = L(p); g.append(d['iou']['base'] - d['iou']['geom_off']); c.append(cross('Cconfidence', s)); ca.append(cross('Cconfidence', s) - cross('Aplain', s))
    r1 = stats.spearmanr(g, c).correlation; r2 = stats.spearmanr(g, ca).correlation
    print(f'  Spearman(geometry reliance base-geom_off, cross-domain IoU_C) = {r1:+.3f};  vs (C-A) = {r2:+.3f} (n=12, descriptive)')
    out['E1'] = {'geom_reliance_per_seed': g, 'spearman_vs_C': float(r1), 'spearman_vs_CminusA': float(r2)}


# ---------------- E2 (post-hoc exploration, added on the evening of 2026-09-11 after the H2a exception was seen; no criterion) ----------------
def e2(out, factors):
    print('=' * 76); print('[E2] operating-point mismatch vs cross-domain performance (**post-hoc exploration**, added after the H2a exception C seed111 was seen; no criterion)'); print('=' * 76)
    D = ROOT / 'results' / 'thrdiag_stage1'
    res = {}
    for f in factors:
        for t in ('Aplain', 'Cconfidence'):
            gap, share, iou = [], [], []
            for s in SEEDS:
                p = D / f'{t}_seed{s}_gsd{f}.json'
                if not p.exists():
                    break
                d = L(p); gap.append(d['frozen_threshold'] - d['oracle_threshold'])
                share.append(1 - d['iou_frozen'] / max(d['iou_oracle_wide'], 1e-12)); iou.append(d['iou_frozen'])
            if len(iou) < len(SEEDS):
                continue
            r_gap = stats.spearmanr(iou, gap).correlation; r_share = stats.spearmanr(iou, share).correlation
            res[f'{SHORT[t]}_gsd{f}'] = {'spearman_iou_vs_threshold_gap': float(r_gap), 'spearman_iou_vs_calib_share': float(r_share)}
            print(f'  f={f} {SHORT[t]}: Spearman(frozen IoU, frozen-oracle threshold gap) = {r_gap:+.3f};  Spearman(frozen IoU, calibration share) = {r_share:+.3f} (n=12, descriptive)')
    out['E2'] = res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rq', default='all')
    ap.add_argument('--gsd-available', default='1,2,4,8')
    ap.add_argument('--out', default=str(ROOT / 'results' / 'rq_summary.json'))
    a = ap.parse_args()
    factors = [int(x) for x in a.gsd_available.split(',')]
    out = {'seeds': SEEDS, 'note': 'appendix S; descriptive; no significance claims'}
    prev = pathlib.Path(a.out)
    if prev.exists():
        out.update({k: v for k, v in json.loads(prev.read_text(encoding='utf-8')).items() if k.startswith(('RQ', 'E1'))})
    if a.rq in ('all', '1'): rq1(out)
    if a.rq in ('all', '2'): rq2(out, factors)
    if a.rq in ('all', '3'): rq3(out)
    if a.rq in ('all', '4'): rq4(out, factors)
    if a.rq in ('all', 'e1'): e1(out)
    if a.rq in ('all', 'e2'): e2(out, factors)
    pathlib.Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n  -> {a.out}')


if __name__ == '__main__':
    main()
