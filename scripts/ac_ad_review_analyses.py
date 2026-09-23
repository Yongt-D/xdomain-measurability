# -*- coding: utf-8 -*-
"""
Appendices AC and AD: descriptive supplementary analyses triggered by the fourth internal review (2026-09-22).
**Committed before any of their numbers existed.** Everything here is descriptive: no hypothesis, no criterion, no verdict is changed.
  AC.2  nested (seed x support-draw) bootstrap of the C-A contrast on four (source, target) pairs, next to the seed-only bootstrap
        and the t interval, and a single-deployment interval (one seed, one support set);
  AD.1  Massachusetts, frozen threshold: predicted-positive fraction, precision, recall, F1 and near-band FPR per arm (from the
        tp/fp/fn already stored by the appendix-U evaluation; no new inference);
  AD.2  Massachusetts, 36 widened sweeps with per-threshold counts (scripts/ad_mass_counts_thrdiag.sh), behind gates G1-G3:
        prevalence-matched threshold per (arm, seed), contrasts at that threshold, and a partial PR-AUC over the 0.05-0.95 grid.
Writes results/ac_ad_review_summary.json.
"""
import json, sys, hashlib, pathlib
import numpy as np
from scipy import stats
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
R = ROOT / 'results'
SEEDS = list(range(100, 112))
ARMS = ('Aplain', 'Bplain', 'Cconfidence')
L = lambda p: json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
T975 = stats.t.ppf(0.975, 11)
B_BOOT, RNG_SEED = 10_000, 20260923
WIDE = [round(0.05 + 0.01 * i, 2) for i in range(91)]


def ci(d):
    d = np.asarray(d, float); se = d.std(ddof=1) / np.sqrt(len(d))
    return [float(d.mean() - T975 * se), float(d.mean() + T975 * se)]


def summ(d):
    d = np.asarray(d, float)
    return {'mean': float(d.mean()), 'sd': float(d.std(ddof=1)), 'ci95': ci(d), 'positive': int((d > 0).sum()), 'n': int(len(d))}


def target_iou(j):
    return j['target']['iou'] if 'target' in j else j['overall']['iou']


PAIRS = {'WHU->Inria': 'stage1', 'WHU->Mass': 'mass', 'Inria->WHU': 'y_whu', 'Inria->Mass': 'y_mass'}


# ====================================================================== AC.2
def ac_bootstrap(out):
    print('=' * 78); print('AC.2 nested bootstrap of C-A (arm C K=4, 12 seeds x 5 draws; A draw-invariant)'); print('=' * 78)
    rng = np.random.default_rng(RNG_SEED)
    res = {}
    for name, d in PAIRS.items():
        M = np.array([[target_iou(L(R / d / f'Cconfidence_seed{s}_fmin0.01_K4_r{r}.json')) for r in range(5)] for s in SEEDS])
        A = np.array([target_iou(L(R / d / f'Aplain_seed{s}_fmin0.01_K4_r0.json')) for s in SEEDS])
        D = M - A[:, None]                       # 12 x 5 per-(seed, draw) contrast
        per_seed = D.mean(1)                     # draw-averaged, as in the paper
        # nested bootstrap: seeds with replacement, then draws within each drawn seed
        idx_s = rng.integers(0, 12, size=(B_BOOT, 12))
        idx_r = rng.integers(0, 5, size=(B_BOOT, 12, 5))
        nested = D[idx_s[:, :, None], idx_r].mean(axis=(1, 2))
        seed_only = per_seed[idx_s].mean(1)
        sd_seed = per_seed.std(ddof=1); sd_within = float(np.mean(D.std(1, ddof=1)))
        pred_sd = float(np.sqrt(sd_seed ** 2 + sd_within ** 2))
        single = D.ravel()
        rec = {'mean': float(per_seed.mean()), 't_ci95': ci(per_seed), 'positive_seeds': int((per_seed > 0).sum()),
               'nested_boot_ci95': [float(np.percentile(nested, 2.5)), float(np.percentile(nested, 97.5))],
               'seed_only_boot_ci95': [float(np.percentile(seed_only, 2.5)), float(np.percentile(seed_only, 97.5))],
               'nested_boot_sd': float(nested.std(ddof=1)), 'seed_only_boot_sd': float(seed_only.std(ddof=1)),
               'sd_seed_of_draw_means': float(sd_seed), 'sd_within_seed_over_draws': sd_within,
               'single_deployment': {'empirical_q025_q975': [float(np.percentile(single, 2.5)), float(np.percentile(single, 97.5))],
                                     'median': float(np.median(single)), 'fraction_positive': float((single > 0).mean()), 'n': int(single.size),
                                     'normal_approx_pi95': [float(per_seed.mean() - T975 * pred_sd), float(per_seed.mean() + T975 * pred_sd)], 'sd_total': pred_sd},
               'B': B_BOOT, 'rng_seed': RNG_SEED}
        res[name] = rec
        print(f"  {name:12s} mean {rec['mean']:+.4f}  t {rec['t_ci95'][0]:+.4f},{rec['t_ci95'][1]:+.4f} | seed-only boot {rec['seed_only_boot_ci95'][0]:+.4f},{rec['seed_only_boot_ci95'][1]:+.4f} | nested {rec['nested_boot_ci95'][0]:+.4f},{rec['nested_boot_ci95'][1]:+.4f} | single-deployment emp {rec['single_deployment']['empirical_q025_q975'][0]:+.4f},{rec['single_deployment']['empirical_q025_q975'][1]:+.4f} normal {rec['single_deployment']['normal_approx_pi95'][0]:+.4f},{rec['single_deployment']['normal_approx_pi95'][1]:+.4f} frac+ {rec['single_deployment']['fraction_positive']:.2f}")
    out['AC2_nested_bootstrap'] = res


# ====================================================================== AD.1
def ad1_frozen_morphology(out):
    print('=' * 78); print('AD.1 Massachusetts, frozen threshold: predicted-positive fraction, precision, recall, near-band FPR (appendix-U files)'); print('=' * 78)
    per = {a: {'PP': [], 'LP': [], 'precision': [], 'recall': [], 'f1': [], 'fpr_near': [], 'iou': []} for a in ARMS}
    for a in ARMS:
        for s in SEEDS:
            j = L(R / 'mass' / f'{a}_seed{s}_fmin0.01_K4_r0.json'); o = j['overall']; N = sum(j['strata_n_total'])
            assert j['strata_names'][8] == 'neg_near'
            tp, fp, fn = o['tp'], o['fp'], o['fn']
            per[a]['PP'].append((tp + fp) / N); per[a]['LP'].append((tp + fn) / N)
            per[a]['precision'].append(tp / (tp + fp)); per[a]['recall'].append(tp / (tp + fn)); per[a]['f1'].append(2 * tp / (2 * tp + fp + fn))
            n8 = sum(r['n'][8] for r in j['rows']); p8 = sum(r['p1'][8] for r in j['rows']); per[a]['fpr_near'].append(p8 / n8)
            per[a]['iou'].append(o['iou'])
    LP = float(np.mean(per['Aplain']['LP'])); assert np.allclose(per['Aplain']['LP'], per['Cconfidence']['LP'])
    res = {'label_positive_fraction': LP, 'arms': {}, 'contrasts': {}}
    for a in ARMS:
        res['arms'][a] = {k: summ(v) for k, v in per[a].items() if k != 'LP'}
        res['arms'][a]['PP_over_LP'] = summ(np.array(per[a]['PP']) / LP)
        print(f"  {a:12s} PP {np.mean(per[a]['PP']):.4f} (PP/LP {np.mean(per[a]['PP'])/LP:.2f})  precision {np.mean(per[a]['precision']):.4f}  recall {np.mean(per[a]['recall']):.4f}  F1 {np.mean(per[a]['f1']):.4f}  near-band FPR {np.mean(per[a]['fpr_near']):.4f}  IoU {np.mean(per[a]['iou']):.4f}")
    for x, y in (('Cconfidence', 'Aplain'), ('Bplain', 'Aplain')):
        res['contrasts'][f'{x[0]}-{y[0]}'] = {k: summ(np.array(per[x][k]) - np.array(per[y][k])) for k in ('PP', 'precision', 'recall', 'f1', 'fpr_near', 'iou')}
        c = res['contrasts'][f'{x[0]}-{y[0]}']
        print(f"  {x[0]}-{y[0]}: dPP {c['PP']['mean']:+.4f} [{c['PP']['ci95'][0]:+.4f},{c['PP']['ci95'][1]:+.4f}] {c['PP']['positive']}/12 | dPrec {c['precision']['mean']:+.4f} ({c['precision']['positive']}/12) | dRec {c['recall']['mean']:+.4f} ({c['recall']['positive']}/12) | dFPRnear {c['fpr_near']['mean']:+.4f} ({c['fpr_near']['positive']}/12)")
    # arm C over the five draws (r0..r4): range of the draw means
    ppC = []
    for r in range(5):
        v = []
        for s in SEEDS:
            j = L(R / 'mass' / f'Cconfidence_seed{s}_fmin0.01_K4_r{r}.json'); o = j['overall']; v.append((o['tp'] + o['fp']) / sum(j['strata_n_total']))
        ppC.append(float(np.mean(v)))
    res['arms']['Cconfidence']['PP_mean_by_draw_r0_r4'] = ppC
    out['AD1_frozen_morphology'] = res
    return LP


# ====================================================================== AD.2
def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def ad2_prevalence_matched(out):
    print('=' * 78); print('AD.2 Massachusetts, 36 sweeps with counts: prevalence-matched threshold and partial PR-AUC'); print('=' * 78)
    D = R / 'thrdiag_mass_counts'
    files = {(a, s): D / f'{a}_seed{s}.json' for a in ARMS for s in SEEDS}
    missing = [str(p.name) for p in files.values() if not p.exists()]
    if missing:
        print(f'G3 FAILED: {len(missing)} of 36 files missing -> stop'); out['AD2_gates'] = {'G3': False, 'missing': missing}; return
    ref = dict(l.split()[::-1] for l in (R / 'stage1_ckpt_sha256.txt').read_text().split('\n') if l.strip())   # name -> sha
    g1 = g2 = True; g1_fail, g2_fail, mx = [], [], 0.0
    J = {}
    for (a, s), p in files.items():
        j = L(p); J[(a, s)] = j
        ck = pathlib.Path(j['checkpoint'])
        if sha256(ck) != ref[ck.name]: g1 = False; g1_fail.append(p.name)
        u = L(R / 'mass' / f'{a}_seed{s}_fmin0.01_K4_r0.json')['overall']['iou']
        d1 = abs(j['iou_frozen'] - u)
        t = L(R / 'thrdiag_mass' / f'{a}_seed{s}.json')['iou_by_threshold']
        d2 = max(abs(j['iou_by_threshold'][k] - t[k]) for k in j['iou_by_threshold'])
        mx = max(mx, d1, d2)
        if d1 >= 5e-4 or d2 >= 5e-4: g2 = False; g2_fail.append([p.name, d1, d2])
    out['AD2_gates'] = {'G1_sha256': g1, 'G1_failures': g1_fail, 'G2_iou_agreement': g2, 'G2_failures': g2_fail, 'G2_max_abs_diff': mx, 'G3_files': True}
    print(f"  gates: G1 sha256 {'pass' if g1 else 'FAIL'} | G2 iou agreement {'pass' if g2 else 'FAIL'} (max |diff| {mx:.2e}) | G3 36/36")
    if not (g1 and g2):
        print('  a gate failed -> stop'); return
    LP = None; per = {a: {'t_pm': [], 'iou_pm': [], 'prec_pm': [], 'rec_pm': [], 'pp_over_lp_pm': [], 'prauc': [], 'iou_frozen': [], 'iou_oracle': [], 'rec_lo': [], 'rec_hi': []} for a in ARMS}
    at_boundary = []
    for (a, s), j in J.items():
        N = j['n_pixels']; lp = j['label_positive_pixels'] / N
        LP = lp if LP is None else LP
        assert abs(lp - LP) < 1e-9
        C = {float(k): v for k, v in j['counts_by_threshold'].items()}
        pp = {t: (C[t][0] + C[t][1]) / N for t in WIDE}
        tpm = min(WIDE, key=lambda t: abs(pp[t] - lp))
        tp, fp, fn = C[tpm]
        per[a]['t_pm'].append(tpm); per[a]['iou_pm'].append(tp / (tp + fp + fn)); per[a]['prec_pm'].append(tp / (tp + fp)); per[a]['rec_pm'].append(tp / (tp + fn))
        per[a]['pp_over_lp_pm'].append(pp[tpm] / lp)
        if tpm in (WIDE[0], WIDE[-1]): at_boundary.append([a, s, tpm])
        pr = sorted(((C[t][0] / (C[t][0] + C[t][2]), C[t][0] / max(C[t][0] + C[t][1], 1)) for t in WIDE))
        rec_ = np.array([x[0] for x in pr]); prec_ = np.array([x[1] for x in pr])
        per[a]['prauc'].append(float(np.trapezoid(prec_, rec_) if hasattr(np, 'trapezoid') else np.trapz(prec_, rec_)))
        per[a]['rec_lo'].append(float(rec_[0])); per[a]['rec_hi'].append(float(rec_[-1]))
        per[a]['iou_frozen'].append(j['iou_frozen']); per[a]['iou_oracle'].append(j['iou_oracle_wide'])
    res = {'label_positive_fraction': LP, 'grid': [WIDE[0], WIDE[-1], 0.01], 't_pm_at_grid_boundary': at_boundary, 'arms': {}, 'contrasts': {}}
    for a in ARMS:
        v = per[a]
        res['arms'][a] = {'t_pm_median': float(np.median(v['t_pm'])), 't_pm_min': float(min(v['t_pm'])), 't_pm_max': float(max(v['t_pm'])), 't_pm_values': v['t_pm'],
                          'iou_pm': summ(v['iou_pm']), 'precision_pm': summ(v['prec_pm']), 'recall_pm': summ(v['rec_pm']), 'pp_over_lp_pm': summ(v['pp_over_lp_pm']),
                          'prauc_partial': summ(v['prauc']), 'recall_range_of_partial_curve': [float(np.mean(v['rec_lo'])), float(np.mean(v['rec_hi']))],
                          'iou_frozen': summ(v['iou_frozen']), 'iou_oracle': summ(v['iou_oracle'])}
        print(f"  {a:12s} t_pm median {np.median(v['t_pm']):.2f} [{min(v['t_pm']):.2f},{max(v['t_pm']):.2f}]  IoU@pm {np.mean(v['iou_pm']):.4f} (frozen {np.mean(v['iou_frozen']):.4f}, oracle {np.mean(v['iou_oracle']):.4f})  prec {np.mean(v['prec_pm']):.4f} rec {np.mean(v['rec_pm']):.4f}  PR-AUC(partial) {np.mean(v['prauc']):.4f} over recall [{np.mean(v['rec_lo']):.2f},{np.mean(v['rec_hi']):.2f}]")
    for x, y in (('Cconfidence', 'Aplain'), ('Bplain', 'Aplain')):
        k = f'{x[0]}-{y[0]}'
        res['contrasts'][k] = {'iou_at_t_pm': summ(np.array(per[x]['iou_pm']) - np.array(per[y]['iou_pm'])),
                               'iou_frozen': summ(np.array(per[x]['iou_frozen']) - np.array(per[y]['iou_frozen'])),
                               'iou_oracle': summ(np.array(per[x]['iou_oracle']) - np.array(per[y]['iou_oracle'])),
                               'prauc_partial': summ(np.array(per[x]['prauc']) - np.array(per[y]['prauc'])),
                               'precision_at_t_pm': summ(np.array(per[x]['prec_pm']) - np.array(per[y]['prec_pm'])),
                               'recall_at_t_pm': summ(np.array(per[x]['rec_pm']) - np.array(per[y]['rec_pm']))}
        c = res['contrasts'][k]
        print(f"  {k}: IoU@pm {c['iou_at_t_pm']['mean']:+.4f} [{c['iou_at_t_pm']['ci95'][0]:+.4f},{c['iou_at_t_pm']['ci95'][1]:+.4f}] {c['iou_at_t_pm']['positive']}/12 | frozen {c['iou_frozen']['mean']:+.4f} ({c['iou_frozen']['positive']}/12) | oracle {c['iou_oracle']['mean']:+.4f} ({c['iou_oracle']['positive']}/12) | PR-AUC {c['prauc_partial']['mean']:+.4f} [{c['prauc_partial']['ci95'][0]:+.4f},{c['prauc_partial']['ci95'][1]:+.4f}] {c['prauc_partial']['positive']}/12")
    out['AD2_prevalence_matched'] = res


def main():
    out = {'note': 'appendices AC/AD (revision 28); descriptive; no significance claims; no verdict changed', 'MDE_mass_12seed': 0.029}
    ac_bootstrap(out)
    ad1_frozen_morphology(out)
    ad2_prevalence_matched(out)
    (R / 'ac_ad_review_summary.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('-> results/ac_ad_review_summary.json')


if __name__ == '__main__':
    main()
