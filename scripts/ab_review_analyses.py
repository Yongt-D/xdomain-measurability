# -*- coding: utf-8 -*-
"""
Appendix AB: descriptive supplementary analyses triggered by the internal peer review of 2026-09-16.
**Committed before the AB.2 sweeps existed.** Everything here is descriptive: no hypothesis, no criterion, no verdict is changed.
  (a) seed x support-draw variance decomposition of arm C (K=4) on four (source, target) pairs;
  (b) seed-level interval of the operating-point part of the C-A contrast (frozen minus each-arm-oracle) on four pairs;
  (c) amplification of the seed SD on the raw IoU scale and on the logit(IoU) scale (GAPL arms; second-bed architectures if per-seed values are available);
  (d) the two MDE values of the Inria primary contrast: pilot (stage 0b s_d) and 12-seed s_d;
  (e) AB.2: per-draw operating-point decomposition on Massachusetts (draws r0..r4), behind gates G1-G3.
Writes results/ab_review_summary.json.
"""
import json, sys, hashlib, pathlib
import numpy as np
from scipy import stats
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
R = ROOT / 'results'
SEEDS = list(range(100, 112))
L = lambda p: json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
T975 = stats.t.ppf(0.975, 11)


def ci(d):
    d = np.asarray(d, float); se = d.std(ddof=1) / np.sqrt(len(d))
    return [float(d.mean() - T975 * se), float(d.mean() + T975 * se)]


def mde(s_d, n=12, power=0.8):
    lo, hi = 1e-5, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2; ncp = mid / (s_d / np.sqrt(n)); crit = stats.t.ppf(0.975, n - 1)
        p = stats.nct.sf(crit, n - 1, ncp) + stats.nct.cdf(-crit, n - 1, ncp)
        if p >= power: hi = mid
        else: lo = mid
    return float(hi)


def target_iou(j):
    return j['target']['iou'] if 'target' in j else j['overall']['iou']


PAIRS = {  # name -> (main-evaluation dir, widened-sweep dir, r0-only)
    'WHU->Inria': ('stage1', 'thrdiag_stage1', '_gsd1'),
    'WHU->Mass': ('mass', 'thrdiag_mass', ''),
    'WHU->Mass (full tile)': ('mass_full', 'thrdiag_mass_full', ''),
    'Inria->WHU': ('y_whu', 'thrdiag_y_whu', ''),
    'Inria->Mass': ('y_mass', 'thrdiag_y_mass', ''),
}


def main():
    out = {'note': 'appendix AB; descriptive; no significance claims; no verdict changed'}
    # ---------------- (a) seed x draw decomposition ----------------
    print('=' * 78); print('(a) seed x support-draw variance decomposition, arm C, K=4'); print('=' * 78)
    a = {}
    for name in ('WHU->Inria', 'WHU->Mass', 'Inria->WHU', 'Inria->Mass'):
        d = PAIRS[name][0]
        M = np.array([[target_iou(L(R / d / f'Cconfidence_seed{s}_fmin0.01_K4_r{r}.json')) for r in range(5)] for s in SEEDS])
        A = np.array([target_iou(L(R / d / f'Aplain_seed{s}_fmin0.01_K4_r0.json')) for s in SEEDS])
        gm = M.mean(); se_ = M.mean(1) - gm; de = M.mean(0) - gm; res = M - gm - se_[:, None] - de[None, :]
        ca = M - A[:, None]
        rec = {'C_mean': float(gm), 'sd_seed_main': float(se_.std(ddof=1)), 'sd_draw_main': float(de.std(ddof=1)), 'sd_residual': float(res.std(ddof=1)),
               'mean_within_seed_sd_over_draws': float(np.mean(M.std(1, ddof=1))), 'sd_across_seeds_of_draw_mean': float(M.mean(1).std(ddof=1)),
               'CA_per_draw_mean': [float(x) for x in ca.mean(0)], 'CA_per_draw_positive': [int((ca[:, r] > 0).sum()) for r in range(5)],
               'CA_per_draw_sd_seed': [float(x) for x in ca.std(0, ddof=1)], 'CA_per_draw_ci95': [ci(ca[:, r]) for r in range(5)]}
        a[name] = rec
        print(f"  {name:22s} C {gm:.4f} | seed SD {rec['sd_across_seeds_of_draw_mean']:.4f} | draw main SD {rec['sd_draw_main']:.4f} | within-seed draw SD {rec['mean_within_seed_sd_over_draws']:.4f} | residual {rec['sd_residual']:.4f}")
        print(f"  {'':22s} C-A per draw {[round(x, 4) for x in rec['CA_per_draw_mean']]} positive {rec['CA_per_draw_positive']}")
    out['a_seed_draw_decomposition'] = a
    # ---------------- (b) operating-point part, seed-level interval ----------------
    print('=' * 78); print('(b) operating-point part of C-A: frozen minus each-arm-oracle, per seed (r0 sweeps)'); print('=' * 78)
    b = {}
    for name, (dmain, dthr, suf) in PAIRS.items():
        fr, orc = {}, {}
        for tag in ('Aplain', 'Cconfidence'):
            for s in SEEDS:
                j = L(R / dthr / f'{tag}_seed{s}{suf}.json'); fr[(tag, s)] = j['iou_frozen']; orc[(tag, s)] = j['iou_oracle_wide']
        fca = np.array([fr[('Cconfidence', s)] - fr[('Aplain', s)] for s in SEEDS]); oca = np.array([orc[('Cconfidence', s)] - orc[('Aplain', s)] for s in SEEDS])
        dd = fca - oca; share = 1 - oca / fca
        rec = {'frozen_CA_mean': float(fca.mean()), 'oracle_CA_mean': float(oca.mean()), 'op_part_mean': float(dd.mean()), 'op_part_ci95': ci(dd),
               'op_part_positive': int((dd > 0).sum()), 'share_of_means': float(1 - oca.mean() / fca.mean()),
               'share_per_seed_median': float(np.median(share)), 'share_per_seed_iqr': [float(np.percentile(share, 25)), float(np.percentile(share, 75))],
               'share_per_seed_min_max': [float(share.min()), float(share.max())]}
        b[name] = rec
        print(f"  {name:22s} frozen {rec['frozen_CA_mean']:+.4f} oracle {rec['oracle_CA_mean']:+.4f} | op part {rec['op_part_mean']:+.4f} {[round(x, 4) for x in rec['op_part_ci95']]} {rec['op_part_positive']}/12 | share of means {rec['share_of_means']:.3f}, per-seed median {rec['share_per_seed_median']:.2f} IQR {[round(x, 2) for x in rec['share_per_seed_iqr']]}")
    out['b_operating_point_part'] = b
    # ---------------- (c) amplification on raw and logit scale ----------------
    print('=' * 78); print('(c) seed-SD amplification, raw IoU vs logit(IoU)'); print('=' * 78)
    lg = lambda x: np.log(np.asarray(x) / (1 - np.asarray(x)))
    ind = L(R / 'stage1_train_indomain_summary.json')['arms']
    c = {}
    for tag in ('Aplain', 'Bplain', 'Cconfidence'):
        ps = ind[tag]['per_seed']; indom = np.array([ps[str(s)] if isinstance(ps, dict) else ps[i] for i, s in enumerate(SEEDS)])
        for dom, d in (('Inria', 'stage1'), ('Mass', 'mass')):
            if tag == 'Cconfidence':
                cr = np.array([np.mean([target_iou(L(R / d / f'{tag}_seed{s}_fmin0.01_K4_r{r}.json')) for r in range(5)]) for s in SEEDS])
            else:
                cr = np.array([target_iou(L(R / d / f'{tag}_seed{s}_fmin0.01_K4_r0.json')) for s in SEEDS])
            rec = {'indomain_mean': float(indom.mean()), 'indomain_sd': float(indom.std(ddof=1)), 'cross_mean': float(cr.mean()), 'cross_sd': float(cr.std(ddof=1)),
                   'ratio_raw': float(cr.std(ddof=1) / indom.std(ddof=1)), 'ratio_logit': float(lg(cr).std(ddof=1) / lg(indom).std(ddof=1))}
            c[f'{tag}_{dom}'] = rec
            print(f"  {tag:12s} {dom:6s} raw {rec['ratio_raw']:5.1f}x  logit {rec['ratio_logit']:5.1f}x")
    # second bed, if per-seed in-domain values are recorded
    try:
        w = L(R / 'w_summary.json')['per_arch']
        for arch in ('deeplabv3_r50', 'segformer_b1'):
            pa = w[arch]; indom = None
            for k in ('indomain_per_seed', 'in_domain_per_seed', 'indomain'):
                v = pa.get(k)
                if isinstance(v, dict) and 'per_seed' in v: v = v['per_seed']
                if isinstance(v, (list, dict)) and len(v) == 12:
                    indom = np.array([v[str(s)] if isinstance(v, dict) else v[i] for i, s in enumerate(SEEDS)]); break
            if indom is None: print(f'  {arch}: no per-seed in-domain values in w_summary.json; skipped'); continue
            for dom, d, key in (('Inria', 'w', '_fmin0.01_K4_r0.json'), ('Mass', 'w_mass', '_fmin0.01_K4_r0.json')):
                cr = np.array([target_iou(L(R / d / f'{arch}_seed{s}{key}')) for s in SEEDS])
                rec = {'indomain_sd': float(indom.std(ddof=1)), 'cross_sd': float(cr.std(ddof=1)), 'ratio_raw': float(cr.std(ddof=1) / indom.std(ddof=1)), 'ratio_logit': float(lg(cr).std(ddof=1) / lg(indom).std(ddof=1))}
                c[f'{arch}_{dom}'] = rec; print(f"  {arch:12s} {dom:6s} raw {rec['ratio_raw']:5.1f}x  logit {rec['ratio_logit']:5.1f}x")
    except Exception as e:
        print('  second bed skipped:', e)
    out['c_amplification'] = c
    # ---------------- (d) the two MDE values on Inria ----------------
    print('=' * 78); print('(d) Inria primary contrast: pilot MDE vs 12-seed MDE'); print('=' * 78)
    st = L(R / 'stage1_summary.json')['summary']['K4_C_minus_A']
    s_d12 = float(st['sd_d']); s_d_pilot = 0.016587
    d = {'pilot_s_d_stage0b': s_d_pilot, 'pilot_MDE_n12': mde(s_d_pilot), 'twelve_seed_s_d': s_d12, 'twelve_seed_MDE_n12': mde(s_d12), 'effect_K4_CA': float(st['effect']),
         'MDE_in_se_units_n12': float(stats.t.ppf(0.975, 11) + stats.t.ppf(0.8, 11)), 't_crit_n12': float(T975)}
    out['d_inria_mde'] = d
    print(f"  pilot s_d {s_d_pilot:.6f} -> MDE {d['pilot_MDE_n12']:.4f};  12-seed s_d {s_d12:.6f} -> MDE {d['twelve_seed_MDE_n12']:.4f};  effect {d['effect_K4_CA']:+.4f} (below both)")
    print(f"  MDE = {d['MDE_in_se_units_n12']:.3f} standard errors at n=12; a 95% interval excludes zero from {d['t_crit_n12']:.3f} standard errors")
    # ---------------- (e) AB.2 per-draw decomposition on Massachusetts ----------------
    print('=' * 78); print('(e) AB.2: Massachusetts operating-point decomposition per support draw'); print('=' * 78)
    D = R / 'thrdiag_mass_draws'
    files = [D / f'Cconfidence_seed{s}_r{r}.json' for s in SEEDS for r in (1, 2, 3, 4)]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        print(f'  G3: {len(missing)}/48 draw sweeps missing; (e) skipped'); out['e_per_draw'] = {'status': 'pending', 'missing': len(missing)}
    else:
        exp = {}
        for line in (R / 'stage1_ckpt_sha256.txt').read_text(encoding='utf-8').splitlines():
            parts = line.split()
            if len(parts) >= 2: exp[pathlib.Path(parts[-1]).name] = parts[0]
        g1 = g2 = 0; hashed = {}
        for s in SEEDS:
            for r in (1, 2, 3, 4):
                j = L(D / f'Cconfidence_seed{s}_r{r}.json'); ck = pathlib.Path(j['checkpoint'])
                if ck not in hashed: hashed[ck] = hashlib.sha256(ck.read_bytes()).hexdigest() if ck.exists() else None
                u = L(R / 'mass' / f'Cconfidence_seed{s}_fmin0.01_K4_r{r}.json')
                if not (hashed[ck] == exp[f'Cconfidence_seed{s}.pth'] == u['checkpoint_sha256']): g1 += 1; print(f'  G1 FAIL seed{s} r{r}')
                if abs(j['iou_frozen'] - u['overall']['iou']) >= 5e-4: g2 += 1; print(f"  G2 FAIL seed{s} r{r}: {j['iou_frozen']:.6f} vs {u['overall']['iou']:.6f}")
        if g1 or g2:
            print(f'  GATES FAILED (G1 {g1}, G2 {g2}); stop'); out['e_per_draw'] = {'status': 'gate-failed', 'G1_fail': g1, 'G2_fail': g2}
            (R / 'ab_review_summary.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8'); sys.exit(1)
        print('  gates G1-G3 passed (48 files; checkpoint digests match stage1_ckpt_sha256.txt and appendix U; iou_frozen agrees with appendix U within 5e-4)')
        A_fr = {s: L(R / 'thrdiag_mass' / f'Aplain_seed{s}.json') for s in SEEDS}
        e = {'gates': {'G1_fail': 0, 'G2_fail': 0, 'n_files': 48}, 'per_draw': {}}
        for r in range(5):
            C = {s: L(R / 'thrdiag_mass' / f'Cconfidence_seed{s}.json') if r == 0 else L(D / f'Cconfidence_seed{s}_r{r}.json') for s in SEEDS}
            fca = np.array([C[s]['iou_frozen'] - A_fr[s]['iou_frozen'] for s in SEEDS]); oca = np.array([C[s]['iou_oracle_wide'] - A_fr[s]['iou_oracle_wide'] for s in SEEDS])
            dd = fca - oca; cshare = np.array([1 - C[s]['iou_frozen'] / C[s]['iou_oracle_wide'] for s in SEEDS])
            rec = {'frozen_CA_mean': float(fca.mean()), 'frozen_CA_ci95': ci(fca), 'frozen_CA_positive': int((fca > 0).sum()),
                   'oracle_CA_mean': float(oca.mean()), 'oracle_CA_ci95': ci(oca), 'oracle_CA_positive': int((oca > 0).sum()),
                   'op_part_mean': float(dd.mean()), 'op_part_ci95': ci(dd), 'op_part_positive': int((dd > 0).sum()), 'share_of_means': float(1 - oca.mean() / fca.mean()),
                   'C_calibration_share_mean': float(cshare.mean()), 'C_oracle_threshold_median': float(np.median([C[s]['oracle_threshold'] for s in SEEDS]))}
            e['per_draw'][f'r{r}'] = rec
            print(f"  r{r}: frozen C-A {rec['frozen_CA_mean']:+.4f} ({rec['frozen_CA_positive']}/12) | oracle C-A {rec['oracle_CA_mean']:+.4f} {[round(x, 4) for x in rec['oracle_CA_ci95']]} ({rec['oracle_CA_positive']}/12) | op part {rec['op_part_mean']:+.4f} {[round(x, 4) for x in rec['op_part_ci95']]} | share {rec['share_of_means']:.3f} | C share {rec['C_calibration_share_mean']:.3f}")
        sh = [e['per_draw'][f'r{r}']['share_of_means'] for r in range(5)]; oc = [e['per_draw'][f'r{r}']['oracle_CA_mean'] for r in range(5)]
        e['share_range_over_draws'] = [min(sh), max(sh)]; e['oracle_CA_range_over_draws'] = [min(oc), max(oc)]
        print(f'  share of means over the five draws: {min(sh):.3f}-{max(sh):.3f}; oracle C-A over draws: {min(oc):+.4f}..{max(oc):+.4f}')
        out['e_per_draw'] = e
    (R / 'ab_review_summary.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n  -> results/ab_review_summary.json')


if __name__ == '__main__':
    main()
