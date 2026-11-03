# -*- coding: utf-8 -*-
"""
Appendix X analysis: can a support-calibrated threshold recover the operating-point loss (X-H1), and is C-A still measurable after calibration (X-H2)? **Committed before any of its numbers existed.**
Inputs results/x/{domain}_{tag}_seed{s}.json; MDE_mass from results/mass_summary.json; MDE_inria from results/stage1_summary.json (0.0147, the R.4 value, if absent).
Nature: diagnostic; not in the results table; the support self-calibration of arm C is optimistic (disclosed).
"""
import json, sys, pathlib
import numpy as np
from scipy import stats
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
SEEDS = list(range(100, 112)); TAGS = ('Aplain', 'Bplain', 'Cconfidence'); SHORT = {'Aplain': 'A', 'Bplain': 'B', 'Cconfidence': 'C'}
L = lambda p: json.loads(pathlib.Path(p).read_text(encoding='utf-8'))


def paired(d):
    d = np.asarray(d, float); n = len(d); m = float(d.mean()); sd = float(d.std(ddof=1)); half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return {'n': n, 'effect': m, 'sd_d': sd, 'ci95_descriptive': [m - half, m + half], 'direction_positive': int((d > 0).sum()), 'per_seed': d.tolist()}


def main():
    mde = {'mass': L(ROOT / 'results' / 'mass_summary.json')['main']['MDE_K4_CA']}
    try:
        st = L(ROOT / 'results' / 'stage1_summary.json'); mde['inria'] = st['main']['MDE_K4_CA'] if 'main' in st else 0.0147
    except Exception: mde['inria'] = 0.0147
    out = {'note': 'appendix X diagnostic; not in the results table; the support self-calibration of arm C is optimistic', 'MDE': mde, 'domains': {}}
    for dom in ('inria', 'mass'):
        print('=' * 78); print(f'[{dom}] frozen / support-calibrated (K=4, mean of 5 draws) / oracle; recovered fraction rho_rec = (sup - frozen)/(oracle - frozen)'); print('=' * 78)
        D = {}
        for tag in (TAGS if dom == 'mass' else ('Aplain', 'Cconfidence')):     # the Inria widened diagnostic exists for A/C only (no B in results/thrdiag_stage1), so B cannot be computed on Inria
            fr, su, orc, rec, tsd, isd, tsup = [], [], [], [], [], [], []
            for s in SEEDS:
                d = L(ROOT / 'results' / 'x' / f'{dom}_{tag}_seed{s}.json')
                q = np.array([x['query_iou_at_t_sup'] for x in d['draws']]); t = np.array([x['t_sup'] for x in d['draws']])
                fr.append(d['iou_frozen']); orc.append(d['iou_oracle']); su.append(float(q.mean())); tsd.append(float(t.std(ddof=1))); isd.append(float(q.std(ddof=1))); tsup.append(float(np.median(t)))
                gain = d['iou_oracle'] - d['iou_frozen']; rec.append((q.mean() - d['iou_frozen']) / gain if gain > 1e-9 else float('nan'))
            fr, su, orc, rec = map(np.array, (fr, su, orc, rec))
            D[SHORT[tag]] = {'iou_frozen': fr.tolist(), 'iou_support_cal': su.tolist(), 'iou_oracle': orc.tolist(), 'rho_rec': rec.tolist(), 't_sup_median': tsup,
                             'sd_t_sup_over_draws': tsd, 'sd_iou_over_draws': isd, 'count_rho_rec_ge_0.5': int(np.nansum(rec >= 0.5))}
            print(f"  {SHORT[tag]}: frozen {fr.mean():.4f} -> support-calibrated {su.mean():.4f} -> oracle {orc.mean():.4f}; rho_rec median {np.nanmedian(rec):.2f}, >=0.5 {int(np.nansum(rec >= 0.5))}/12; t_sup median {np.median(tsup):.2f}; SD over draws: t {np.mean(tsd):.3f} IoU {np.mean(isd):.4f}")
        ca_f = paired(np.array(D['C']['iou_frozen']) - np.array(D['A']['iou_frozen'])); ca_s = paired(np.array(D['C']['iou_support_cal']) - np.array(D['A']['iou_support_cal'])); ca_o = paired(np.array(D['C']['iou_oracle']) - np.array(D['A']['iou_oracle']))
        ba_s = paired(np.array(D['B']['iou_support_cal']) - np.array(D['A']['iou_support_cal'])) if 'B' in D else None
        for nm, pr in [('C-A frozen', ca_f), ('C-A support-cal.', ca_s), ('C-A oracle', ca_o)] + ([('B-A support-cal.', ba_s)] if ba_s else []):
            print(f"  {nm:18s} effect {pr['effect']:+.4f}  CI [{pr['ci95_descriptive'][0]:+.4f}, {pr['ci95_descriptive'][1]:+.4f}]  positive {pr['direction_positive']}/12  (MDE_{dom} {mde[dom]:.4f})")
        out['domains'][dom] = {'arms': D, 'CA_frozen': ca_f, 'CA_support_cal': ca_s, 'CA_oracle': ca_o, 'BA_support_cal': ba_s}
    print('\n' + '=' * 78); print('[verdicts]'); print('=' * 78)
    m = out['domains']['mass']
    h1 = m['arms']['A']['count_rho_rec_ge_0.5'] >= 9
    ca = m['CA_support_cal']; h2 = 'gone' if ca['effect'] < mde['mass'] else ('independent' if ca['direction_positive'] >= 10 else 'unresolved')
    print(f"  X-H1  rho_rec >= 0.5 for A on Mass: {m['arms']['A']['count_rho_rec_ge_0.5']}/12 -> {'few-shot calibration recovers most of the loss' if h1 else 'does not hold (reported as is)'}")
    h2_txt = {'gone': 'the gain disappears with calibration', 'independent': 'the gain is partly independent of calibration', 'unresolved': 'below the direction count; not judged'}[h2]
    print(f"  X-H2  C-A on Mass after calibration = {ca['effect']:+.4f} vs MDE {mde['mass']:.4f}, {ca['direction_positive']}/12 -> {h2_txt}")
    out['verdict'] = {'X-H1': h1, 'X-H2': h2}
    (ROOT / 'results' / 'x_summary.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8'); print('\n  -> results/x_summary.json')


if __name__ == '__main__':
    main()
