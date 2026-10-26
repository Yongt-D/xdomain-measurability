# -*- coding: utf-8 -*-
"""
Appendix R.3 (M2): counter-example rate of the single-seed protocol and pseudo-replication significance rate -- computed from the **existing** stage-1 and L2 data.
**Committed before it produced any number.**

With the sign of the 12-seed mean effect as reference (section 3.2: independent unit = training seed), quantify what the two practices forbidden by section 3.2 would conclude:
  rho_sign    : number of seeds whose single-seed paired-difference sign is opposite to the 12-seed reference / 12
  (a) draw level : within one seed, treat the R=5 support draws as samples and run a one-sample t-test on C_r - A (alpha=0.05) -- pseudo-replication
  (b) image level: within one seed, treat the 2997 query images as samples and run a paired t-test on per-image (IoU_C - IoU_A) (alpha=0.05) -- pseudo-replication
and count the seeds that are "significant but opposite to the 12-seed sign".

The p-values here are **the object of criticism**, not a claim of this study; the primary criterion makes no significance claim (D.4).
The list of comparisons is fixed in Appendix R.3: C-A and C-B at K in {1,4,8} (L1); C-A at the 4 GSD points (L2).
(a)(b) are computed at K=4 only ((b) uses r0); L2 has no R draws and no extension beyond the per-image A-arm pairing.
Reading rule (fixed in R.3): if every rho_sign <= 1/12 and the "significant but opposite" count is 0 => M2 has no teeth in this setting; report as such.
"""
import argparse, json, sys, pathlib
import numpy as np
from scipy import stats
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
SEEDS = list(range(100, 112))
ALPHA = 0.05


def L(p):
    return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))


def l1(tag, s, k, r=0):
    return L(ROOT / 'results' / 'stage1' / f'{tag}_seed{s}_fmin0.01_K{k}_r{r}.json')['target']['iou']


def l1_seed(tag, s, k):
    rs = range(5) if tag == 'Cconfidence' else range(1)
    return float(np.mean([l1(tag, s, k, r) for r in rs]))


def l2(tag, s, f):
    return L(ROOT / 'results' / 'l2' / f'{tag}_seed{s}_gsd{f}.json')['target']['iou']


def per_sample(tag, s, k=4, r=0):
    rows = L(ROOT / 'results' / 'stage1' / f'{tag}_seed{s}_fmin0.01_K{k}_r{r}.per_sample.json')
    return {x['name']: x['iou'] for x in rows}


def row(name, d_per_seed):
    d = np.asarray(d_per_seed, float)
    ref = np.sign(d.mean())
    rho = int((np.sign(d) == -ref).sum())
    return ref, rho, float(d.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'results' / 'm2_single_seed_counterexample.json'))
    a = ap.parse_args()
    out = {'seeds': SEEDS, 'alpha': ALPHA, 'rows': [],
           'note': 'the p-values are the object of criticism (pseudo-replication), not a claim; reference = sign of the 12-seed mean effect'}
    print('Appendix R.3 (M2): counter-example rate rho_sign of the single-seed protocol (seeds with the opposite sign / 12)')
    print(f"  {'comparison':22s}{'12-seed effect':>16s}{'ref sign':>10s}{'rho_sign':>10s}")
    comps = []
    for k in (1, 4, 8):
        comps.append((f'L1 C-A K={k}', [l1_seed('Cconfidence', s, k) - l1_seed('Aplain', s, k) for s in SEEDS]))
        comps.append((f'L1 C-B K={k}', [l1_seed('Cconfidence', s, k) - l1_seed('Bplain', s, k) for s in SEEDS]))
    for f in (1, 2, 4, 8):
        comps.append((f'L2 C-A gsd{f}', [l2('Cconfidence', s, f) - l2('Aplain', s, f) for s in SEEDS]))
    for name, d in comps:
        ref, rho, m = row(name, d)
        print(f'  {name:22s}{m:>+16.6f}{int(ref):>10d}{rho:>7d}/12')
        out['rows'].append({'comparison': name, 'effect_12seed': m, 'ref_sign': int(ref), 'rho_sign_count': rho,
                            'per_seed': list(map(float, d))})

    print('\npseudo-replication significance rate (K=4; alpha=0.05; **the criticised practice**)')
    print(f"  {'seed':>5s}{'(a) draw-level t (R=5)':>24s}{'p':>10s}{'(b) image-level paired t (2997)':>32s}{'p':>10s}{'sign':>6s}")
    ref_ca = np.sign(np.mean([l1_seed('Cconfidence', s, 4) - l1_seed('Aplain', s, 4) for s in SEEDS]))
    a_sig = b_sig = a_wrong = b_wrong = 0
    for s in SEEDS:
        # (a) support-draw level: each of C's 5 draws minus A (A is draw-invariant, F.3)
        d_a = np.array([l1('Cconfidence', s, 4, r) for r in range(5)]) - l1('Aplain', s, 4, 0)
        t_a, p_a = stats.ttest_1samp(d_a, 0.0)
        # (b) image level: per-image IoU difference (r0)
        pc, pa = per_sample('Cconfidence', s), per_sample('Aplain', s)
        names = sorted(pc)
        d_b = np.array([pc[n] - pa[n] for n in names])
        t_b, p_b = stats.ttest_rel([pc[n] for n in names], [pa[n] for n in names])
        sa, sb = bool(p_a < ALPHA), bool(p_b < ALPHA)
        wa = bool(sa and np.sign(d_a.mean()) == -ref_ca)
        wb = bool(sb and np.sign(d_b.mean()) == -ref_ca)
        a_sig += sa; b_sig += sb; a_wrong += wa; b_wrong += wb
        print(f'  {s:>5d}{d_a.mean():>+14.6f}{"sig." if sa else "  - ":>10s}{p_a:>10.3g}'
              f'{d_b.mean():>+20.6f}{"sig." if sb else "  - ":>12s}{p_b:>10.3g}{int(np.sign(d_b.mean())):>6d}')
        out.setdefault('pseudo_replication', []).append({'seed': s, 'a_mean': float(d_a.mean()), 'a_p': float(p_a),
                                                          'b_mean': float(d_b.mean()), 'b_p': float(p_b),
                                                          'b_n': int(d_b.size)})
    print(f'  (a) significant {a_sig}/12, of which significant but opposite to the 12-seed sign {a_wrong}/12')
    print(f'  (b) significant {b_sig}/12, of which significant but opposite to the 12-seed sign {b_wrong}/12')
    out['pseudo_summary'] = {'a_sig': int(a_sig), 'a_wrong_sign_sig': int(a_wrong), 'b_sig': int(b_sig),
                             'b_wrong_sign_sig': int(b_wrong), 'ref_sign_CA_K4': int(ref_ca)}
    max_rho = max(r['rho_sign_count'] for r in out['rows'])
    toothless = bool(max_rho <= 1 and a_wrong == 0 and b_wrong == 0)
    out['M2_toothless_by_R3_rule'] = toothless
    print(f'\nR.3 reading rule: max rho_sign = {max_rho}/12, significant-but-opposite = {a_wrong + b_wrong}'
          f' => {"**M2 has no teeth in this setting; report as such**" if toothless else "M2 has evidence (descriptive proportions, no significance claim)"}')
    pathlib.Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'  -> {a.out}')


if __name__ == '__main__':
    main()
