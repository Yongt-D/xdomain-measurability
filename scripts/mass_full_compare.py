# -*- coding: utf-8 -*-
"""
Appendix V analysis: comparison of U (501 patches) with V (full tiles) -- V-H1..V-H4 and V-E1. **Committed before any V number existed.**
Inputs: results/mass_summary.json (U), results/mass_full_summary.json (V, produced by mass_analyze.py --dir mass_full),
        results/thrdiag_mass_summary.json (U.9), results/thrdiag_mass_full_summary.json (operating-point diagnostic of V, optional).
"""
import json, sys, pathlib
import numpy as np
from scipy import stats
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
L = lambda p: json.loads(pathlib.Path(p).read_text(encoding='utf-8'))


def main():
    U = L(ROOT / 'results' / 'mass_summary.json'); V = L(ROOT / 'results' / 'mass_full_summary.json')
    out = {'note': 'appendix V; descriptive; no significance claims'}
    if V.get('stop_loss', {}).get('triggered'):
        print('V stop-loss triggered: descriptive only, not judged'); out['verdict'] = {'stop_loss': True}
    uCA, vCA = U['main']['K4_C_minus_A'], V['main']['K4_C_minus_A']
    mdeV = V['main']['MDE_K4_CA']
    print('=' * 76); print('[V-H1] primary criterion holds (K=4, C-A, R=5 mean)'); print('=' * 76)
    print(f"  U: {uCA['effect']:+.4f} [{uCA['ci95_descriptive'][0]:+.4f}, {uCA['ci95_descriptive'][1]:+.4f}] positive {uCA['direction_positive']}/12  MDE_U {U['main']['MDE_K4_CA']:.4f}")
    print(f"  V: {vCA['effect']:+.4f} [{vCA['ci95_descriptive'][0]:+.4f}, {vCA['ci95_descriptive'][1]:+.4f}] positive {vCA['direction_positive']}/12  MDE_V {mdeV:.4f}")
    h1 = bool(vCA['direction_positive'] >= 10 and vCA['effect'] >= mdeV)
    print(f"  -> {'holds' if h1 else 'does not hold (reported as is)'}")
    print('\n' + '=' * 76); print('[V-H2] per-seed agreement (Spearman U vs V)'); print('=' * 76)
    rho_ca = float(stats.spearmanr(uCA['per_seed'], vCA['per_seed']).correlation)
    rho_arm = {a: float(stats.spearmanr(U['main']['K4_arms'][a]['per_seed'], V['main']['K4_arms'][a]['per_seed']).correlation) for a in 'ABC'}
    print(f"  C-A per-seed rho = {rho_ca:+.3f} (criterion >= 0.7); per-arm IoU per-seed rho: A {rho_arm['A']:+.3f}  B {rho_arm['B']:+.3f}  C {rho_arm['C']:+.3f}")
    h2 = bool(rho_ca >= 0.7)
    print('\n' + '=' * 76); print('[V-E1] per-arm mean and seed-SD, U -> V'); print('=' * 76)
    e1 = {}
    for a in 'ABC':
        u, v = U['main']['K4_arms'][a], V['main']['K4_arms'][a]
        e1[a] = {'U_mean': u['mean'], 'V_mean': v['mean'], 'U_sd': u['sd'], 'V_sd': v['sd']}
        print(f"  {a}: mean U {u['mean']:.4f} -> V {v['mean']:.4f} (d {v['mean']-u['mean']:+.4f});  seed-SD U {u['sd']:.4f} -> V {v['sd']:.4f} (ratio {v['sd']/u['sd']:.2f})")
    print('\n' + '=' * 76); print('[V-H4] stratified orderings hold (K4_r0; 1 px band)'); print('=' * 76)
    sU, sV = U['H']['strata'], V['H']['strata']
    c_size, c_int1, c_int4, c_near = sV['C_1px']['count_S4_lt_S1'], sV['C_1px']['count_int_lt_edge'], sV['C_4px']['count_int_lt_edge'], sV['C_1px']['count_near_gt0']
    print(f"  S4<S1: U {sU['C_1px']['count_S4_lt_S1']}/12 -> V {c_size}/12;  interior<edge: U {sU['C_1px']['count_int_lt_edge']}/{sU['C_4px']['count_int_lt_edge']} -> V {c_int1}/{c_int4} (1px/4px);  near-band dFPR>0: U {sU['C_1px']['count_near_gt0']}/12 -> V {c_near}/12")
    h4 = bool(c_size >= 9 and c_int1 >= 9 and c_int4 >= 9 and c_near >= 9)
    h3 = None
    tv = ROOT / 'results' / 'thrdiag_mass_full_summary.json'
    if tv.exists():
        T = L(tv); tU = L(ROOT / 'results' / 'thrdiag_mass_summary.json')
        print('\n' + '=' * 76); print('[V-H3] operating-point decomposition holds'); print('=' * 76)
        print(f"  U.9: C-A at oracle {tU['CA_oracle']['effect']:+.4f}, share {tU['operating_point_share']:.3f};  V: C-A at oracle {T['CA_oracle']['effect']:+.4f} (MDE_V {mdeV:.4f}), share {T['operating_point_share']:.3f}")
        h3 = bool(T['CA_oracle']['effect'] < mdeV and T['operating_point_share'] >= 0.5)
        out['H3'] = {'CA_oracle_V': T['CA_oracle']['effect'], 'share_V': T['operating_point_share']}
    else:
        print('\n  (operating-point diagnostic of V not generated yet; V-H3 pending)')
    print('\n' + '=' * 76); print('[verdicts]'); print('=' * 76)
    print(f"  V-H1 primary criterion holds      -> {'holds' if h1 else 'does not hold'}")
    print(f"  V-H2 per-seed agreement  rho={rho_ca:+.3f} -> {'supported' if h2 else 'not supported'}")
    print(f"  V-H3 operating point holds        -> {('holds' if h3 else 'does not hold') if h3 is not None else 'pending'}")
    print(f"  V-H4 stratified orderings hold    -> {'holds' if h4 else 'does not hold'}")
    out.update({'H1': h1, 'H2': {'rho_CA': rho_ca, 'rho_arms': rho_arm, 'pass': h2}, 'H4': h4, 'H3_pass': h3, 'E1': e1,
                'U_CA': uCA['effect'], 'V_CA': vCA['effect'], 'MDE_V': mdeV})
    (ROOT / 'results' / 'mass_full_compare.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n  -> results/mass_full_compare.json')


if __name__ == '__main__':
    main()
