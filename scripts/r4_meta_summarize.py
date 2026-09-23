# -*- coding: utf-8 -*-
"""
Summary script of the Appendix R.4 literature screening. **Committed before any paper data were entered.**
Input: results/r4_meta_analysis.csv (entered by hand; one main target-domain comparison per paper and row)
  columns: id, ref, year, source, target, task_type (UDA|DG|source-only|few-shot), metric (IoU|F1), delta (target-domain improvement over the baseline, IoU/F1 on a 0-1 scale),
           seeds_reported (yes|no), n_seeds (integer or empty), sd_or_ci_reported (yes|no), evidence (fulltext|abstract|snippet), note
Yardsticks (fixed in pre-registration R.4): s_d = 0.018398, MDE(n=12) = 0.0147, single-run cross-domain fluctuation 0.013-0.034.
Stop-loss: fewer than 10 included papers => report "insufficient sample" only, no proportion statements.
Boundary: the variance of other papers may differ; the yardsticks are an order-of-magnitude reference only, never a claim that "paper X's result is not significant".
"""
import csv, json, sys, pathlib
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
S_D, MDE, FLUCT_LO, FLUCT_HI = 0.018398, 0.0147, 0.013, 0.034
MIN_N = 10


def main():
    p = ROOT / 'results' / 'r4_meta_analysis.csv'
    rows = [r for r in csv.DictReader(open(p, encoding='utf-8')) if r.get('id') and not r['id'].startswith('#')]
    inc = [r for r in rows if r.get('delta', '').strip() != '']
    print(f'{len(rows)} records; {len(inc)} papers report a target-domain delta (included)')
    out = {'n_screened': len(rows), 'n_included': len(inc), 'yardsticks': {'s_d': S_D, 'MDE_n12': MDE, 'single_run_fluct': [FLUCT_LO, FLUCT_HI]}}
    if len(inc) < MIN_N:
        print(f'  => fewer than {MIN_N} included: **insufficient sample**, no proportion statements (R.4 stop-loss)')
        out['verdict'] = 'insufficient-sample'
    d = np.array([float(r['delta']) for r in inc]) if inc else np.array([])
    seeds = [r['seeds_reported'].strip().lower() == 'yes' for r in inc]
    sdci = [r['sd_or_ci_reported'].strip().lower() == 'yes' for r in inc]
    full = [r['evidence'].strip().lower() == 'fulltext' for r in inc]
    if len(inc):
        print(f'  delta: median {np.median(d):.4f} mean {d.mean():.4f} [{d.min():.4f}, {d.max():.4f}]')
        print(f'  multi-seed reported: {sum(seeds)}/{len(inc)}; SD/CI reported: {sum(sdci)}/{len(inc)}; checked against the full text: {sum(full)}/{len(inc)}')
        print(f'  delta < MDE ({MDE}): {int((d < MDE).sum())}/{len(inc)}; delta < single-run fluctuation upper bound ({FLUCT_HI}): {int((d < FLUCT_HI).sum())}/{len(inc)}; delta < s_d ({S_D}): {int((d < S_D).sum())}/{len(inc)}')
        out.update({'delta_median': float(np.median(d)), 'delta_mean': float(d.mean()), 'delta_min': float(d.min()), 'delta_max': float(d.max()),
                    'n_seeds_reported': int(sum(seeds)), 'n_sd_ci_reported': int(sum(sdci)), 'n_fulltext': int(sum(full)),
                    'n_delta_lt_MDE': int((d < MDE).sum()), 'n_delta_lt_fluct_hi': int((d < FLUCT_HI).sum()), 'n_delta_lt_sd': int((d < S_D).sum()),
                    'rows': inc})
        if len(inc) >= MIN_N: out['verdict'] = 'reported'
    (ROOT / 'results' / 'r4_meta_summary.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('  -> results/r4_meta_summary.json')
    print('Reminder (R.4 boundary): order-of-magnitude reference only; never write that "the result of paper X is not significant".')


if __name__ == '__main__':
    main()
