# -*- coding: utf-8 -*-
"""Each arm's OWN recall loss from f=1 to f=2 by stratum (descriptive; added with all results known).

Separates the general effect of the down-sample-and-resize input (where every arm, the baseline included,
loses recall) from the extra loss of the geometry-bearing arms relative to the baseline that Appendix T /
results/rq5_summary.json reports.  Reads the committed per-stratum files results/rq5/{arm}_seed{s}_gsd{f}.json
and writes results/rq5_own_loss_summary.json.  No verdict is attached.
"""
import json, io, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEEDS = range(100, 112)
ARMS = {'A': 'Aplain', 'B': 'Bplain', 'C': 'Cconfidence'}


def recall_by_stratum(path):
    d = json.load(io.open(path, encoding='utf-8'))
    n = np.zeros(10); p = np.zeros(10)
    for r in d['rows']:
        n += r['n']; p += r['p1']
    return p, n, d['strata_names']


def pooled(ix, p, n):
    return p[ix].sum() / n[ix].sum()


def main():
    out = {'note': 'own recall loss R(f=1) - R(f=2) per arm and stratum group, pooled over query pixels; '
                   'counts = seeds with loss(S4) > loss(S1) and loss(large interior) > loss(large edge); descriptive',
           'arms': {}}
    for arm, pref in ARMS.items():
        rows = []
        for s in SEEDS:
            p1, n1, names = recall_by_stratum(ROOT / 'results' / 'rq5' / f'{pref}_seed{s}_gsd1.json')
            p2, n2, _ = recall_by_stratum(ROOT / 'results' / 'rq5' / f'{pref}_seed{s}_gsd2.json')
            assert np.array_equal(n1, n2), 'label gate: stratum totals differ'
            g = {'S1': [i for i, nm in enumerate(names) if nm.startswith('S1')],
                 'S4': [i for i, nm in enumerate(names) if nm.startswith('S4')],
                 'large_edge': [i for i, nm in enumerate(names) if nm[:2] in ('S3', 'S4') and nm.endswith('edge')],
                 'large_int': [i for i, nm in enumerate(names) if nm[:2] in ('S3', 'S4') and nm.endswith('int')]}
            rows.append({k: pooled(ix, p1, n1) - pooled(ix, p2, n2) for k, ix in g.items()})
        mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        out['arms'][arm] = {'mean_loss': mean,
                            'count_S4_gt_S1': int(sum(r['S4'] > r['S1'] for r in rows)),
                            'count_int_gt_edge': int(sum(r['large_int'] > r['large_edge'] for r in rows)),
                            'per_seed': rows}
        print(f"{arm}: S1 {mean['S1']:.3f} S4 {mean['S4']:.3f} ({out['arms'][arm]['count_S4_gt_S1']}/12); "
              f"large edge {mean['large_edge']:.3f} interior {mean['large_int']:.3f} ({out['arms'][arm]['count_int_gt_edge']}/12)")
    (ROOT / 'results' / 'rq5_own_loss_summary.json').write_text(json.dumps(out, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
