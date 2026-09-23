# -*- coding: utf-8 -*-
"""
Full pipeline-fidelity report (extension of Appendix C.4). Uses **in-domain** test IoU only; no cross-domain numbers.
This is a fidelity check, not a result of the study (section 3.3).
"""
import json, sys, pathlib
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
CH4 = {  # chapter-4 retrospective document, section 3.5 (private archive)
    'A':      dict(per_seed=None, mean=0.856343, std=0.003593, src='retrospective section 3.5 table (no matching JSON in the archive)'),
    'C+conf': dict(per_seed=[0.865368, 0.860085, 0.860791], mean=0.862081, std=0.002342,
                   src='results/ch4_reference/v5_confidence_seed{0,1,2}.json'),
}
def ours(tag):
    return [json.loads((ROOT/'outputs'/f'stage0_{tag}_seed{s}'/'results.json')
            .read_text(encoding='utf-8'))['test_metrics']['iou'] for s in (0,1,2)]

a, c = ours('Aplain'), ours('Cconfidence')
print('in-domain WHU-100 test IoU -- this study (dyt/3090) vs chapter 4 (30482)')
print(f"{'':9s} {'seed0':>10s} {'seed1':>10s} {'seed2':>10s} {'mean':>10s}")
print(f"{'A ours':9s} " + ' '.join(f'{v:10.6f}' for v in a) + f' {np.mean(a):10.6f}')
print(f"{'A ch.4':9s} " + ' '.join(f'{"-":>10s}' for _ in range(3)) + f' {CH4["A"]["mean"]:10.6f}')
print(f"{'C ours':9s} " + ' '.join(f'{v:10.6f}' for v in c) + f' {np.mean(c):10.6f}')
print(f"{'C ch.4':9s} " + ' '.join(f'{v:10.6f}' for v in CH4['C+conf']['per_seed']) + f' {CH4["C+conf"]["mean"]:10.6f}')

print('\nper-seed difference (C+conf - A), i.e. the prototype gain claimed in chapter 4:')
d_ours = np.array(c) - np.array(a)
d_ch4  = np.array([0.005464, 0.002383, 0.009368])   # the V5-A column of retrospective section 3.5
print(f"  ours (dyt)  {[round(v,6) for v in d_ours]}  mean {d_ours.mean():+.6f}  positive {int((d_ours>0).sum())}/3")
print(f"  chapter 4   {[round(v,6) for v in d_ch4]}  mean {d_ch4.mean():+.6f}  positive {int((d_ch4>0).sum())}/3")

print('\ndeviations:')
print(f"  arm A mean   d = {np.mean(a)-CH4['A']['mean']:+.6f}   (gate criterion |d| <= 0.01 => pass)")
print(f"  arm C mean   d = {np.mean(c)-CH4['C+conf']['mean']:+.6f}   = {abs(np.mean(c)-CH4['C+conf']['mean'])/CH4['C+conf']['std']:.1f} x the chapter-4 seed SD")
print(f"  arm C seed0  d = {c[0]-CH4['C+conf']['per_seed'][0]:+.6f}  <- the deviation is concentrated in seed0")
print('\nnote: the arm-A reference comes from the retrospective table (no matching JSON in the archive), so its provenance is weaker than arm C (three JSONs).')
