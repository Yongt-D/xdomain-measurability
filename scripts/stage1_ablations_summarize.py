# -*- coding: utf-8 -*-
"""
Appendix F.1: summary and verdict of the five mandatory causal ablations of section 4 on the 12 stage-1 C+conf checkpoints.
**Committed before the ablation produced any number.**

Criteria (section 4 / F.2 / G.2 verbatim; thresholds unchanged; written per seed into `verdict` by causal_ablations.py):
  (1)  zeroing the geometry (DGM residual) changes more than 0.5% of the output pixels
  (2a) mismatched support masks => IoU must be lower than with matched supports
  (2b) background-only supports => IoU must be lower than with matched supports
  (3)  mean normalised entropy of a_k in [0.05, 0.995] and SD > 1e-4 (already failed in 0b; H.3 U3 not reopened; reported as is)
  (4)  per-image variance of the relative DGM residual magnitude Var/mean^2 > 1e-8
  (5)  C+conf must beat the parameter-matched random prototype
Aggregation rule (fixed in advance, as strict as the 0b verdict): **a criterion passes only if all 12 seeds satisfy it**;
one failing seed marks it as failed.

Also reported per criterion: effect size (IoU difference base − intervention), seed SD, descriptive 95% CI and direction
consistency; **no p-values** (D.4).
Provenance: the checkpoint_sha256 of every result must equal the value for the same seed in results/stage1_ckpt_sha256.txt;
the threshold must equal the L1 source_val_threshold; otherwise stop.
"""
import argparse, json, sys, pathlib
import numpy as np
from scipy import stats
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
CRIT = [('1_geometry_zeroed_changes_output', '(1) zeroed geometry changes output (>0.5% pixels)'),
        ('2a_mask_mismatch_hurts', '(2a) mismatched masks must hurt'),
        ('2b_background_support_hurts', '(2b) background-only supports must hurt'),
        ('3_attention_entropy_nondegenerate', '(3) a_k entropy non-degenerate'),
        ('4_geometry_signal_varies_across_images', '(4) geometry signal varies across images (>>1e-8)'),
        ('5_beats_param_matched_random_prototype', '(5) beats parameter-matched random prototype')]


def ci(d):
    d = np.asarray(d, float); n = d.size
    m, sd = float(d.mean()), float(d.std(ddof=1))
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return m, sd, m - half, m + half, int((d > 0).sum()), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default=','.join(str(s) for s in range(100, 112)))
    ap.add_argument('--prefix', default='stage1_Cconfidence')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    shas = {}
    for line in (ROOT / 'results' / 'stage1_ckpt_sha256.txt').read_text(encoding='utf-8').splitlines():
        h, name = line.split()
        shas[name.strip('*')] = h
    R = {}
    for s in seeds:
        p = ROOT / 'results' / 'ablations' / f'{a.prefix}_seed{s}.json'
        if not p.exists():
            print(f'  missing {p}'); sys.exit(1)
        R[s] = json.loads(p.read_text(encoding='utf-8'))
        l1 = json.loads((ROOT / 'results' / 'stage1' / f'Cconfidence_seed{s}_fmin0.01_K4_r0.json').read_text(encoding='utf-8'))
        if R[s]['checkpoint_sha256'] != shas[f'Cconfidence_seed{s}.pth']:
            print(f'  seed{s}: checkpoint sha256 differs from results/stage1_ckpt_sha256.txt => stop'); sys.exit(2)
        if abs(R[s]['threshold'] - l1['source_val_threshold']) > 1e-9:
            print(f'  seed{s}: threshold {R[s]["threshold"]} != L1 frozen threshold {l1["source_val_threshold"]} => stop'); sys.exit(2)
        if R[s]['manifest_key'] != 'fmin0.01_K4_r0':
            print(f'  seed{s}: manifest_key is not fmin0.01_K4_r0 => stop'); sys.exit(2)
    print(f'provenance: sha256 / frozen threshold / manifest all passed (n={len(seeds)})')

    out = {'seeds': seeds, 'criteria': {}, 'effects': {}, 'rule': 'a criterion passes only if all 12 seeds satisfy it; no significance claims'}
    print('\n' + '=' * 76)
    print('[verdicts] the five criteria of section 4 (thresholds verbatim; pass only if all 12 seeds pass)')
    print('=' * 76)
    for key, label in CRIT:
        v = [bool(R[s]['verdict'][key]) for s in seeds]
        ok = all(v)
        fails = [s for s, x in zip(seeds, v) if not x]
        print(f'  {label:52s} passed {sum(v):2d}/{len(v)}  -> {"pass" if ok else "**FAIL**"}'
              f'{"  failing seeds: " + str(fails) if fails else ""}')
        out['criteria'][key] = {'pass_count': sum(v), 'n': len(v), 'all_pass': ok, 'failed_seeds': fails}

    print('\n' + '=' * 76)
    print('[effect sizes] base - intervention (IoU), descriptive CI, direction consistency')
    print('=' * 76)
    for key, label in (('geom_off', '(1) base - geom_off'), ('mismatch_a', '(2a) base - mismatched masks'),
                       ('mismatch_b', '(2b) base - background supports'), ('rand_proto', '(5) base - random prototype')):
        d = [R[s]['iou']['base'] - R[s]['iou'][key] for s in seeds]
        m, sd, lo, hi, pos, n = ci(d)
        print(f'  {label:34s} effect {m:+.6f}   CI [{lo:+.6f}, {hi:+.6f}]   SD {sd:.6f}   positive {pos}/{n}')
        out['effects'][key] = {'effect': m, 'sd_d': sd, 'ci95_descriptive': [lo, hi], 'direction_positive': pos, 'per_seed': d}
    base = np.array([R[s]['iou']['base'] for s in seeds])
    print(f'  base IoU mean {base.mean():.6f}  SD {base.std(ddof=1):.6f}')
    r1 = np.array([R[s]['ablation_1_changed_pixel_ratio'] for s in seeds])
    h = np.array([R[s]['ablation_3_entropy']['mean'] for s in seeds])
    hsd = np.array([R[s]['ablation_3_entropy']['sd'] for s in seeds])
    v4 = np.array([R[s]['ablation_4_relvar_across_images'] for s in seeds])
    print(f'  (1) changed-pixel ratio: mean {r1.mean()*100:.3f}%  [{r1.min()*100:.3f}%, {r1.max()*100:.3f}%]')
    print(f'  (3) mean normalised entropy: {h.mean():.6f}  [{h.min():.6f}, {h.max():.6f}]   entropy SD range [{hsd.min():.2e}, {hsd.max():.2e}]')
    print(f'  (4) Var/mean^2:    median {np.median(v4):.3e}  [{v4.min():.3e}, {v4.max():.3e}]')
    out['effects'].update({'changed_pixel_ratio_per_seed': r1.tolist(), 'entropy_mean_per_seed': h.tolist(),
                           'entropy_sd_per_seed': hsd.tolist(), 'relvar_per_seed': v4.tolist(), 'base_per_seed': base.tolist()})
    outp = pathlib.Path(a.out) if a.out else ROOT / 'results' / 'ablations_stage1_summary.json'
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n  -> {outp}')
    print('\nReminder: (1) is necessary but not sufficient; (5) rules out "the source is capacity", (2) rules out "the source is '
          'noise regularisation" (judgement document on the mandatory causal ablations, section 4). Section 7: if any of (1)-(5) fails, no positive verdict.')


if __name__ == '__main__':
    main()
