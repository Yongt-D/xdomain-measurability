# -*- coding: utf-8 -*-
"""
Appendix AA / diagnostic Y.6: source-threshold censoring rule and the **leakage-free** wide-grid secondary policy (revision 26). **Committed before any of its numbers existed.**

For every (arm, seed):
  1. sweep 0.05-0.95 (step 0.01, 91 points) on the source Inria validation split (exactly the L1 split: create_dataloaders with the same
     sample_size/sample_seed); the argmax (ties -> the lowest threshold) is t_wide; the same sweep also yields the argmax inside 0.40-0.60,
     which is checked against the L1 frozen threshold t_frozen;
  2. the target IoU is read directly from results/thrdiag_y_{whu,mass}/{tag}_seed{s}.json, iou_by_threshold[t] (K4_r0, same checkpoint, same supports);
     **no new target-domain inference, no target labels touched**;
  3. write results/y_srcwide/{tag}_seed{s}.json; once all 36 exist, write results/y_srcwide_summary.json (effect, CI, direction, MDE and verdict class
     of C-A / B-A under both policies).
Diagnostic only: no hypothesis is judged, nothing enters the main results table, the primary criterion is not replaced (Appendix AA.2 item 3).
"""
import argparse, glob, hashlib, json, pathlib, sys, time
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import eval_crossdomain as E                      # reused: M (model/data), sweep_counts, iou_of, load_checkpoint
from scipy import stats

SEEDS = list(range(100, 112)); TAGS = ('Aplain', 'Bplain', 'Cconfidence'); SHORT = {'Aplain': 'A', 'Bplain': 'B', 'Cconfidence': 'C'}
WIDE = [round(0.05 + 0.01 * i, 2) for i in range(91)]; NARROW = [round(0.40 + 0.01 * i, 2) for i in range(21)]
KEY = 'fmin0.01_K4_r0'


def L(p):
    p = pathlib.Path(p)
    if not p.exists(): print(f'  missing {p}'); sys.exit(1)
    return json.loads(p.read_text(encoding='utf-8'))


def mde(s_d, n=12):
    lo, hi = 1e-5, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2; ncp = mid / (s_d / np.sqrt(n)); crit = stats.t.ppf(0.975, n - 1)
        p = stats.nct.sf(crit, n - 1, ncp) + stats.nct.cdf(-crit, n - 1, ncp)
        if p >= 0.80: hi = mid
        else: lo = mid
    return float(hi)


def paired(d):
    d = np.asarray(d, float); n = len(d); m = float(d.mean()); sd = float(d.std(ddof=1)); half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return {'n': n, 'effect': m, 'sd_d': sd, 'MDE': mde(sd, n), 'ci95_descriptive': [m - half, m + half], 'direction_positive': int((d > 0).sum()), 'per_seed': d.tolist()}


def verdict(pr):
    if abs(pr['effect']) >= pr['MDE'] and (pr['direction_positive'] >= 10 or pr['direction_positive'] <= 2): return 'pos' if pr['effect'] > 0 else 'neg'
    return 'unresolved'


def argmax_lowest(acc, grid):
    best = max(E.iou_of(acc[t]) for t in grid)
    return min(t for t in grid if E.iou_of(acc[t]) == best)      # ties -> the lowest threshold


def sweep_one(tag, seed, a, device):
    out = ROOT / 'results' / 'y_srcwide' / f'{tag}_seed{seed}.json'
    if out.exists(): print(f'skip {out.name}'); return
    ckpt = pathlib.Path(a.ckpt_dir) / f'y_{tag}_seed{seed}.pth'
    model, state, info = E.load_checkpoint(str(ckpt), device); targs = state['args']
    loaders = E.M.create_dataloaders(a.source_view, targs['sample_size'], a.batch_size, targs['sample_seed'], a.num_workers, targs['support_shots'])
    t0 = time.time(); acc = E.sweep_counts(model, loaders[1], device, WIDE)
    t_wide = argmax_lowest(acc, WIDE); t_narrow = argmax_lowest(acc, NARROW)
    l1 = L(ROOT / 'results' / 'y_whu' / f'{tag}_seed{seed}_{KEY}.json'); t_frozen = l1['source_val_threshold']
    sha = hashlib.sha256(open(ckpt, 'rb').read()).hexdigest()
    rec = {'tag': tag, 'seed': seed, 'checkpoint': str(ckpt), 'checkpoint_sha256': sha, 'l1_checkpoint_sha256': l1['checkpoint_sha256'],
           'source_val_iou_by_threshold': {str(t): E.iou_of(acc[t]) for t in WIDE},
           't_frozen': t_frozen, 't_narrow_recomputed': t_narrow, 't_wide': t_wide,
           'censored_frozen': bool(t_frozen in (0.40, 0.60)), 'censored_wide': bool(t_wide in (0.05, 0.95)),
           'source_val_iou_frozen': E.iou_of(acc[t_frozen]), 'source_val_iou_wide': E.iou_of(acc[t_wide]), 'target': {}, 'seconds': time.time() - t0}
    for dom in ('whu', 'mass'):
        th = L(ROOT / 'results' / f'thrdiag_y_{dom}' / f'{tag}_seed{seed}.json')
        rec['target'][dom] = {'thrdiag_sha256': th['checkpoint_sha256'] if 'checkpoint_sha256' in th else None, 'iou_frozen_thrdiag': th['iou_frozen'],
                              'iou_at_frozen': th['iou_by_threshold'][str(t_frozen)], 'iou_at_wide': th['iou_by_threshold'][str(t_wide)]}
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"  {SHORT[tag]} seed{seed}: t_frozen {t_frozen} (recomputed {t_narrow}) t_wide {t_wide}  val {rec['source_val_iou_frozen']:.4f}->{rec['source_val_iou_wide']:.4f}  "
          f"whu {rec['target']['whu']['iou_at_frozen']:.4f}->{rec['target']['whu']['iou_at_wide']:.4f}  mass {rec['target']['mass']['iou_at_frozen']:.4f}->{rec['target']['mass']['iou_at_wide']:.4f}  {rec['seconds']:.0f}s")


def summarize():
    R = {(t, s): L(ROOT / 'results' / 'y_srcwide' / f'{t}_seed{s}.json') for t in TAGS for s in SEEDS}
    print('=' * 78); print('[gate] sha256 consistent; iou_by_threshold[t_frozen] = iou_frozen of the widened diagnostic; recomputed 0.40-0.60 argmax = L1 frozen threshold,'
                             ' or the source-val IoU of the two thresholds differs by < 5e-4 (E1: near-tie flip across machines, same mechanism as Case 3; revision 26 change log, 2026-09-16 second row)'); print('=' * 78)
    E1 = 5e-4
    bad = [(t, s) for (t, s), r in R.items() if r['checkpoint_sha256'] != r['l1_checkpoint_sha256']
           or any(abs(r['target'][d]['iou_at_frozen'] - r['target'][d]['iou_frozen_thrdiag']) > 1e-9 for d in ('whu', 'mass'))]
    flips = {}
    for (t, s), r in R.items():
        if r['t_narrow_recomputed'] != r['t_frozen']:
            v = r['source_val_iou_by_threshold']; gap = abs(v[str(r['t_frozen'])] - v[str(r['t_narrow_recomputed'])])
            flips[f'{SHORT[t]}{s}'] = {'t_frozen': r['t_frozen'], 't_recomputed': r['t_narrow_recomputed'], 'val_gap': gap}
            if gap >= E1: bad.append((t, s))
    if bad: print('  mismatch:', bad); sys.exit(2)
    print(f"  pass; cross-machine re-selection flips {len(flips)}/36: {flips if flips else 'none'}")
    near = {}
    for (t, s), r in R.items():
        v = r['source_val_iou_by_threshold']; m = max(v[str(g)] for g in NARROW); ties = [g for g in NARROW if m - v[str(g)] < 1e-4]
        if len(ties) > 1: near[f'{SHORT[t]}{s}'] = ties
    print(f"  more than one threshold within 1e-4 of the maximum inside 0.40-0.60: {len(near)}/36 (the source-val curve is flat around its optimum)")
    out = {'seeds': SEEDS, 'note': 'appendix AA / Y.6 diagnostic: leakage-free wide-grid secondary policy; no significance claims; does not replace the primary criterion', 'gate_flips': flips, 'near_ties_narrow': near, 'per_arm': {}, 'contrasts': {}}
    print('\n' + '=' * 78); print('[arms] frozen (0.40-0.60) vs wide-grid (0.05-0.95) source thresholds; censoring counts; source-val and target IoU'); print('=' * 78)
    for t in TAGS:
        tf = [R[(t, s)]['t_frozen'] for s in SEEDS]; tw = [R[(t, s)]['t_wide'] for s in SEEDS]
        cf = sum(R[(t, s)]['censored_frozen'] for s in SEEDS); cw = sum(R[(t, s)]['censored_wide'] for s in SEEDS)
        vf = np.mean([R[(t, s)]['source_val_iou_frozen'] for s in SEEDS]); vw = np.mean([R[(t, s)]['source_val_iou_wide'] for s in SEEDS])
        row = {'t_frozen': tf, 't_wide': tw, 'median_frozen': float(np.median(tf)), 'median_wide': float(np.median(tw)), 'censored_frozen': int(cf), 'censored_wide': int(cw),
               'source_val_iou_frozen_mean': float(vf), 'source_val_iou_wide_mean': float(vw), 'n_wide_below_0.40': int(sum(1 for x in tw if x < 0.40))}
        for d in ('whu', 'mass'):
            row[f'{d}_iou_frozen_mean'] = float(np.mean([R[(t, s)]['target'][d]['iou_at_frozen'] for s in SEEDS]))
            row[f'{d}_iou_wide_mean'] = float(np.mean([R[(t, s)]['target'][d]['iou_at_wide'] for s in SEEDS]))
        out['per_arm'][SHORT[t]] = row
        print(f"  {SHORT[t]}: frozen median {row['median_frozen']:.2f} (censored {cf}/12)  wide median {row['median_wide']:.2f} (censored {cw}/12; below 0.40 {row['n_wide_below_0.40']}/12)  "
              f"val IoU {vf:.4f}->{vw:.4f}  WHU {row['whu_iou_frozen_mean']:.4f}->{row['whu_iou_wide_mean']:.4f}  Mass {row['mass_iou_frozen_mean']:.4f}->{row['mass_iou_wide_mean']:.4f}")
    print('\n' + '=' * 78); print('[contrasts] C-A / B-A (K4_r0) under both policies: effect, CI, direction, MDE, verdict class'); print('=' * 78)
    for d in ('whu', 'mass'):
        for pol in ('frozen', 'wide'):
            v = {SHORT[t]: np.array([R[(t, s)]['target'][d][f'iou_at_{pol}'] for s in SEEDS]) for t in TAGS}
            for name, x in (('C_minus_A', v['C'] - v['A']), ('B_minus_A', v['B'] - v['A']), ('C_minus_B', v['C'] - v['B'])):
                pr = paired(x); pr['verdict'] = verdict(pr); out['contrasts'][f'{d}_{pol}_{name}'] = pr
        for name in ('C_minus_A', 'B_minus_A', 'C_minus_B'):
            f, w = out['contrasts'][f'{d}_frozen_{name}'], out['contrasts'][f'{d}_wide_{name}']
            same = f['verdict'] == w['verdict']
            print(f"  {d} {name}: frozen {f['effect']:+.4f} [{f['ci95_descriptive'][0]:+.4f}, {f['ci95_descriptive'][1]:+.4f}] {f['direction_positive']}/12 MDE {f['MDE']:.4f} -> {f['verdict']}  |  "
                  f"wide {w['effect']:+.4f} [{w['ci95_descriptive'][0]:+.4f}, {w['ci95_descriptive'][1]:+.4f}] {w['direction_positive']}/12 MDE {w['MDE']:.4f} -> {w['verdict']}  |  verdict {'same' if same else '**differs (depends on the operating-point policy)**'}")
            out['contrasts'][f'{d}_{name}_verdict_same'] = bool(same)
    (ROOT / 'results' / 'y_srcwide_summary.json').write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    print('\n  -> results/y_srcwide_summary.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-view', default='data_view/inria'); ap.add_argument('--ckpt-dir', default='ckpt/y')
    ap.add_argument('--gpu', type=int, default=0); ap.add_argument('--batch-size', type=int, default=4); ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--seeds', default=' '.join(map(str, SEEDS))); ap.add_argument('--tags', default=' '.join(TAGS)); ap.add_argument('--summary-only', action='store_true')
    a = ap.parse_args()
    if not a.summary_only:
        device = E.M.setup_device(a.gpu)
        for s in map(int, a.seeds.split()):
            for t in a.tags.split(): sweep_one(t, s, a, device)
    summarize()


if __name__ == '__main__':
    main()
