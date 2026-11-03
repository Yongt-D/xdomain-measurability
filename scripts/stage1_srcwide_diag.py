# -*- coding: utf-8 -*-
"""Leakage-free wide-grid source-threshold check for the WHU-trained (stage-1) checkpoints (descriptive; added after all results were known).

The counterpart of scripts/y_srcwide_diag.py (Appendix AA) for the primary source domain. For every (arm, seed):
  1. sweep 0.05-0.95 (step 0.01) on the WHU source validation split (the L1 split: create_dataloaders with the checkpoint's
     sample_size / sample_seed); the argmax (ties -> lowest threshold) is t_wide, the argmax inside 0.40-0.60 is checked against
     the L1 frozen threshold;
  2. the target IoU at t_frozen and t_wide is read from the committed widened sweeps (results/thrdiag_stage1/{tag}_seed{s}_gsd1.json
     for Inria, results/thrdiag_mass/{tag}_seed{s}.json for Massachusetts): no new target inference, no target labels touched;
  3. write results/stage1_srcwide/{tag}_seed{s}.json and, once all 36 exist, results/stage1_srcwide_summary.json.
Diagnostic only: nothing enters a results table and the primary criterion is not replaced.
"""
import argparse, hashlib, json, pathlib, sys, time
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import eval_crossdomain as E
import y_srcwide_diag as Y          # reuse mde / paired / verdict / argmax_lowest / grids
SEEDS, TAGS, SHORT, WIDE, NARROW, KEY = Y.SEEDS, Y.TAGS, Y.SHORT, Y.WIDE, Y.NARROW, Y.KEY
OUTDIR = ROOT / 'results' / 'stage1_srcwide'
TARGETS = {'inria': lambda tag, s: ROOT / 'results' / 'thrdiag_stage1' / f'{tag}_seed{s}_gsd1.json',
           'mass': lambda tag, s: ROOT / 'results' / 'thrdiag_mass' / f'{tag}_seed{s}.json'}


def L(p):
    p = pathlib.Path(p)
    if not p.exists(): print(f'  missing {p}'); sys.exit(1)
    return json.loads(p.read_text(encoding='utf-8'))


def sweep_one(tag, seed, a, device):
    out = OUTDIR / f'{tag}_seed{seed}.json'
    if out.exists(): print(f'skip {out.name}'); return
    ckpt = pathlib.Path(a.ckpt_dir) / f'{tag}_seed{seed}.pth'
    model, state, info = E.load_checkpoint(str(ckpt), device); targs = state['args']
    loaders = E.M.create_dataloaders(a.source_view, targs['sample_size'], a.batch_size, targs['sample_seed'], a.num_workers, targs['support_shots'])
    t0 = time.time(); acc = E.sweep_counts(model, loaders[1], device, WIDE)
    t_wide = Y.argmax_lowest(acc, WIDE); t_narrow = Y.argmax_lowest(acc, NARROW)
    l1 = L(ROOT / 'results' / 'stage1' / f'{tag}_seed{seed}_{KEY}.json'); t_frozen = l1['source_val_threshold']
    sha = hashlib.sha256(open(ckpt, 'rb').read()).hexdigest()
    rec = {'tag': tag, 'seed': seed, 'checkpoint': str(ckpt), 'checkpoint_sha256': sha, 'l1_checkpoint_sha256': l1.get('checkpoint_sha256'),
           'source_val_iou_by_threshold': {str(t): E.iou_of(acc[t]) for t in WIDE},
           't_frozen': t_frozen, 't_narrow_recomputed': t_narrow, 't_wide': t_wide,
           'censored_frozen': bool(t_frozen in (0.40, 0.60)), 'censored_wide': bool(t_wide in (0.05, 0.95)),
           'source_val_iou_frozen': E.iou_of(acc[t_frozen]), 'source_val_iou_wide': E.iou_of(acc[t_wide]), 'target': {}, 'seconds': time.time() - t0}
    for dom, path in TARGETS.items():
        p = path(tag, seed)
        if not p.exists():            # the Inria widened sweeps (Appendix P) were run for arms A and C only
            rec['target'][dom] = None; continue
        th = L(p)
        rec['target'][dom] = {'thrdiag_frozen_threshold': th['frozen_threshold'], 'iou_frozen_thrdiag': th['iou_frozen'],
                              'iou_at_frozen': th['iou_by_threshold'][str(t_frozen)], 'iou_at_wide': th['iou_by_threshold'][str(t_wide)]}
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding='utf-8')
    tg = '  '.join(f"{d} {v['iou_at_frozen']:.4f}->{v['iou_at_wide']:.4f}" if v else f'{d} n/a' for d, v in rec['target'].items())
    print(f"  {SHORT[tag]} seed{seed}: t_frozen {t_frozen} (recomputed {t_narrow}) t_wide {t_wide}  val {rec['source_val_iou_frozen']:.4f}->{rec['source_val_iou_wide']:.4f}  {tg}  {rec['seconds']:.0f}s")


def summarize():
    R = {(t, s): L(OUTDIR / f'{t}_seed{s}.json') for t in TAGS for s in SEEDS}
    have = lambda r, d: r['target'].get(d) is not None
    bad = [(t, s) for (t, s), r in R.items() if (r['l1_checkpoint_sha256'] and r['checkpoint_sha256'] != r['l1_checkpoint_sha256'])
           or any(have(r, d) and abs(r['target'][d]['iou_at_frozen'] - r['target'][d]['iou_frozen_thrdiag']) > 1e-9 for d in TARGETS)
           or any(have(r, d) and r['target'][d]['thrdiag_frozen_threshold'] != r['t_frozen'] for d in TARGETS)]
    flips = {}
    for (t, s), r in R.items():
        if r['t_narrow_recomputed'] != r['t_frozen']:
            v = r['source_val_iou_by_threshold']; gap = abs(v[str(r['t_frozen'])] - v[str(r['t_narrow_recomputed'])])
            flips[f'{SHORT[t]}{s}'] = {'t_frozen': r['t_frozen'], 't_recomputed': r['t_narrow_recomputed'], 'val_gap': gap}
            if gap >= 5e-4: bad.append((t, s))
    if bad: print('  mismatch:', bad); sys.exit(2)
    print(f"  gate pass; re-selection flips {len(flips)}/36: {flips if flips else 'none'}")
    out = {'seeds': SEEDS, 'note': 'WHU-source counterpart of Appendix AA: leakage-free wide-grid secondary policy; descriptive; does not replace the primary criterion',
           'gate_flips': flips, 'per_arm': {}, 'contrasts': {}}
    for t in TAGS:
        tf = [R[(t, s)]['t_frozen'] for s in SEEDS]; tw = [R[(t, s)]['t_wide'] for s in SEEDS]
        row = {'t_frozen': tf, 't_wide': tw, 'median_frozen': float(np.median(tf)), 'median_wide': float(np.median(tw)),
               'censored_frozen': int(sum(R[(t, s)]['censored_frozen'] for s in SEEDS)), 'censored_wide': int(sum(R[(t, s)]['censored_wide'] for s in SEEDS)),
               'n_wide_equals_frozen': int(sum(1 for a, b in zip(tf, tw) if a == b)), 'max_abs_shift': float(max(abs(a - b) for a, b in zip(tf, tw))),
               'source_val_iou_frozen_mean': float(np.mean([R[(t, s)]['source_val_iou_frozen'] for s in SEEDS])),
               'source_val_iou_wide_mean': float(np.mean([R[(t, s)]['source_val_iou_wide'] for s in SEEDS]))}
        for d in TARGETS:
            if all(have(R[(t, s)], d) for s in SEEDS):
                row[f'{d}_iou_frozen_mean'] = float(np.mean([R[(t, s)]['target'][d]['iou_at_frozen'] for s in SEEDS]))
                row[f'{d}_iou_wide_mean'] = float(np.mean([R[(t, s)]['target'][d]['iou_at_wide'] for s in SEEDS]))
        out['per_arm'][SHORT[t]] = row
        tg = '  '.join(f"{d} {row[f'{d}_iou_frozen_mean']:.4f}->{row[f'{d}_iou_wide_mean']:.4f}" for d in TARGETS if f'{d}_iou_wide_mean' in row)
        print(f"  {SHORT[t]}: frozen median {row['median_frozen']:.2f}  wide median {row['median_wide']:.2f} (equal {row['n_wide_equals_frozen']}/12, max shift {row['max_abs_shift']:.2f}, censored wide {row['censored_wide']}/12)  "
              f"val {row['source_val_iou_frozen_mean']:.4f}->{row['source_val_iou_wide_mean']:.4f}  {tg}")
    for d in TARGETS:
        arms = [t for t in TAGS if all(have(R[(t, s)], d) for s in SEEDS)]
        pairs = [(n, a, b) for n, a, b in (('C_minus_A', 'Cconfidence', 'Aplain'), ('B_minus_A', 'Bplain', 'Aplain'), ('C_minus_B', 'Cconfidence', 'Bplain')) if a in arms and b in arms]
        for pol in ('frozen', 'wide'):
            v = {t: np.array([R[(t, s)]['target'][d][f'iou_at_{pol}'] for s in SEEDS]) for t in arms}
            for name, a_, b_ in pairs:
                pr = Y.paired(v[a_] - v[b_]); pr['verdict'] = Y.verdict(pr); out['contrasts'][f'{d}_{pol}_{name}'] = pr
        for name, _, _ in pairs:
            f, w = out['contrasts'][f'{d}_frozen_{name}'], out['contrasts'][f'{d}_wide_{name}']
            same = f['verdict'] == w['verdict']; out['contrasts'][f'{d}_{name}_verdict_same'] = bool(same)
            print(f"  {d} {name}: frozen {f['effect']:+.4f} {f['direction_positive']}/12 MDE {f['MDE']:.4f} -> {f['verdict']}  |  wide {w['effect']:+.4f} {w['direction_positive']}/12 MDE {w['MDE']:.4f} -> {w['verdict']}  |  {'same' if same else '**differs**'}")
    (ROOT / 'results' / 'stage1_srcwide_summary.json').write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    print('  -> results/stage1_srcwide_summary.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-view', default='data_view/whu_building'); ap.add_argument('--ckpt-dir', default='ckpt/stage1')
    ap.add_argument('--gpu', type=int, default=0); ap.add_argument('--batch-size', type=int, default=4); ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--seeds', default=' '.join(map(str, SEEDS))); ap.add_argument('--tags', default=' '.join(TAGS)); ap.add_argument('--summary-only', action='store_true')
    a = ap.parse_args()
    if not a.summary_only:
        import torch
        device = torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu'); print('device', device)
        for tag in a.tags.split():
            for s in map(int, a.seeds.split()):
                sweep_one(tag, s, a, device)
    if all((OUTDIR / f'{t}_seed{s}.json').exists() for t in TAGS for s in SEEDS):
        summarize()
    print('STAGE1 SRCWIDE DONE')


if __name__ == '__main__':
    main()
