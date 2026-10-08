# -*- coding: utf-8 -*-
"""
Appendix X: re-select the threshold over 0.05-0.95 using the labels of the K=4 support patches (few-shot calibration diagnostic). **Committed before any of its numbers existed.**
For every (arm, seed, target domain, support draw r in 0..4):
  - run the 4 support patches of that draw as queries (for arm C the prototype is still built from the same supports -- optimistic for C, disclosed);
  - pool tp/fp/fn over the 4 patches and take the grid threshold t_sup(r) with the highest IoU;
  - the full-query IoU is read from the committed widened-threshold curve at t_sup(r) (A/B do not depend on the supports; C uses the r0 curve, which is an approximation for r != 0, disclosed).
Writes results/x/{domain}_{tag}_seed{s}.json. Nature: diagnostic; not in the results table; the primary protocol is unchanged.
"""
import argparse, json, pathlib, sys, time, socket
import torch
from torch.utils.data import DataLoader
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
import gaplsegnet_v5_ch5 as M
from eval_crossdomain import FixedSupportSet, sweep_counts, iou_of
from model_loader import load_checkpoint
WIDE = [round(0.05 + 0.01 * i, 2) for i in range(91)]
DOMAINS = {
    'inria': dict(view='data_view_gsd/inria_gsd1', manifest='results/support_manifests/inria_support_manifests.json',
                  thr=lambda tag, s: ROOT / 'results' / 'thrdiag_stage1' / f'{tag}_seed{s}_gsd1.json'),
    'mass': dict(view='data_view_mass', manifest='results/support_manifests/mass_support_manifests.json',
                 thr=lambda tag, s: ROOT / 'results' / 'thrdiag_mass' / f'{tag}_seed{s}.json'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True); ap.add_argument('--tag', required=True); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--domain', choices=sorted(DOMAINS), required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--gpu', type=int, default=None)
    ap.add_argument('--draws', default='0,1,2,3,4')
    a = ap.parse_args()
    D = DOMAINS[a.domain]
    thr = json.loads(D['thr'](a.tag, a.seed).read_text(encoding='utf-8'))
    curve = {float(k): v for k, v in thr['iou_by_threshold'].items()}
    device = M.setup_device(a.gpu)
    model, state, info = load_checkpoint(a.checkpoint, device); model.eval()
    tv = ROOT / D['view']
    mans = json.loads((ROOT / D['manifest']).read_text(encoding='utf-8'))
    s_ds = M.BuildingDataset(tv / 'train' / 'image', tv / 'train' / 'label', train=False)
    s_names = sorted(p.name for p in (tv / 'train' / 'image').iterdir()); pos = {n: i for i, n in enumerate(s_names)}
    t0 = time.time(); per_draw = []
    for r in [int(x) for x in a.draws.split(',')]:
        key = f'fmin0.01_K4_r{r}'; man = mans[key]; sup_pos = [pos[f] for f in man['files']]
        assert len(sup_pos) == 4
        sub = torch.utils.data.Subset(s_ds, sup_pos)                  # the 4 supports as queries
        ds = FixedSupportSet(sub, s_ds, sup_pos)
        acc = sweep_counts(model, DataLoader(ds, batch_size=4, shuffle=False, num_workers=0), device, WIDE)
        sup_iou = {t: iou_of(acc[t]) for t in WIDE}
        t_sup = max(WIDE, key=lambda t: sup_iou[t])
        per_draw.append({'r': r, 'key': key, 'support_files': man['files'], 't_sup': t_sup, 'support_iou_at_t_sup': sup_iou[t_sup],
                         'support_iou_at_frozen': sup_iou[round(thr['frozen_threshold'], 2)] if round(thr['frozen_threshold'], 2) in sup_iou else None,
                         'query_iou_at_t_sup': curve[t_sup], 'curve_is_r0_approx': (info['kind'] == 'gapl' and state['args'].get('ablation') == 'C' and r != 0)})
    res = {'tag': a.tag, 'seed': a.seed, 'domain': a.domain, 'checkpoint': a.checkpoint, 'desc': info['desc'],
           'frozen_threshold': thr['frozen_threshold'], 'iou_frozen': thr['iou_frozen'],
           'oracle_threshold': thr['oracle_threshold'], 'iou_oracle': thr['iou_oracle_wide'], 'thrdiag_file': str(D['thr'](a.tag, a.seed).relative_to(ROOT)),
           'draws': per_draw, 'hostname': socket.gethostname(), 'seconds': time.time() - t0,
           'note': 'appendix X diagnostic; the support self-calibration of arm C is optimistic; not in the results table'}
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    m = sum(d['query_iou_at_t_sup'] for d in per_draw) / len(per_draw)
    print(f"  {a.domain} {a.tag} seed{a.seed}: frozen {thr['iou_frozen']:.4f} -> support-cal {m:.4f} (oracle {thr['iou_oracle_wide']:.4f}); t_sup {[d['t_sup'] for d in per_draw]}  ({res['seconds']:.0f}s) -> {out}")


if __name__ == '__main__':
    main()
