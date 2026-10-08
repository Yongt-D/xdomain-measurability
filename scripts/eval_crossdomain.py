# -*- coding: utf-8 -*-
"""
Cross-domain evaluation entry point (pre-registration section 2 / Appendix A). **Committed before any number existed.**

Protocol:
  - the model is trained on the source domain only (by the unmodified chapter-4 code; this script does not train);
  - the threshold is selected by a sweep on the **source-domain validation set** and then frozen; target labels are never used (section 2.3);
  - target domain: query = the full test split; support = the fixed K patches of the manifest (Appendix A.4);
    **all arms share the same support set for a given draw**;
  - per-sample quantities are written to disk (aggregate ratios cannot locate the shape of a distribution).

One **deliberate difference** from chapter 4 (stated in pre-registration A.4):
  the chapter-4 EpisodeDataset rotates the supports for every query; here the support is **one fixed set of K patches**,
  because what is operationalised is "you hold K labelled target patches", not "every query may draw a new support set".
"""
import argparse, json, hashlib, sys, pathlib, time
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import gaplsegnet_v5_ch5 as M
sys.path.insert(0, str(ROOT / 'scripts'))
from model_loader import load_checkpoint

EPS = M.EPS

class FixedSupportSet(Dataset):
    """Every query is paired with the same fixed set of K supports."""
    def __init__(self, query_ds, support_ds, support_positions):
        self.q = query_ds
        pairs = [support_ds[i] for i in support_positions]
        self.s_img = torch.stack([p[0] for p in pairs], 0)
        self.s_msk = torch.stack([p[1] for p in pairs], 0)
    def __len__(self): return len(self.q)
    def __getitem__(self, i):
        qi, qm = self.q[i]
        return qi, qm, self.s_img, self.s_msk


@torch.no_grad()
def sweep_counts(model, loader, device, thresholds, per_sample=None, names=None):
    """Return the global {threshold: (tp,fp,fn)} counts; when per_sample is given, also write per-sample rows at the primary threshold."""
    acc = {t: [0, 0, 0] for t in thresholds}
    model.eval()
    idx = 0
    for batch in loader:
        qi, qm, si, sm = [x.to(device, non_blocking=True) for x in batch]
        prob = model(qi, si, sm)["predictions"].detach()
        tgt = qm.detach().long()
        for t in thresholds:
            p = (prob > t).long()
            acc[t][0] += int(((p == 1) & (tgt == 1)).sum())
            acc[t][1] += int(((p == 1) & (tgt == 0)).sum())
            acc[t][2] += int(((p == 0) & (tgt == 1)).sum())
        if per_sample is not None:
            t = per_sample['threshold']
            p = (prob > t).long()
            for b in range(p.shape[0]):
                tp = int(((p[b] == 1) & (tgt[b] == 1)).sum())
                fp = int(((p[b] == 1) & (tgt[b] == 0)).sum())
                fn = int(((p[b] == 0) & (tgt[b] == 1)).sum())
                per_sample['rows'].append({
                    'name': names[idx] if names else idx,
                    'tp': tp, 'fp': fp, 'fn': fn,
                    'iou': tp / (tp + fp + fn + EPS),
                })
                idx += 1
    return acc


def iou_of(c): return c[0] / (c[0] + c[1] + c[2] + EPS)
def f1_of(c):  return 2 * c[0] / (2 * c[0] + c[1] + c[2] + EPS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--source-view', default=str(ROOT / 'data_view' / 'whu_building'))
    ap.add_argument('--target-view', default=str(ROOT / 'data_view' / 'inria'))
    ap.add_argument('--manifest', default=str(ROOT / 'results' / 'support_manifests' / 'inria_support_manifests.json'))
    ap.add_argument('--manifest-key', required=True, help='e.g. fmin0.01_K4_r0')
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--gpu', type=int, default=None)
    a = ap.parse_args()

    device = M.setup_device(a.gpu)
    model, state, info = load_checkpoint(a.checkpoint, device)      # Appendix W.2: unified loader for GAPL / standard architectures
    targs = state['args']
    print(f"loaded {a.checkpoint}\n  {info['desc']} best_epoch={state['epoch']} best_val_iou={state['best_val_iou']:.6f}")

    thresholds = [round(0.40 + 0.01 * i, 2) for i in range(21)]

    # ---- 1) threshold selection on the source validation set (target labels never used) ----
    src_loaders = M.create_dataloaders(
        a.source_view, targs['sample_size'], a.batch_size, targs['sample_seed'],
        a.num_workers, targs['support_shots'])
    val_loader = src_loaders[1]
    t0 = time.time()
    src_acc = sweep_counts(model, val_loader, device, thresholds)
    best_t = max(thresholds, key=lambda t: iou_of(src_acc[t]))
    print(f"  source validation threshold = {best_t}  (val IoU {iou_of(src_acc[best_t]):.6f}), "
          f"sweep {time.time()-t0:.1f}s")

    # ---- 2) target-domain evaluation with the manifest supports ----
    man = json.loads(pathlib.Path(a.manifest).read_text(encoding='utf-8'))[a.manifest_key]
    tv = pathlib.Path(a.target_view)
    q_ds = M.BuildingDataset(tv / 'test' / 'image', tv / 'test' / 'label', train=False)
    s_ds = M.BuildingDataset(tv / 'train' / 'image', tv / 'train' / 'label', train=False)
    s_names = sorted(p.name for p in (tv / 'train' / 'image').iterdir())
    pos = {n: i for i, n in enumerate(s_names)}
    sup_pos = [pos[f] for f in man['files']]
    assert len(sup_pos) == man['K']
    q_names = sorted(p.name for p in (tv / 'test' / 'image').iterdir())

    ds = FixedSupportSet(q_ds, s_ds, sup_pos)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                        num_workers=a.num_workers, pin_memory=True)
    per = {'threshold': best_t, 'rows': []}
    t0 = time.time()
    tgt_acc = sweep_counts(model, loader, device, thresholds, per_sample=per, names=q_names)

    res = {
        'checkpoint': str(a.checkpoint),
        'checkpoint_sha256': hashlib.sha256(open(a.checkpoint, 'rb').read()).hexdigest(),
        'train_args': targs, 'model_kind': info['kind'], 'arch': info['arch'],
        'manifest_key': a.manifest_key,
        'manifest': {k: man[k] for k in ('f_min', 'K', 'draw', 'n_candidates', 'distinct_tiles')},
        'support_files': man['files'],
        'source_val_threshold': best_t,
        'source_val_iou_at_threshold': iou_of(src_acc[best_t]),
        'target': {
            'n_query': len(q_ds),
            'iou': iou_of(tgt_acc[best_t]),
            'f1': f1_of(tgt_acc[best_t]),
            'tp_fp_fn': tgt_acc[best_t],
        },
        'target_threshold_sensitivity': {str(t): iou_of(tgt_acc[t]) for t in thresholds},
        'eval_seconds': time.time() - t0,
    }
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    ps = out.with_suffix('.per_sample.json')
    ps.write_text(json.dumps(per['rows'], ensure_ascii=False), encoding='utf-8')
    print(f"  target IoU = {res['target']['iou']:.6f}  (threshold {best_t}, n={len(q_ds)}), "
          f"{res['eval_seconds']:.1f}s")
    print(f"  -> {out}\n  -> {ps} ({len(per['rows'])} per-sample rows)")


if __name__ == '__main__':
    main()
