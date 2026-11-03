# -*- coding: utf-8 -*-
"""
RQ5 (pre-registration Appendix T): pixel-level stratified evaluation + DGM residual hook. **Committed before it produced any number.**

Same model loading, support construction (FixedSupportSet) and forward pass as scripts/eval_crossdomain.py; differences:
  - the threshold is **passed in** (--threshold, the L1 frozen threshold) and not re-selected (avoids near-tie flips across machines);
  - per image, (n, p1) are recorded for the 10 strata of T.2: positive pixels S1-S4 x {edge <=4 px, interior} = 8 strata; negative pixels {near <=4 px, far} = 2 strata;
    the strata are determined by the label alone (8-connected instance area + Euclidean distance transform) and computed inside the dataloader workers;
  - for arms B/C a forward hook on geometry2/3/4 records the per-image relative DGM residual magnitude r_i (mean of the three stages; same definition as causal_ablations (4)).
Image-level tp/fp/fn = sum over strata, which must agree with the per-image counts of the same key in results/l2 (T.3.2).
"""
import argparse, json, hashlib, sys, pathlib, time, socket, platform
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import gaplsegnet_v5_ch5 as M
sys.path.insert(0, str(ROOT / 'scripts'))
from model_loader import load_checkpoint
EPS = M.EPS
SIZE_BINS = (256, 1024, 4096)     # S1 <256, S2 <1024, S3 <4096, S4 >=4096
BAND = 4.0                         # edge-band / near-band width (px)
N_STRATA = 10
STRATA_NAMES = ['S1_edge', 'S1_int', 'S2_edge', 'S2_int', 'S3_edge', 'S3_int', 'S4_edge', 'S4_int', 'neg_near', 'neg_far']


def strata_map(mask: np.ndarray, size_bins=SIZE_BINS, band=BAND) -> np.ndarray:
    """mask: HxW bool. Returns HxW uint8 with values 0..9 indexing STRATA_NAMES. (Revision 21: size_bins/band are parameters; defaults as in RQ5.)"""
    out = np.empty(mask.shape, dtype=np.uint8)
    if mask.any():
        lab, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        areas = np.bincount(lab.ravel())
        area_px = areas[lab]                                   # area of the instance each positive pixel belongs to
        size_bin = np.digitize(area_px, size_bins)             # 0..3
        d_in = ndimage.distance_transform_edt(mask)            # distance of a positive pixel to the nearest negative pixel
        edge = d_in <= band
        pos_code = size_bin * 2 + np.where(edge, 0, 1)         # 0..7
        d_out = ndimage.distance_transform_edt(~mask)
        neg_code = np.where(d_out <= band, 8, 9)
        out[:] = np.where(mask, pos_code, neg_code).astype(np.uint8)
    else:
        out[:] = 9
    return out


class StrataSupportSet(Dataset):
    """Every query is paired with the same fixed supports; the stratum map of the query is computed inside the worker."""
    def __init__(self, query_ds, support_ds, support_positions, size_bins=SIZE_BINS, bands=(BAND,)):
        self.q = query_ds
        self.size_bins, self.bands = tuple(size_bins), tuple(bands)
        pairs = [support_ds[i] for i in support_positions]
        self.s_img = torch.stack([p[0] for p in pairs], 0)
        self.s_msk = torch.stack([p[1] for p in pairs], 0)

    def __len__(self):
        return len(self.q)

    def __getitem__(self, i):
        qi, qm = self.q[i]
        m = qm[0].numpy() > 0.5
        sm = np.stack([strata_map(m, self.size_bins, b) for b in self.bands], 0)   # nb,H,W
        return qi, qm, torch.from_numpy(sm), self.s_img, self.s_msk, i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--target-view', required=True)
    ap.add_argument('--manifest', default=str(ROOT / 'results' / 'support_manifests' / 'inria_support_manifests.json'))
    ap.add_argument('--manifest-key', default='fmin0.01_K4_r0')
    ap.add_argument('--threshold', type=float, required=True, help='L1 frozen threshold (not re-selected)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--gpu', type=int, default=None)
    ap.add_argument('--size-bins', default='256,1024,4096', help='instance-area bins (px); default as in RQ5')
    ap.add_argument('--band', default='4', help='edge-band / near-band width (px); comma list allowed; the first goes to n/p1, the others to alt')
    a = ap.parse_args()
    size_bins = tuple(int(x) for x in a.size_bins.split(','))
    bands = tuple(float(x) for x in a.band.split(','))

    device = M.setup_device(a.gpu)
    model, state, info = load_checkpoint(a.checkpoint, device)      # Appendix W.2: unified loader for GAPL / standard architectures
    targs = state['args']
    model.eval()
    print(f"loaded {a.checkpoint}  {info['desc']}  threshold={a.threshold}")

    dgm = {}
    def mk_hook(tag):
        def hook(mod, inp, out):
            f = inp[0]
            rel = (out - f).flatten(1).norm(dim=1) / f.flatten(1).norm(dim=1).clamp_min(EPS)
            dgm.setdefault(tag, []).append(rel.detach().cpu())
        return hook
    for tag in ('geometry2', 'geometry3', 'geometry4'):
        if hasattr(model, tag):                                       # standard architectures have no DGM: dgm_rel is recorded as 0.0 (else branch below)
            getattr(model, tag).register_forward_hook(mk_hook(tag))

    tv = pathlib.Path(a.target_view)
    man = json.loads(pathlib.Path(a.manifest).read_text(encoding='utf-8'))[a.manifest_key]
    q_ds = M.BuildingDataset(tv / 'test' / 'image', tv / 'test' / 'label', train=False)
    s_ds = M.BuildingDataset(tv / 'train' / 'image', tv / 'train' / 'label', train=False)
    s_names = sorted(p.name for p in (tv / 'train' / 'image').iterdir())
    pos = {n: i for i, n in enumerate(s_names)}
    sup_pos = [pos[f] for f in man['files']]
    assert len(sup_pos) == man['K']
    q_names = sorted(p.name for p in (tv / 'test' / 'image').iterdir())
    ds = StrataSupportSet(q_ds, s_ds, sup_pos, size_bins, bands)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers,
                        pin_memory=True, persistent_workers=a.num_workers > 0)

    T = a.threshold
    rows = []
    t0 = time.time()
    with torch.no_grad():
        for qi, qm, sm, si, smk, idx in loader:
            qi, si, smk = qi.to(device, non_blocking=True), si.to(device, non_blocking=True), smk.to(device, non_blocking=True)
            prob = model(qi, si, smk)['predictions'].detach()
            pred = (prob > T).to(torch.uint8).cpu()[:, 0]          # B,H,W
            sm = sm.to(torch.int64)                                  # B,nb,H,W
            for b in range(pred.shape[0]):
                p = pred[b].reshape(-1).to(torch.int64)
                row = {'name': q_names[int(idx[b])]}
                for bi, band in enumerate(bands):
                    s = sm[b, bi].reshape(-1)
                    n = torch.bincount(s, minlength=N_STRATA).tolist()
                    p1 = torch.bincount(s, weights=p.to(torch.float64), minlength=N_STRATA).round().to(torch.int64).tolist()
                    if bi == 0:
                        row['n'], row['p1'] = n, p1
                    else:
                        row.setdefault('alt', {})[str(band)] = {'n': n, 'p1': p1}
                rows.append(row)
    # DGM residual (same order as the batches)
    if dgm:
        rel = torch.stack([torch.cat(dgm[t]) for t in ('geometry2', 'geometry3', 'geometry4')], 1).mean(1).numpy()
        for r, v in zip(rows, rel):
            r['dgm_rel'] = float(v)
    else:
        for r in rows:
            r['dgm_rel'] = 0.0
    tp = sum(sum(r['p1'][:8]) for r in rows); fn = sum(sum(r['n'][:8]) - sum(r['p1'][:8]) for r in rows)
    fp = sum(sum(r['p1'][8:]) for r in rows)
    res = {
        'checkpoint': str(a.checkpoint),
        'checkpoint_sha256': hashlib.sha256(open(a.checkpoint, 'rb').read()).hexdigest(),
        'train_args': targs, 'model_kind': info['kind'], 'arch': info['arch'], 'manifest_key': a.manifest_key, 'support_files': man['files'],
        'target_view': str(tv), 'threshold': T, 'n_query': len(q_ds),
        'strata_names': STRATA_NAMES, 'size_bins': list(size_bins), 'band_px': bands[0], 'bands_px': list(bands),
        'overall': {'tp': tp, 'fp': fp, 'fn': fn, 'iou': tp / (tp + fp + fn + EPS)},
        'strata_n_total': [sum(r['n'][k] for r in rows) for k in range(N_STRATA)],
        'strata_n_total_alt': {str(b): [sum(r['alt'][str(b)]['n'][k] for r in rows) for k in range(N_STRATA)] for b in bands[1:]},
        'hostname': socket.gethostname(), 'torch': torch.__version__, 'platform': platform.platform(),
        'batch_size': a.batch_size, 'num_workers': a.num_workers, 'eval_seconds': time.time() - t0,
        'rows': rows,
    }
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False), encoding='utf-8')
    print(f"  image-level IoU = {res['overall']['iou']:.6f}  (tp={tp} fp={fp} fn={fn}, n={len(rows)})  {res['eval_seconds']:.1f}s  -> {out}")


if __name__ == '__main__':
    main()
