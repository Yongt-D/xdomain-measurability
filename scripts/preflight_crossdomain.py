# -*- coding: utf-8 -*-
"""
Preflight: actually construct a Dataset / EpisodeDataset once and draw one sample; no GPU, no training.
Purpose: before any model code is written, verify that the cross-domain protocol of pre-registration section 2 can be
implemented on the existing chapter-4 code:
  is EpisodeDataset(query_set=Inria, support_set=Inria-support-pool, ...) feasible?
"Preflight" principle: build it once for real instead of guessing import by import.
Produces no performance number.
"""
import sys, pathlib, torch
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import gaplsegnet_v5_ch5 as M
print('import OK:', M.__file__)
print('ABLATIONS:', {k: (v.geometry, v.prototype_mode, v.geometry_weighting) for k, v in M.ABLATIONS.items()})

D = ROOT / 'data'
def show(tag, ds):
    img, lab = ds[0][0], ds[0][1]
    print(f'  {tag:28s} n={len(ds):5d}  img{tuple(img.shape)} {img.dtype} [{img.min():.3f},{img.max():.3f}]'
          f'  lab{tuple(lab.shape)} uniq={torch.unique(lab).tolist()[:4]}')
    return ds

print('\n[1] single-domain Dataset (source WHU / target Inria)')
whu_tr = show('WHU train', M.BuildingDataset(D/'whu_building'/'train'/'images', D/'whu_building'/'train'/'labels', train=True))
inr_te = show('Inria test  (query pool)', M.BuildingDataset(D/'inria'/'test'/'images',  D/'inria'/'test'/'labels',  train=False))
inr_tr = show('Inria train (support pool)', M.BuildingDataset(D/'inria'/'train'/'images', D/'inria'/'train'/'labels', train=False))

print('\n[2] cross-domain EpisodeDataset: query=Inria-test, support=Inria-train (K=4)')
ep = M.EpisodeDataset(
    query_set=inr_te, support_set=inr_tr,
    query_indices=list(range(8)), support_indices=list(range(16)),
    seed=0, require_distinct=False, support_shots=4,
)
q_img, q_mask, s_imgs, s_masks = ep[0]
print(f'  episode: query {tuple(q_img.shape)} / mask {tuple(q_mask.shape)}')
print(f'           support {tuple(s_imgs.shape)} / masks {tuple(s_masks.shape)}')
assert s_imgs.shape[0] == 4, 'support shots is not 4'

print('\n[3] split self-consistency (against the tile-level split of the ESWA revision R1-Q5)')
_bad = 0
for tag, ds, tiles in [('Inria test', inr_te, 37), ('Inria train', inr_tr, 125)]:
    n_exp = tiles * 81
    ok = len(ds) == n_exp
    _bad += (not ok)
    print(f'  {tag:12s} observed {len(ds):6d}   expected {tiles} tiles x 81 = {n_exp:6d}   {"OK" if ok else "**MISMATCH**"}')
assert _bad == 0, 'the Inria split does not match the ESWA revision statement R1-Q5; do not run'

print('\n[4] key conclusions')
print('  * query_set and support_set are independent parameters => the cross-domain protocol needs no change to EpisodeDataset;')
print('  * but the support pool here is the full Inria train split; pre-registration section 2.1 requires the support pool to be')
print('    **tile-disjoint** from the test set -- a separate split script following the R1-Q5 tile list is needed.')

print('\n[5] check of not-yet-resolved dependencies')
import os
vgg = pathlib.Path(os.path.expanduser('~/.cache/torch/hub/checkpoints/vgg16_bn-6c64b313.pth'))
print(f'  VGG16-BN weights: {"present" if vgg.exists() else "**missing** (downloaded on first run; verify the SHA256 afterwards)"}')
print('\nPREFLIGHT OK')
