# -*- coding: utf-8 -*-
"""
Appendix U.2: the Massachusetts support manifests. Pool = data_view_mass/train (the original test split, 40 patches, 10 tiles).
Same rule as build_support_manifest.py: f_min=0.01, K in {1,4,8}, R=5, the K patches from distinct tiles whenever possible,
the seed depends only on (f_min,K,r). No GPU, no performance number.
"""
import json, sys, hashlib, pathlib
import numpy as np
from PIL import Image
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
POOL = ROOT / 'data_view_mass' / 'train'
MAN = ROOT / 'results' / 'support_manifests'; MAN.mkdir(parents=True, exist_ok=True)
names = sorted(p.name for p in (POOL / 'image').iterdir())
tile_of = {fn: fn.split('_patch')[0] for fn in names}
fg = {fn: float((np.asarray(Image.open(POOL / 'label' / fn).convert('L')) > 127).mean()) for fn in names}
arr = np.array(list(fg.values()))
print(f'pool {len(names)} patches / {len(set(tile_of.values()))} tiles; median building fraction {np.median(arr):.4f}, all-zero {int((arr==0).sum())}')
manifests = {}
f_min = 0.01
cand = sorted([fn for fn, v in fg.items() if v >= f_min])
print(f'f >= {f_min}: {len(cand)} candidates')
for K in (1, 4, 8):
    for r in range(5):
        rng = np.random.default_rng(abs(hash((round(f_min, 4), K, r))) % (2**32))
        order = rng.permutation(len(cand))
        picked, used = [], set()
        for idx in order:
            fn = cand[idx]
            if tile_of[fn] in used: continue
            picked.append(fn); used.add(tile_of[fn])
            if len(picked) == K: break
        if len(picked) < K:
            for idx in order:
                fn = cand[idx]
                if fn in picked: continue
                picked.append(fn)
                if len(picked) == K: break
        assert len(picked) == K
        manifests[f'fmin{f_min}_K{K}_r{r}'] = {'f_min': f_min, 'K': K, 'draw': r, 'n_candidates': len(cand),
                                               'distinct_tiles': len({tile_of[f] for f in picked}), 'files': picked}
payload = json.dumps(manifests, indent=1, sort_keys=True, ensure_ascii=False)
path = MAN / 'mass_support_manifests.json'
path.write_text(payload, encoding='utf-8')
digest = hashlib.sha256(payload.encode('utf-8')).hexdigest()
(MAN / 'mass_support_manifests.sha256').write_text(digest + '  mass_support_manifests.json\n')
print(f'{len(manifests)} manifests -> {path.relative_to(ROOT)}  SHA256 {digest}')
for key in ('fmin0.01_K1_r0', 'fmin0.01_K4_r0', 'fmin0.01_K8_r0'):
    m = manifests[key]; print(f'  {key}: distinct tiles={m["distinct_tiles"]}/{m["K"]}  {m["files"][:3]}')
print('MANIFEST OK')
