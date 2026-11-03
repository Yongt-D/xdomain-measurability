# -*- coding: utf-8 -*-
"""
Generic support-manifest builder (Appendices Y/Z): pool = <view>/train; same rule as build_support_manifest_mass.py
(f_min=0.01, K in {1,4,8}, R=5, the K patches from distinct tiles whenever possible, the random seed depends only on (f_min,K,r)).
The tile id is the first capture group of --tile-regex; without a tile structure use --tile-regex '^(.*)$' (every patch is its own tile,
which degenerates to random draws). No GPU, no performance number. Usage:
  python scripts/build_support_manifest_generic.py --view data_view/whu_building --name whu --tile-regex '^(.*)$'
  python scripts/build_support_manifest_generic.py --view data_view_z_sat1 --name z_sat1 --tile-regex '^(\\d+)_'
"""
import argparse, json, sys, hashlib, pathlib, re
import numpy as np
from PIL import Image
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
MAN = ROOT / 'results' / 'support_manifests'; MAN.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--view', required=True); ap.add_argument('--name', required=True); ap.add_argument('--tile-regex', required=True)
    ap.add_argument('--f-min', type=float, default=0.01)
    a = ap.parse_args()
    POOL = ROOT / a.view / 'train'; rx = re.compile(a.tile_regex)
    names = sorted(p.name for p in (POOL / 'image').iterdir())
    def tile_of(fn):
        m = rx.match(fn); assert m, f'{fn} does not match {a.tile_regex}'; return m.group(1)
    tiles = {fn: tile_of(fn) for fn in names}
    fg = {fn: float((np.asarray(Image.open(POOL / 'label' / fn).convert('L')) > 127).mean()) for fn in names}
    arr = np.array(list(fg.values()))
    print(f'pool {len(names)} patches / {len(set(tiles.values()))} tiles; median building fraction {np.median(arr):.4f}, all-zero {int((arr == 0).sum())}')
    f_min = a.f_min
    cand = sorted([fn for fn, v in fg.items() if v >= f_min])
    print(f'f >= {f_min}: {len(cand)} candidates')
    manifests = {}
    for K in (1, 4, 8):
        for r in range(5):
            rng = np.random.default_rng(abs(hash((round(f_min, 4), K, r))) % (2 ** 32))
            order = rng.permutation(len(cand))
            picked, used = [], set()
            for idx in order:
                fn = cand[idx]
                if tiles[fn] in used: continue
                picked.append(fn); used.add(tiles[fn])
                if len(picked) == K: break
            if len(picked) < K:
                for idx in order:
                    fn = cand[idx]
                    if fn in picked: continue
                    picked.append(fn)
                    if len(picked) == K: break
            assert len(picked) == K
            manifests[f'fmin{f_min}_K{K}_r{r}'] = {'f_min': f_min, 'K': K, 'draw': r, 'n_candidates': len(cand),
                                                   'distinct_tiles': len({tiles[f] for f in picked}), 'files': picked}
    payload = json.dumps(manifests, indent=1, sort_keys=True, ensure_ascii=False)
    path = MAN / f'{a.name}_support_manifests.json'
    path.write_text(payload, encoding='utf-8')
    digest = hashlib.sha256(payload.encode('utf-8')).hexdigest()
    (MAN / f'{a.name}_support_manifests.sha256').write_text(digest + f'  {a.name}_support_manifests.json\n')
    print(f'{len(manifests)} manifests -> {path.relative_to(ROOT)}  SHA256 {digest}')
    for key in ('fmin0.01_K1_r0', 'fmin0.01_K4_r0', 'fmin0.01_K8_r0'):
        m = manifests[key]; print(f'  {key}: distinct tiles={m["distinct_tiles"]}/{m["K"]}  {m["files"][:3]}')
    print('MANIFEST OK')


if __name__ == '__main__':
    main()
