# -*- coding: utf-8 -*-
"""
Appendix V.2: the full-tile Massachusetts query view.
  query   = the original train (137) + valid (4) tiles, each cut into a 3x3 grid of 512x512 patches (stride 494, 18 px overlap between neighbours),
            excluding patches with more than 50% pure-white pixels (255,255,255) -- a rule independent of the labels;
  support = the 40-patch support pool of Appendix U copied verbatim (data_view_mass/train), so that the U manifests and thresholds are reused as is.
Writes data_view_mass_full/{test,train}/{image,label} and results/mass_full_view_manifest.json (per-file sha256, exclusion list).
No GPU, no performance number. Usage: CH5_MASS_RAW=<...>/mass_buildings_dataset python scripts/make_mass_full_view.py
"""
import os, sys, json, hashlib, shutil, pathlib
import numpy as np
from PIL import Image
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = pathlib.Path(os.environ.get('CH5_MASS_RAW', str(ROOT / 'data' / 'mass_buildings_dataset')))
U_SUPPORT = ROOT / 'data_view_mass' / 'train'
OUT = ROOT / 'data_view_mass_full'
P, STRIDE, N = 512, 494, 3            # 3x3 grid: 0, 494, 988 (988+512 = 1500)
WHITE_MAX = 0.50
def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()

def main():
    for d in ('test/image', 'test/label', 'train/image', 'train/label'):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    rows, excluded = [], []
    for split, sat, mp in (('train', 'train_sat', 'train_map'), ('valid', 'valid_sat', 'valid_map')):
        for fn in sorted(os.listdir(RAW / sat)):
            stem = pathlib.Path(fn).stem
            lab_fn = next((f for f in os.listdir(RAW / mp) if pathlib.Path(f).stem == stem), None)
            assert lab_fn, f'missing label {stem}'
            img = np.asarray(Image.open(RAW / sat / fn).convert('RGB'))
            # the original map is RGB with building = (255,0,0); converted to grey this is 76, so binarise as "any channel > 127"
            # (revision: the first version used convert('L') > 127, which gave all-zero labels and was discarded)
            lab = (np.asarray(Image.open(RAW / mp / lab_fn).convert('RGB')).max(axis=2)).astype(np.uint8)
            assert img.shape[:2] == (1500, 1500) and lab.shape == (1500, 1500), (fn, img.shape, lab.shape)
            for i in range(N):
                for j in range(N):
                    y, x = i * STRIDE, j * STRIDE
                    im = img[y:y + P, x:x + P]; lb = lab[y:y + P, x:x + P]
                    name = f'{stem}_g{y}_{x}.png'
                    white = float((im == 255).all(axis=2).mean())
                    if white > WHITE_MAX:
                        excluded.append({'name': name, 'src_split': split, 'white_frac': white}); continue
                    Image.fromarray(im).save(OUT / 'test' / 'image' / name)
                    Image.fromarray(np.where(lb > 127, 255, 0).astype(np.uint8)).save(OUT / 'test' / 'label' / name)
                    rows.append({'name': name, 'src_split': split, 'tile': stem, 'y': y, 'x': x, 'white_frac': white,
                                 'sha256_image': sha(OUT / 'test' / 'image' / name), 'sha256_label': sha(OUT / 'test' / 'label' / name)})
    # support: copy U pool verbatim
    sup = []
    for kind in ('image', 'label'):
        for fn in sorted(os.listdir(U_SUPPORT / kind)):
            shutil.copyfile(U_SUPPORT / kind / fn, OUT / 'train' / kind / fn)
    for fn in sorted(os.listdir(OUT / 'train' / 'image')):
        sup.append({'name': fn, 'tile': fn.split('_patch')[0], 'sha256_image': sha(OUT / 'train' / 'image' / fn), 'sha256_label': sha(OUT / 'train' / 'label' / fn)})
    q_tiles = {r['tile'] for r in rows}; s_tiles = {r['tile'] for r in sup}
    inter = q_tiles & s_tiles
    assert not inter, f'tile-level leak: {sorted(inter)[:5]}'
    assert len(sup) == 40, len(sup)
    man = {'raw': str(RAW), 'grid': {'patch': P, 'stride': STRIDE, 'n': N, 'overlap_px': P - STRIDE}, 'white_max': WHITE_MAX,
           'counts': {'query': len(rows), 'query_tiles': len(q_tiles), 'excluded': len(excluded), 'support_pool': len(sup), 'support_tiles': len(s_tiles)},
           'files': {'test': rows, 'train': sup}, 'excluded': excluded}
    outp = ROOT / 'results' / 'mass_full_view_manifest.json'
    outp.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f"query {len(rows)} patches / {len(q_tiles)} tiles (excluded {len(excluded)}); support 40 / {len(s_tiles)} tiles; tile-level disjointness OK")
    print(f"manifest -> {outp}  sha256 {sha(outp)}")
    print('MASS FULL VIEW OK')

if __name__ == '__main__':
    main()
