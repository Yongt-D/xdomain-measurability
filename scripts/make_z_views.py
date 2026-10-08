# -*- coding: utf-8 -*-
"""
Appendix Z.2: the views of the two satellite target domains (no GPU, no performance number).
  Z1  WHU satellite I (global_cities_processed, 204 patches): merge train/val/test, rank the tiles by their numeric prefix (the number before the underscore),
      tiles with rank % 5 == 4 form the support pool (train/), the rest are queries (test/); RGB labels are binarised as "any channel > 127" to L 0/255.
  Z2  WHU satellite II (east_Asia_processed): query = test (903), support pool = train (2 194); labels as is (L 0/255; asserted to contain only 0/255).
Writes data_view_z_sat1/, data_view_z_sat2/ and results/z_{sat1,sat2}_view_manifest.json (per-file sha256, tile membership).
Usage: CH5_MAFBE=<...>/MAFBE/data python scripts/make_z_views.py
"""
import os, sys, json, hashlib, pathlib, shutil
import numpy as np
from PIL import Image
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = pathlib.Path(os.environ.get('CH5_MAFBE', str(ROOT / 'data' / 'MAFBE')))
def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def write_pair(img_src, lab_src, out_dir, name, binarize_rgb):
    (out_dir / 'image').mkdir(parents=True, exist_ok=True); (out_dir / 'label').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(img_src, out_dir / 'image' / name)
    lab = np.asarray(Image.open(lab_src))
    if binarize_rgb:
        lab = lab.max(axis=2) if lab.ndim == 3 else lab
    vals = set(np.unique(lab).tolist()); assert vals <= {0, 255}, (lab_src, sorted(vals)[:6])
    Image.fromarray(np.where(lab > 127, 255, 0).astype(np.uint8)).save(out_dir / 'label' / name)
    return {'name': name, 'src': str(img_src.relative_to(RAW)), 'sha256_image': sha(out_dir / 'image' / name), 'sha256_label': sha(out_dir / 'label' / name)}


def z1():
    src = RAW / 'global_cities_processed'; out = ROOT / 'data_view_z_sat1'
    items = []
    for split in ('train', 'val', 'test'):
        for fn in sorted(os.listdir(src / split / 'image')):
            items.append((fn, split))
    tiles = sorted({fn.split('_')[0] for fn, _ in items}, key=lambda t: int(t))
    rank = {t: i for i, t in enumerate(tiles)}
    sup_tiles = {t for t in tiles if rank[t] % 5 == 4}
    rows = {'test': [], 'train': []}
    for fn, split in items:
        t = fn.split('_')[0]; dest = 'train' if t in sup_tiles else 'test'
        r = write_pair(src / split / 'image' / fn, src / split / 'label' / fn, out / dest, fn, True)
        r.update({'tile': t, 'orig_split': split}); rows[dest].append(r)
    assert not ({r['tile'] for r in rows['test']} & {r['tile'] for r in rows['train']}), 'tile-level leak'
    man = {'raw': str(src), 'rule': 'tiles sorted by numeric prefix; rank % 5 == 4 -> support pool', 'n_tiles': len(tiles),
           'counts': {'query': len(rows['test']), 'query_tiles': len({r['tile'] for r in rows['test']}), 'support_pool': len(rows['train']), 'support_tiles': len(sup_tiles)},
           'files': rows}
    p = ROOT / 'results' / 'z_sat1_view_manifest.json'; p.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f"Z1: query {man['counts']['query']} / {man['counts']['query_tiles']} tiles; support {man['counts']['support_pool']} / {len(sup_tiles)} tiles; tiles disjoint OK -> {p.name} sha256 {sha(p)}")


def z2():
    src = RAW / 'east_Asia_processed'; out = ROOT / 'data_view_z_sat2'
    rows = {'test': [], 'train': []}
    for split, dest in (('test', 'test'), ('train', 'train')):
        for fn in sorted(os.listdir(src / split / 'image')):
            r = write_pair(src / split / 'image' / fn, src / split / 'label' / fn, out / dest, fn, False)
            r.update({'tile': fn.split('_')[0] if '_' in fn else 'test', 'orig_split': split}); rows[dest].append(r)
    man = {'raw': str(src), 'rule': 'query = original test; support pool = original train (tile-level disjointness relies on the dataset split)',
           'counts': {'query': len(rows['test']), 'support_pool': len(rows['train']), 'support_tiles': len({r['tile'] for r in rows['train']})}, 'files': rows}
    p = ROOT / 'results' / 'z_sat2_view_manifest.json'; p.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f"Z2: query {man['counts']['query']}; support {man['counts']['support_pool']} / {man['counts']['support_tiles']} groups -> {p.name} sha256 {sha(p)}")


if __name__ == '__main__':
    z1(); z2(); print('Z VIEWS OK')
