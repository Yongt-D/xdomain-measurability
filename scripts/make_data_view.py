# -*- coding: utf-8 -*-
"""
Build a directory-name mapping view: the chapter-4 code expects <split>/image and <split>/label (singular),
whereas the dataset archive uses images / labels (plural).

To keep src/gaplsegnet_v5_ch5.py **bit-identical** to the chapter-4 archive (pre-registration section 4 item 6),
the code is not changed; a view directory is built from symbolic links / junctions instead (junction on Windows, symlink on Linux).
No GPU; no data is copied.
"""
import os, sys, subprocess, pathlib
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC  = pathlib.Path(os.environ.get('CH5_DATA_ROOT', ROOT / 'data')).resolve()
VIEW = ROOT / 'data_view'

def link(target: pathlib.Path, linkpath: pathlib.Path):
    if linkpath.exists() or linkpath.is_symlink():
        return 'exists'
    linkpath.parent.mkdir(parents=True, exist_ok=True)
    if os.name == 'nt':
        subprocess.run(['cmd', '/c', 'mklink', '/J', str(linkpath), str(target)],
                       check=True, capture_output=True)
        return 'junction'
    os.symlink(target, linkpath, target_is_directory=True)
    return 'symlink'

made = []
for ds in ('whu_building', 'inria', 'massachusetts'):
    for split in ('train', 'val', 'test'):
        for plural, singular in (('images', 'image'), ('labels', 'label')):
            tgt = SRC / ds / split / plural
            if not tgt.is_dir():
                continue
            made.append((f'{ds}/{split}/{singular}', link(tgt, VIEW / ds / split / singular)))

print(f'data root: {SRC}')
print(f'view root: {VIEW}')
for name, how in made:
    print(f'  {how:9s} data_view/{name}')

# self-check: the view reads the same number of files as the source
import itertools
bad = 0
for ds in ('whu_building', 'inria'):
    for split in ('train', 'val', 'test'):
        a, b = SRC / ds / split / 'images', VIEW / ds / split / 'image'
        if not a.is_dir():
            continue
        na, nb = len(list(a.iterdir())), len(list(b.iterdir()))
        ok = na == nb
        bad += (not ok)
        print(f'  check {ds}/{split}: source {na} / view {nb} {"OK" if ok else "**MISMATCH**"}')
print('\nDATA VIEW OK' if bad == 0 else '\n**DATA VIEW FAILED**')
sys.exit(0 if bad == 0 else 1)
