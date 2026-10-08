# -*- coding: utf-8 -*-
"""
GSD-degradation views for L2 (pre-registration Appendix O.1). **Committed before any L2 number existed.**

Operator (O.1, fixed before running):
    I_f = TF.resize(I, [512 // f, 512 // f], antialias=True)     # images only, saved as lossless .tif
followed by the **unmodified** `BuildingDataset.__getitem__` (`src/gaplsegnet_v5_ch5.py:95`), which resizes back to 512
with `TF.resize(..., [512, 512], antialias=True)`.
=> the model input is always 512x512; the information content drops to 512/f.

- **labels are not down-sampled**: the evaluation geometry is identical at every GSD point (O.1);
- **support images share the query's GSD** (O.1);
- **file names and extensions are unchanged (.tif)**: `eval_crossdomain.py` locates supports in the manifest by file name,
  so a changed extension would break the lookup;
- `f = 1` goes through the same read/write path (no symlink shortcut) so that the O.3 cross-check can detect I/O problems.

Only the **target domain** (Inria) is processed: the full `test` split + the `train` patches referenced by the support manifests.
"""
import argparse, json, sys, pathlib, time
from PIL import Image
import torchvision.transforms.functional as TF
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
FACTORS = (1, 2, 4, 8)          # GSD = 0.3 * f metres
BASE = 512


def degrade_and_save(src: pathlib.Path, dst: pathlib.Path, f: int) -> None:
    img = Image.open(src).convert('RGB')
    if img.size != (BASE, BASE):
        img = TF.resize(img, [BASE, BASE], antialias=True)
    small = TF.resize(img, [BASE // f, BASE // f], antialias=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    small.save(dst, format='TIFF', compression=None)   # lossless


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-view', default=str(ROOT / 'data_view' / 'inria'),
                    help='the original 0.3 m Inria data view (with test/{image,label} and train/{image,label})')
    ap.add_argument('--out-root', default=str(ROOT / 'data_view_gsd'))
    ap.add_argument('--manifest', default=str(ROOT / 'results' / 'support_manifests' / 'inria_support_manifests.json'))
    ap.add_argument('--manifest-prefix', default='fmin0.01_K',
                    help='only the train images referenced by these manifests are down-sampled')
    a = ap.parse_args()

    sv = pathlib.Path(a.source_view)
    out_root = pathlib.Path(a.out_root)
    mans = json.loads(pathlib.Path(a.manifest).read_text(encoding='utf-8'))
    need_train = sorted({fn for k, m in mans.items()
                         if k.startswith(a.manifest_prefix) for fn in m['files']})
    test_imgs = sorted(p for p in (sv / 'test' / 'image').iterdir() if p.is_file())
    print(f'source view {sv}')
    print(f'  test images {len(test_imgs)}; train images referenced by the support manifests {len(need_train)}')

    for f in FACTORS:
        t0 = time.time()
        root = out_root / f'inria_gsd{f}'
        for split, files in (('test', test_imgs),
                             ('train', [sv / 'train' / 'image' / n for n in need_train])):
            for p in files:
                degrade_and_save(p, root / split / 'image' / p.name, f)
            # labels: symlink to the original directory (not down-sampled, O.1)
            lab = root / split / 'label'
            if not lab.exists():
                lab.parent.mkdir(parents=True, exist_ok=True)
                try:
                    lab.symlink_to((sv / split / 'label').resolve(), target_is_directory=True)
                except OSError as e:
                    print(f'  directory symlink failed ({e}); falling back to per-file symlinks')
                    lab.mkdir(parents=True, exist_ok=True)
                    for p in files:
                        for ext in ('.tif', '.tiff', '.png', '.jpg'):
                            q = (sv / split / 'label' / (p.stem + ext))
                            if q.exists():
                                (lab / q.name).symlink_to(q.resolve()); break
        n_t = len(list((root / 'test' / 'image').iterdir()))
        n_s = len(list((root / 'train' / 'image').iterdir()))
        sz = Image.open(next((root / 'test' / 'image').iterdir())).size
        print(f'  f={f}  GSD={0.3*f:.1f}m  -> {root}   test {n_t} / train {n_s}   '
              f'stored pixels {sz}   {time.time()-t0:.0f}s')

    print('\nGSD VIEWS OK')


if __name__ == '__main__':
    main()
