# -*- coding: utf-8 -*-
"""
Build the support-set manifests of pre-registration Appendix A. No GPU, no training, no performance number.

Step 1: scan the (city, tile) of Inria train/val/test and assert pairwise tile-level disjointness (A.2);
Step 2: compute the building-pixel fraction of every train label, cached in outputs/inria_train_fg.json;
Step 3: build the support manifests (K x R x f_min) per A.3/A.4 and write results/support_manifests/.

Manifests identify patches by **file name**, so they are machine-independent and can be copied to any machine.
"""
import json, re, sys, hashlib, pathlib
import numpy as np
from PIL import Image
sys.stdout.reconfigure(encoding='utf-8')

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / 'data' / 'inria'
OUT = ROOT / 'outputs'; OUT.mkdir(exist_ok=True)
MAN = ROOT / 'results' / 'support_manifests'; MAN.mkdir(parents=True, exist_ok=True)

NAME = re.compile(r'^([a-z\-]+?)(\d+)_(\d+)\.tif$')
def parse(fn):
    m = NAME.match(fn)
    if not m: raise ValueError(f'file name does not match <city><tile>_<patch>.tif: {fn}')
    return m.group(1), int(m.group(2)), int(m.group(3))

# ---- step 1: tile-level disjointness ----
print('[1] tile-level split assertion (pre-registration A.2)')
tiles, names = {}, {}
for split in ('train', 'val', 'test'):
    fns = sorted(p.name for p in (D / split / 'images').iterdir())
    names[split] = fns
    tiles[split] = {(c, t) for c, t, _ in map(parse, fns)}
    print(f'  {split:5s} patches={len(fns):6d}  tiles={len(tiles[split]):4d}  '
          f'cities={sorted({c for c, _ in tiles[split]})}')
bad = 0
for a, b in (('train','val'), ('train','test'), ('val','test')):
    inter = tiles[a] & tiles[b]
    print(f'  {a} & {b} = {len(inter)} {"OK" if not inter else "**LEAK: " + str(sorted(inter)[:5]) + "**"}')
    bad += len(inter)
assert bad == 0, 'tile-level overlap violates pre-registration A.2; do not run'
assert len(tiles['test']) == 37 and len(tiles['train']) == 125 and len(tiles['val']) == 18, \
    'tile counts do not match the ESWA revision statement R1-Q5'
print('  => tiles pairwise disjoint, and 125/18/37 agree with R1-Q5')

# ---- step 2: building fraction (cached) ----
cache = OUT / 'inria_train_fg.json'
if cache.exists():
    fg = json.loads(cache.read_text()); print(f'\n[2] building fraction: cache hit {cache.name} ({len(fg)} patches)')
else:
    print(f'\n[2] building fraction: scanning {len(names["train"])} labels (once; cached afterwards)')
    fg = {}
    lab_dir = D / 'train' / 'labels'
    for i, fn in enumerate(names['train']):
        a = np.asarray(Image.open(lab_dir / fn))
        fg[fn] = float((a > 0).mean())
        if (i + 1) % 2000 == 0: print(f'    {i+1}/{len(names["train"])}')
    cache.write_text(json.dumps(fg))
    print(f'    cached -> {cache}')
arr = np.array(list(fg.values()))
print(f'  building-fraction distribution: all-zero patches {int((arr==0).sum())} ({(arr==0).mean()*100:.1f}%),'
      f' median {np.median(arr):.4f}, mean {arr.mean():.4f}')
for f_min in (0.0, 0.01, 0.05):
    print(f'  f >= {f_min:<5}: candidates {int((arr > f_min).sum()) if f_min==0 else int((arr>=f_min).sum()):6d}')

# ---- step 3: build manifests ----
print('\n[3] building support manifests (A.4: K in {1,4,8}, R=5, f_min in {0,0.01,0.05})')
tile_of = {fn: parse(fn)[:2] for fn in names['train']}
manifests = {}
for f_min in (0.0, 0.01, 0.05):
    cand = sorted([fn for fn, v in fg.items() if (v > 0 if f_min == 0 else v >= f_min)])
    for K in (1, 4, 8):
        for r in range(5):
            # deterministic: the seed depends only on (f_min, K, r), not on the arm => all arms share the same support set
            rng = np.random.default_rng(abs(hash((round(f_min, 4), K, r))) % (2**32))
            # A.3: the K patches come from distinct tiles whenever possible
            order = rng.permutation(len(cand))
            picked, used_tiles = [], set()
            for idx in order:
                fn = cand[idx]
                if tile_of[fn] in used_tiles: continue
                picked.append(fn); used_tiles.add(tile_of[fn])
                if len(picked) == K: break
            if len(picked) < K:  # relax when there are not enough distinct tiles
                for idx in order:
                    fn = cand[idx]
                    if fn in picked: continue
                    picked.append(fn)
                    if len(picked) == K: break
            assert len(picked) == K
            manifests[f'fmin{f_min}_K{K}_r{r}'] = {
                'f_min': f_min, 'K': K, 'draw': r,
                'n_candidates': len(cand),
                'distinct_tiles': len({tile_of[f] for f in picked}),
                'files': picked,
            }
payload = json.dumps(manifests, indent=1, sort_keys=True, ensure_ascii=False)
path = MAN / 'inria_support_manifests.json'
path.write_text(payload, encoding='utf-8')
digest = hashlib.sha256(payload.encode('utf-8')).hexdigest()
(MAN / 'inria_support_manifests.sha256').write_text(digest + '  inria_support_manifests.json\n')
print(f'  {len(manifests)} manifests -> {path.relative_to(ROOT)}')
print(f'  SHA256 {digest}')
for key in ('fmin0.01_K1_r0', 'fmin0.01_K4_r0', 'fmin0.01_K8_r0'):
    m = manifests[key]
    print(f'  {key}: distinct tiles={m["distinct_tiles"]}/{m["K"]}  {m["files"][:3]}{"..." if m["K"]>3 else ""}')
print('\nMANIFEST OK')
