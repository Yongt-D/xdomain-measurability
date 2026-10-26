# -*- coding: utf-8 -*-
"""
Appendix U.2: the Massachusetts view. query = original train + original val (501), support pool = original test (40).
Copied to data_view_mass/{test,train}/{image,label} (test = query, train = support pool, following the directory convention of the evaluation scripts),
with a per-file sha256 manifest results/mass_view_manifest.json. No GPU, no performance number.
Usage: CH5_DATA_ROOT=<...>/CRBS/data python scripts/make_mass_view.py
"""
import os, sys, json, hashlib, shutil, pathlib
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = pathlib.Path(os.environ.get('CH5_DATA_ROOT', ROOT / 'data')).resolve() / 'massachusetts'
OUT = ROOT / 'data_view_mass'
def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()
def tile_of(name):
    return name.split('_patch')[0]
plan = {'test': ['train', 'val'], 'train': ['test']}   # view split -> list of source splits
man = {'source': str(SRC), 'query_from': plan['test'], 'support_from': plan['train'], 'files': {}}
for view_split, src_splits in plan.items():
    names = []
    for ss in src_splits:
        for fn in sorted(os.listdir(SRC / ss / 'images')):
            assert (SRC / ss / 'labels' / fn).is_file(), f'missing label {ss}/{fn}'
            names.append((ss, fn))
    assert len({fn for _, fn in names}) == len(names), 'duplicate file names'
    for kind in ('image', 'label'):
        (OUT / view_split / kind).mkdir(parents=True, exist_ok=True)
    rows = []
    for ss, fn in names:
        rec = {'name': fn, 'src_split': ss, 'tile': tile_of(fn)}
        for kind, plural in (('image', 'images'), ('label', 'labels')):
            src, dst = SRC / ss / plural / fn, OUT / view_split / kind / fn
            if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                shutil.copyfile(src, dst)
            rec[f'sha256_{kind}'] = sha(dst)
        rows.append(rec)
    man['files'][view_split] = rows
    print(f'  view {view_split:5s} <- {src_splits}: {len(rows)} patches, tiles={len({r["tile"] for r in rows})}')
q = man['files']['test']; sp = man['files']['train']
assert len(q) == 501 and len(sp) == 40, f'counts {len(q)}/{len(sp)} != 501/40'
inter = {r['tile'] for r in q} & {r['tile'] for r in sp}
assert not inter, f'tile-level leak: {sorted(inter)[:5]}'
man['counts'] = {'query': len(q), 'support_pool': len(sp), 'query_tiles': len({r['tile'] for r in q}), 'support_tiles': len({r['tile'] for r in sp})}
outp = ROOT / 'results' / 'mass_view_manifest.json'
outp.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding='utf-8')
print(f'  tile-level disjointness OK; manifest -> {outp}  sha256(manifest) {sha(outp)}')
print('MASS VIEW OK')
