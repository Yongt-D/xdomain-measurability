# -*- coding: utf-8 -*-
"""
Lineage check: confirm that src_forked_readonly/ is bit-identical to the chapter-4 archive, and that the working copy in src/
was also identical to it before being modified. No GPU.
Corresponds to pre-registration section 4 item 6 ("the identity control must really be identical"; CoRS final report section 6.4).
Exit code 0 = OK; non-zero = drift, do not run.
"""
import hashlib, sys, pathlib
sys.stdout.reconfigure(encoding='utf-8')
ROOT = pathlib.Path(__file__).resolve().parent.parent
RO = ROOT / 'src_forked_readonly'

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

# expected digests of source/ and results/ from the archive manifest
expected = {}
for line in (RO / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
    if not line.strip(): continue
    digest, name = line.split(' *', 1)
    expected[name.strip()] = digest.strip()

checks = [
    ('source/gaplsegnet_v5.py',     RO / 'gaplsegnet_v5.py'),
    ('source/README_V5_FINAL.md',   RO / 'README_V5_FINAL.md'),
    ('source/V5_CONFIRM_RESULT.md', RO / 'V5_CONFIRM_RESULT.md'),
    ('source/run_v5_confirm.sh',    RO / 'run_v5_confirm.sh'),
    ('source/run_v5_seed0.sh',      RO / 'run_v5_seed0.sh'),
    ('source/start_v5_confirm.sh',  RO / 'start_v5_confirm.sh'),
    ('source/start_v5_seed0.sh',    RO / 'start_v5_seed0.sh'),
]
for name in sorted(expected):
    if name.startswith('results/'):
        checks.append((name, ROOT / 'results' / 'ch4_reference' / pathlib.Path(name).name))

bad = 0
for name, path in checks:
    if not path.exists():
        print(f'MISSING  {name} -> {path}'); bad += 1; continue
    got = sha(path)
    ok = (got == expected[name])
    print(f'{"OK      " if ok else "MISMATCH"} {name}')
    if not ok:
        print(f'         expected {expected[name]}\n         got      {got}')
        bad += 1

# working copy: it should match the read-only copy only as long as no modification has started
work = ROOT / 'src' / 'gaplsegnet_v5_ch5.py'
if work.exists():
    same = sha(work) == sha(RO / 'gaplsegnet_v5.py')
    print(f'\nworking copy src/gaplsegnet_v5_ch5.py {"still bit-identical to the archive (not yet modified)" if same else "differs from the archive (modification has started -- expected, but the arm-A behaviour must be verified separately)"}')

print(f'\n{"FORK OK" if bad == 0 else f"FORK DRIFT: {bad} item(s) differ"}')
sys.exit(0 if bad == 0 else 1)
