#!/usr/bin/env bash
# Build the Zenodo archive of the per-evaluation result files (the material the paper's reproducibility statement promises).
#   Contents: every git-tracked file under results/ (per-evaluation JSONs incl. per-sample records, analysis summaries,
#   support/view manifests, checkpoint digests), the pre-registration document (original language) with its commit
#   provenance table, README, licenses. English paths only. No checkpoints (provided on request owing to size).
# Usage (Git Bash):  bash scripts/build_zenodo_bundle.sh [zenodo/xdomain-measurability-results-v1]
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"; cd "$SRC"
VER=${VER:-v1}
OUT=${1:-zenodo/xdomain-measurability-results-$VER}
rm -rf "$OUT"; mkdir -p "$OUT/docs"
# 1) results: every tracked file, same layout as the working repository
git ls-files results | grep -vE '^results/(abibe|vlm_probe|probe)' | while read -r f; do mkdir -p "$OUT/$(dirname "$f")"; cp "$f" "$OUT/$f"; done
# 2) pre-registration + provenance (same generator as the public code snapshot)
cp "docs/预注册_L1L2_CH5Geo.md" "$OUT/docs/preregistration_zh.md"
{
  echo "# Provenance of the pre-registration document"; echo
  echo "\`docs/preregistration_zh.md\` is the pre-registration document of the study (original language: Chinese; the criteria are restated in English in Section 3 of the paper). The table lists every commit of the authors' private working repository that modified it."; echo
  echo "| Commit date | Private commit | Changed lines (+/−) |"; echo "|---|---|---|"
  git log --date=short --format='%ad %h' --numstat -- "docs/预注册_L1L2_CH5Geo.md" | awk 'NF==2 && $2 ~ /^[0-9a-f]{7,}$/ {d=$1; h=$2; next} NF==3 {printf "| %s | %s | +%s / −%s |\n", d, h, $1, $2}' | tac
} > "$OUT/docs/PROVENANCE.md"
cp docs/README_zenodo.md "$OUT/README.md"; cp LICENSE-CC-BY-4.0 "$OUT/LICENSE"
echo "working-repository commit: $(git rev-parse HEAD)  ($(git log -1 --date=short --format=%ad))" > "$OUT/SOURCE_COMMIT.txt"
# 3) checks: ASCII paths, no chapter-3 / private material
cd "$OUT"
PYTHONUTF8=1 "${PY:-/d/Anaconda/envs/beoc/python.exe}" -c "
import os,sys
bad=[os.path.join(r,f) for r,_,fs in os.walk('.') for f in fs if not os.path.join(r,f).isascii()]
low=[os.path.join(r,f) for r,_,fs in os.walk('.') for f in fs if any(k in os.path.join(r,f).lower() for k in ('abibe','vlm_probe','probe/','probe\\\\'))]
print('non-ASCII paths:',len(bad),bad[:3]); print('excluded material:',len(low),low[:3]); sys.exit(1 if bad or low else 0)"
# 4) digests of every file, then the zip next to the folder
find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.txt
echo "files: $(wc -l < SHA256SUMS.txt)  size: $(du -sh . | cut -f1)"
cd ..; rm -f "$(basename "$OUT").zip"; zip -qr "$(basename "$OUT").zip" "$(basename "$OUT")"
ls -la "$(basename "$OUT").zip"; sha256sum "$(basename "$OUT").zip"
echo "ZENODO BUNDLE OK"
