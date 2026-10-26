#!/usr/bin/env bash
# Appendix Z: evaluation of the 36 stage-1 checkpoints on WHU satellite I (z_sat1) and satellite II (z_sat2) (a5000). **Committed before any Z number existed.**
# Primary evaluation = mass_driver.sh (C 15 keys, A/B K4_r0; frozen threshold from results/stage1); widened-threshold diagnostic = mass_thrdiag_all.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
export PY=${PY:-$HOME/miniforge3/envs/beoc/bin/python}
export GPU=${GPU:-0} NW=${NW:-8} BS=${BS:-8}
export CKPT_LAYOUT=${CKPT_LAYOUT:-flat} CKPT_DIR=${CKPT_DIR:-ckpt/stage1}
export SEEDS=${SEEDS:-"100 101 102 103 104 105 106 107 108 109 110 111"}
export TAGS=${TAGS:-"Aplain Bplain Cconfidence"}
DOMAINS=${DOMAINS:-"sat1 sat2"}
for DOM in ${DOMAINS}; do
  case "$DOM" in
    sat1) SB="256,1024,4096"; BD="4" ;;
    sat2) SB="114,454,1822"; BD="4,2" ;;
  esac
  echo "##### $(date +%F_%T) Z domain ${DOM}: size-bins ${SB} band ${BD}"
  VIEW="data_view_z_${DOM}" MANIFEST="results/support_manifests/z_${DOM}_support_manifests.json" OUTDIR="results/z_${DOM}" SIZE_BINS="$SB" BAND="$BD" \
    bash scripts/mass_driver.sh
  VIEW="data_view_z_${DOM}" MANIFEST="results/support_manifests/z_${DOM}_support_manifests.json" OUTDIR="results/thrdiag_z_${DOM}" \
    bash scripts/mass_thrdiag_all.sh
done
echo "Z EVAL DONE $(hostname) domains=[${DOMAINS}]"
