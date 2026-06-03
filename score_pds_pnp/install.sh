#!/usr/bin/env bash
set -euo pipefail

# This script is meant to be run from the workspace root:
# /home/dc3000/shares/mdrive/PhD/projects_6oct2025/score_primal_dual_pnp_project

WORKSPACE="${1:-$(pwd)}"
PDS_REF="$WORKSPACE/refs/pds_pnp_ref"
SCORE_REF="$WORKSPACE/refs/score_pnp_ref"
TARGET="$WORKSPACE/score_pds_pnp"

if [[ ! -d "$PDS_REF" ]]; then
  echo "ERROR: Could not find PDS reference repo: $PDS_REF" >&2
  exit 1
fi
if [[ ! -d "$SCORE_REF" ]]; then
  echo "ERROR: Could not find score_pnp reference repo: $SCORE_REF" >&2
  exit 1
fi

mkdir -p "$TARGET/results" "$TARGET/logs"

# Patch the PDS reference repo for Linux paths and PyTorch >= 2.6 checkpoint loading.
python "$TARGET/tools/patch_pds_ref.py" --pds_repo "$PDS_REF"

chmod +x "$TARGET/scripts"/*.sh

echo "Installed Score-PDS-PnP bridge in: $TARGET"
echo "Next: python $TARGET/tools/check_assets.py --workspace $WORKSPACE"
