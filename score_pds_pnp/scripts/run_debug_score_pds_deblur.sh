#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${WORKSPACE:-/home/dc3000/shares/mdrive/PhD/projects_6oct2025/score_primal_dual_pnp_project}"
cd "$WORKSPACE"
python score_pds_pnp/main_score_pds_gaussian.py \
  --workspace "$WORKSPACE" \
  --task deblur \
  --noise_levels 0.0025 \
  --max_iter 20 \
  --max_images 1 \
  --alpha 0.82 \
  --score_sigma_begin 120 \
  --score_sigma_end 10 \
  --verbose_every 1 \
  --result_tag debug_score_pds
