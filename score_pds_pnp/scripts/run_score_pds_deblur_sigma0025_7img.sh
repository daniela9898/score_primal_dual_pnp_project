#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${WORKSPACE:-/home/dc3000/shares/mdrive/PhD/projects_6oct2025/score_primal_dual_pnp_project}"
cd "$WORKSPACE"
python score_pds_pnp/main_score_pds_gaussian.py \
  --workspace "$WORKSPACE" \
  --task deblur \
  --noise_levels 0.0025 \
  --max_iter 1200 \
  --max_images 7 \
  --alpha 0.82 \
  --score_sigma_begin 120 \
  --score_sigma_end 10 \
  --verbose_every 50 \
  --result_tag score_pds_deblur_sigma0025
