#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${WORKSPACE:-/home/dc3000/shares/mdrive/PhD/projects_6oct2025/score_primal_dual_pnp_project}"
cd "$WORKSPACE"
python score_pds_pnp/main_score_pds_gaussian.py \
  --workspace "$WORKSPACE" \
  --task both \
  --noise_levels 0.0025,0.005,0.01,0.02,0.04 \
  --max_iter 300 \
  --max_images 7 \
  --alpha 0.82 \
  --score_sigma_begin 120 \
  --score_sigma_end 10 \
  --verbose_every 100 \
  --result_tag score_pds_gaussian_quick300
