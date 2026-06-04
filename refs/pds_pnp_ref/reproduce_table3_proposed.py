#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pnppds.operators import get_observation_operators
from utils.utils_noise import add_gaussian_noise
import pnppds.iteration as iteration


NOISE_LEVELS = [0.0025, 0.005, 0.01, 0.02, 0.04]

ALPHA = {
    0.0025: 0.82,
    0.005: 0.86,
    0.01: 0.92,
    0.02: 0.96,
    0.04: 1.00,
}

KERNEL_DIR = Path("/home/dc3000/shares/mdrive/PhD/projects_6oct2025/PnP_PD_paper_hypo/2025-hypomono-IEEE-SPL/Pnp-PD/blur_models")

KERNEL_FILES = [
    "blur_1.mat",
    "blur_2.mat",
    "blur_3.mat",
    "blur_4.mat",
    "blur_5.mat",
    "blur_6.mat",
    "blur_7.mat",
    "blur_8.mat",
    "gaussian_1_6.mat",
    "square_7.mat",
]

METHOD = "Gaussian-PnPPDS"
ARCHITECTURE = "DnCNN_nobn_nch_3_nlev_0.01"

MAX_ITER = 1200
GAMMA1 = 0.5
GAMMA2 = 0.99
R = 1
CH = 3

ALPHA_S = 1
MY_LAMBDA = 1
M1 = 15
M2 = 15
GAMMA_IN_ADMM_STEP1 = 0.1
LAMBDA_IN_STEP2 = 0.1
SP_NL = 0.0
POISSON_ETA = 1

with open(ROOT / "config" / "setup.json", "r") as f:
    config = json.load(f)

image_dir = Path(config["path_test"])
if not image_dir.exists():
    image_dir = ROOT / "datasets" / "imagenet"

image_paths = sorted(image_dir.glob("*.png"))[:7]

if len(image_paths) != 7:
    raise RuntimeError(f"Expected 7 ImageNet png images, found {len(image_paths)} in {image_dir}")

model_path = ROOT / "nn" / f"{ARCHITECTURE}.pth"
if not model_path.exists():
    raise FileNotFoundError(
        f"Missing denoiser checkpoint: {model_path}\n"
        "Run git lfs pull, or place DnCNN_nobn_nch_3_nlev_0.01.pth inside refs/pds_pnp_ref/nn/"
    )

kernel_paths = []
for name in KERNEL_FILES:
    p = KERNEL_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"Missing kernel: {p}")
    kernel_paths.append(p)

out_dir = ROOT / "results_table3_proposed"
out_dir.mkdir(exist_ok=True)

rows = []
matrix_rows = []

print("\nReproducing Table III: Proposed only")
print(f"Images: {[p.name for p in image_paths]}")
print(f"Kernels: {[p.name for p in kernel_paths]}")
print(f"Model:  {model_path}")
print(f"Output: {out_dir}\n")

for nl in NOISE_LEVELS:
    kernel_psnrs = []

    for kernel_index, kernel_path in enumerate(kernel_paths, start=1):
        phi, adj_phi = get_observation_operators(
            operator="blur",
            path_kernel=str(kernel_path),
            r=R,
        )
        Id, _ = get_observation_operators(
            operator="Id",
            path_kernel=str(kernel_path),
            r=R,
        )

        psnrs = []
        ssims = []

        print(f"\n=== sigma={nl} | kernel {kernel_index}/10: {kernel_path.name} | alpha={ALPHA[nl]} ===")

        for img_index, img_path in enumerate(image_paths, start=1):
            img_true = cv2.imread(str(img_path))
            if img_true is None:
                raise RuntimeError(f"Could not read image: {img_path}")

            img_true = np.asarray(img_true, dtype=np.float32) / 255.0
            img_true = np.moveaxis(img_true, -1, 0)

            img_obsrv = phi(img_true)
            img_obsrv = add_gaussian_noise(img_obsrv, nl, Id)
            x0 = np.copy(img_obsrv)

            img_sol, s_sol, c_evolution, psnr_evolution, ssim_evolution, avg_time, other_data = iteration.test_iter(
                x_0=x0,
                x_obsrv=img_obsrv,
                x_true=img_true,
                phi=phi,
                adj_phi=adj_phi,
                gamma1=GAMMA1,
                gamma2=GAMMA2,
                alpha_s=ALPHA_S,
                alpha_n=ALPHA[nl],
                myLambda=MY_LAMBDA,
                m1=M1,
                m2=M2,
                gammaInADMMStep1=GAMMA_IN_ADMM_STEP1,
                lambydaInStep2=LAMBDA_IN_STEP2,
                gaussian_nl=nl,
                sp_nl=SP_NL,
                poisson_eta=POISSON_ETA,
                path_prox=str(model_path),
                max_iter=MAX_ITER,
                method=METHOD,
                ch=CH,
                r=R,
            )

            final_psnr = float(psnr_evolution[-1])
            final_ssim = float(ssim_evolution[-1])

            psnrs.append(final_psnr)
            ssims.append(final_ssim)

            print(
                f"image {img_index}/7 {img_path.name}: "
                f"PSNR={final_psnr:.4f}, SSIM={final_ssim:.4f}"
            )

            rows.append({
                "noise": nl,
                "kernel": kernel_index,
                "kernel_file": kernel_path.name,
                "image": img_path.name,
                "psnr": final_psnr,
                "ssim": final_ssim,
                "alpha": ALPHA[nl],
                "gamma1": GAMMA1,
                "gamma2": GAMMA2,
                "max_iter": MAX_ITER,
                "method": METHOD,
                "architecture": ARCHITECTURE,
            })

        avg_psnr = float(np.mean(psnrs))
        avg_ssim = float(np.mean(ssims))
        kernel_psnrs.append(avg_psnr)

        print(
            f"AVERAGE sigma={nl}, kernel={kernel_index}, file={kernel_path.name}: "
            f"PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}"
        )

    avg_all_kernels = float(np.mean(kernel_psnrs))

    matrix_row = {
        "noise": nl,
        **{f"kernel_{i+1}": kernel_psnrs[i] for i in range(10)},
        "average": avg_all_kernels,
    }
    matrix_rows.append(matrix_row)

    print("\nTABLE III ROW")
    print(
        f"sigma={nl}: "
        + " ".join(f"{v:.2f}" for v in kernel_psnrs)
        + f" | Average {avg_all_kernels:.2f}"
    )

detail_csv = out_dir / "table3_proposed_detail.csv"
matrix_csv = out_dir / "table3_proposed_matrix.csv"

with open(detail_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

with open(matrix_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(matrix_rows[0].keys()))
    writer.writeheader()
    writer.writerows(matrix_rows)

print("\nSaved:")
print(detail_csv)
print(matrix_csv)

print("\nFinal Table III Proposed-only matrix:")
for row in matrix_rows:
    vals = [row[f"kernel_{i}"] for i in range(1, 11)]
    print(
        f"sigma={row['noise']}: "
        + " & ".join(f"{v:.2f}" for v in vals)
        + f" & {row['average']:.2f}"
    )
