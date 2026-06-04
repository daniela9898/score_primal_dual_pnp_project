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
from utils.utils_noise import apply_poisson_noise
from utils.utils_eval import eval_psnr, eval_ssim
import pnppds.iteration as iteration


# Table VII protocol
ETA_LIST = [1, 2, 10, 50, 100, 200]
TASKS = ["blur", "random_sampling"]

# Table VI lambda values for Proposed method
LAMBDA_DEBLUR = {
    1: 2.00e-3,
    2: 2.00e-3,
    10: 1.50e-3,
    50: 1.25e-3,
    100: 1.25e-3,
    200: 1.00e-3,
}

LAMBDA_INPAINT = {
    1: 1.25e-3,
    2: 1.25e-3,
    10: 1.00e-3,
    50: 7.50e-4,
    100: 5.00e-4,
    200: 5.00e-4,
}

METHOD = "Poisson-PnPPDS"
ARCHITECTURE = "DnCNN_nobn_nch_1_nlev_0.01"

GAMMA1 = 0.5
GAMMA2 = 0.99
SAMPLING_RATE = 0.8
CH = 1

# Unused/default parameters required by iteration.test_iter
ALPHA_S = 1
ALPHA_N = 1
M1 = 15
M2 = 15
GAMMA_IN_ADMM_STEP1 = 0.1
LAMBDA_IN_STEP2 = 0.1
GAUSSIAN_NL = 0.0
SP_NL = 0.0

with open(ROOT / "config" / "setup.json", "r") as f:
    config = json.load(f)

# Prefer Set3 if present. If your Set3 path has a different name, edit this line.
candidate_dirs = [
    ROOT / "datasets" / "set3",
    ROOT / "datasets" / "Set3",
    ROOT / "datasets" / "poisson",
    Path(config["path_test"]),
]

image_dir = None
for d in candidate_dirs:
    if d.exists() and len(list(d.glob("*.png"))) >= 3:
        image_dir = d
        break

if image_dir is None:
    raise FileNotFoundError(
        "Could not find 3 Set3 PNG images. Expected one of:\n"
        + "\n".join(str(d) for d in candidate_dirs)
        + "\n\nPut the 3 Set3 grayscale images in refs/pds_pnp_ref/datasets/set3/"
    )

image_paths = sorted(image_dir.glob("*.png"))[:3]

if len(image_paths) != 3:
    raise RuntimeError(f"Expected 3 Set3 images, found {len(image_paths)} in {image_dir}")

kernel_path = ROOT / "blur_models" / "blur_1.mat"
if not kernel_path.exists():
    raise FileNotFoundError(f"Missing kernel: {kernel_path}")

model_path = ROOT / "nn" / f"{ARCHITECTURE}.pth"
if not model_path.exists():
    raise FileNotFoundError(
        f"Missing denoiser checkpoint: {model_path}\n"
        "Run git lfs pull, or place DnCNN_nobn_nch_1_nlev_0.01.pth inside refs/pds_pnp_ref/nn/"
    )

out_dir = ROOT / "results_table7_proposed"
out_dir.mkdir(exist_ok=True)

detail_rows = []
matrix_rows = []

print("\nReproducing Table VII: PnP-PDS Proposed only")
print(f"Images: {[p.name for p in image_paths]}")
print(f"Image dir: {image_dir}")
print(f"Kernel: {kernel_path}")
print(f"Model: {model_path}")
print(f"Output: {out_dir}\n")

for task in TASKS:
    task_name = "Deblurring" if task == "blur" else "Inpainting"
    task_psnrs = []

    for eta in ETA_LIST:
        if task == "blur":
            r = 1
            lambda_val = LAMBDA_DEBLUR[eta]
            max_iter = 4800 if eta in [1, 2, 10] else 1200
        else:
            r = SAMPLING_RATE
            lambda_val = LAMBDA_INPAINT[eta]
            max_iter = 12000 if eta in [1, 2, 10] else 3000

        phi, adj_phi = get_observation_operators(
            operator=task,
            path_kernel=str(kernel_path),
            r=r,
        )

        psnrs = []
        ssims = []

        print(
            f"\n=== {task_name} | eta={eta} | lambda={lambda_val} "
            f"| max_iter={max_iter} | r={r} ==="
        )

        for img_index, img_path in enumerate(image_paths, start=1):
            img_true = cv2.imread(str(img_path))
            if img_true is None:
                raise RuntimeError(f"Could not read image: {img_path}")

            img_true = np.asarray(img_true, dtype=np.float32) / 255.0
            img_true = cv2.cvtColor(img_true, cv2.COLOR_BGR2GRAY)

            img_obsrv = phi(img_true)
            img_obsrv = apply_poisson_noise(img_obsrv, eta)
            x0 = img_obsrv / eta

            img_sol, s_sol, c_evolution, psnr_evolution, ssim_evolution, avg_time, other_data = iteration.test_iter(
                x_0=x0,
                x_obsrv=img_obsrv,
                x_true=img_true,
                phi=phi,
                adj_phi=adj_phi,
                gamma1=GAMMA1,
                gamma2=GAMMA2,
                alpha_s=ALPHA_S,
                alpha_n=ALPHA_N,
                myLambda=lambda_val,
                m1=M1,
                m2=M2,
                gammaInADMMStep1=GAMMA_IN_ADMM_STEP1,
                lambydaInStep2=LAMBDA_IN_STEP2,
                gaussian_nl=GAUSSIAN_NL,
                sp_nl=SP_NL,
                poisson_eta=eta,
                path_prox=str(model_path),
                max_iter=max_iter,
                method=METHOD,
                ch=CH,
                r=r,
            )

            final_psnr = float(psnr_evolution[-1])
            final_ssim = float(ssim_evolution[-1])
            final_update = float(c_evolution[-1])

            psnrs.append(final_psnr)
            ssims.append(final_ssim)

            print(
                f"image {img_index}/3 {img_path.name}: "
                f"PSNR={final_psnr:.4f}, SSIM={final_ssim:.4f}, update={final_update:.3e}"
            )

            detail_rows.append({
                "task": task_name,
                "eta": eta,
                "image": img_path.name,
                "psnr": final_psnr,
                "ssim": final_ssim,
                "update": final_update,
                "lambda": lambda_val,
                "max_iter": max_iter,
                "gamma1": GAMMA1,
                "gamma2": GAMMA2,
                "r": r,
                "method": METHOD,
                "architecture": ARCHITECTURE,
            })

        avg_psnr = float(np.mean(psnrs))
        avg_ssim = float(np.mean(ssims))
        task_psnrs.append(avg_psnr)

        print(
            f"AVERAGE {task_name}, eta={eta}: "
            f"PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}"
        )

    matrix_rows.append({
        "task": task_name,
        **{f"eta_{eta}": task_psnrs[i] for i, eta in enumerate(ETA_LIST)},
    })

detail_csv = out_dir / "table7_proposed_detail.csv"
matrix_csv = out_dir / "table7_proposed_matrix.csv"

with open(detail_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
    writer.writeheader()
    writer.writerows(detail_rows)

with open(matrix_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(matrix_rows[0].keys()))
    writer.writeheader()
    writer.writerows(matrix_rows)

print("\nSaved:")
print(detail_csv)
print(matrix_csv)

print("\nFinal Table VII Proposed-only row:")
for row in matrix_rows:
    vals = [row[f"eta_{eta}"] for eta in ETA_LIST]
    print(row["task"] + ": " + " & ".join(f"{v:.2f}" for v in vals))
