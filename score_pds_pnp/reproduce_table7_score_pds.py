#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_float_or_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def log_sigma_schedule(begin: float, end: float, max_iter: int) -> np.ndarray:
    b = float(begin)
    e = float(end)
    if b > 1.0:
        b /= 255.0
    if e > 1.0:
        e /= 255.0
    return np.logspace(np.log10(b), np.log10(e), int(max_iter)).astype(np.float32)


def denoise_gray_with_rgb_score(
    score_denoiser,
    x_gray: np.ndarray,
    sigma: float,
    clip_input: bool = False,
) -> np.ndarray:
    x = np.asarray(x_gray, dtype=np.float32)

    if clip_input:
        x = np.clip(x, 0.0, 1.0)

    x_chw = np.stack([x, x, x], axis=0).astype(np.float32)
    y_chw = score_denoiser.denoise_numpy_chw(x_chw, sigma=sigma)

    # Convert pseudo-RGB score output back to grayscale.
    y_gray = np.mean(y_chw, axis=0).astype(np.float32)
    return y_gray


def score_pds_poisson_iter(
    x_0: np.ndarray,
    x_obsrv: np.ndarray,
    x_true: np.ndarray,
    phi,
    adj_phi,
    score_denoiser,
    prox_GKL,
    proj_C,
    eval_psnr,
    eval_ssim,
    poisson_eta: float,
    lambda_val: float,
    gamma1: float = 0.5,
    gamma2: float = 0.99,
    sigma_begin: float = 80.0,
    sigma_end: float = 2.55,
    denoise_relax: float = 0.1,
    max_iter: int = 1200,
    verbose_every: int = 100,
    clip_denoiser_input: bool = False,
):
    x_n = np.asarray(x_0, dtype=np.float32).copy()
    y_n = np.zeros_like(x_n, dtype=np.float32)
    y2_n = np.zeros_like(x_n, dtype=np.float32)

    psnr = np.zeros(max_iter, dtype=np.float32)
    ssim = np.zeros(max_iter, dtype=np.float32)
    update = np.zeros(max_iter, dtype=np.float32)
    sigma_schedule = log_sigma_schedule(sigma_begin, sigma_end, max_iter)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.process_time()

    for i in range(max_iter):
        x_prev = x_n.copy()

        # Score-PDS denoising step.
        denoiser_input = x_n - gamma1 * (adj_phi(y_n) + y2_n)
        x_score = denoise_gray_with_rgb_score(
            score_denoiser,
            denoiser_input,
            sigma=float(sigma_schedule[i]),
            clip_input=clip_denoiser_input,
        )
        x_n = (1.0 - denoise_relax) * denoiser_input + denoise_relax * x_score
        x_n = np.clip(x_n, 0.0, 1.0).astype(np.float32)

        # Poisson-PDS dual updates, matching Algorithm 2 / original code.
        y_tmp = y_n + gamma2 * phi(2.0 * x_n - x_prev)
        y2_tmp = y2_n + gamma2 * (2.0 * x_n - x_prev)

        y_n = y_tmp - gamma2 * prox_GKL(
            y_tmp / gamma2,
            lambda_val / gamma2,
            poisson_eta,
            x_obsrv,
        )
        y2_n = y2_tmp - gamma2 * proj_C(y2_tmp / gamma2)

        denom = max(float(np.linalg.norm(x_prev.reshape(-1), 2)), 1e-12)
        update[i] = float(np.linalg.norm((x_n - x_prev).reshape(-1), 2) / denom)
        psnr[i] = float(eval_psnr(x_true, x_n))
        ssim[i] = float(eval_ssim(x_true, x_n))

        if verbose_every > 0 and ((i + 1) % verbose_every == 0 or i == 0 or i == max_iter - 1):
            print(
                f"iter {i+1:05d}/{max_iter} "
                f"sigma={float(sigma_schedule[i]):.5f} "
                f"PSNR={psnr[i]:.3f} SSIM={ssim[i]:.4f} update={update[i]:.3e}",
                flush=True,
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    runtime_per_iter = (time.process_time() - start) / max_iter
    return x_n, {
        "psnr": psnr,
        "ssim": ssim,
        "update": update,
        "runtime_per_iter": runtime_per_iter,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Table VII Poisson protocol using Score-PDS-PnP."
    )

    parser.add_argument(
        "--workspace",
        default="/home/dc3000/shares/mdrive/PhD/projects_6oct2025/score_primal_dual_pnp_project",
    )
    parser.add_argument("--eta_list", default="1,2,10,50,100,200")
    parser.add_argument("--tasks", default="blur,random_sampling")
    parser.add_argument("--max_images", type=int, default=3)

    parser.add_argument("--gamma1", type=float, default=0.5)
    parser.add_argument("--gamma2", type=float, default=0.99)
    parser.add_argument("--sampling_rate", type=float, default=0.8)

    # Initial Score-PDS schedule. Tune later if needed.
    parser.add_argument("--score_sigma_begin", type=float, default=80.0)
    parser.add_argument("--score_sigma_end", type=float, default=2.55)
    parser.add_argument("--denoise_relax", type=float, default=0.1)
    parser.add_argument("--clip_denoiser_input", action="store_true")

    parser.add_argument("--max_iter_override", type=int, default=None)
    parser.add_argument("--verbose_every", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score_ckpt", default=None)
    parser.add_argument("--result_tag", default="table7_score_pds")

    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    pds_repo = workspace / "refs" / "pds_pnp_ref"
    score_repo = workspace / "refs" / "score_pnp_ref"

    if args.score_ckpt is None:
        score_ckpt = score_repo / "pretrained_models" / "score" / "imagenet256.pt"
    else:
        score_ckpt = Path(args.score_ckpt).resolve()

    if not pds_repo.exists():
        raise FileNotFoundError(f"PDS repo not found: {pds_repo}")
    if not score_repo.exists():
        raise FileNotFoundError(f"score_pnp repo not found: {score_repo}")
    if not score_ckpt.exists():
        raise FileNotFoundError(f"Score checkpoint not found: {score_ckpt}")

    sys.path.insert(0, str(pds_repo))
    sys.path.insert(0, str(workspace / "score_pds_pnp"))

    from pnppds.operators import get_observation_operators, prox_GKL, proj_C  # type: ignore
    from utils.utils_noise import apply_poisson_noise  # type: ignore
    from utils.utils_eval import eval_psnr, eval_ssim  # type: ignore
    from score_pds.score_denoiser import ScoreDenoiser

    eta_list = parse_float_or_int_list(args.eta_list)
    tasks = parse_str_list(args.tasks)

    lambda_deblur = {
        1: 2.00e-3,
        2: 2.00e-3,
        10: 1.50e-3,
        50: 1.25e-3,
        100: 1.25e-3,
        200: 1.00e-3,
    }

    lambda_inpaint = {
        1: 1.25e-3,
        2: 1.25e-3,
        10: 1.00e-3,
        50: 7.50e-4,
        100: 5.00e-4,
        200: 5.00e-4,
    }

    candidate_dirs = [
        pds_repo / "datasets" / "set3",
        pds_repo / "datasets" / "Set3",
    ]

    image_dir = None
    for d in candidate_dirs:
        if d.exists() and len(list(d.glob("*.png"))) >= args.max_images:
            image_dir = d
            break

    if image_dir is None:
        raise FileNotFoundError(
            "Could not find Set3 PNG images in refs/pds_pnp_ref/datasets/set3 or Set3."
        )

    image_paths = sorted(image_dir.glob("*.png"))[: args.max_images]

    kernel_path = pds_repo / "blur_models" / "blur_1.mat"
    if not kernel_path.exists():
        raise FileNotFoundError(f"Missing kernel: {kernel_path}")

    result_root = workspace / "score_pds_pnp" / "results" / args.result_tag
    result_root.mkdir(parents=True, exist_ok=True)
    (result_root / "traces").mkdir(parents=True, exist_ok=True)

    print("\nLoading score denoiser...")
    denoiser = ScoreDenoiser(score_repo=score_repo, checkpoint=score_ckpt, device=args.device)
    print("Score denoiser loaded.")

    print("\nRunning Score-PDS-PnP with Table VII protocol")
    print(f"Images: {[p.name for p in image_paths]}")
    print(f"Image dir: {image_dir}")
    print(f"Kernel: {kernel_path}")
    print(f"Etas: {eta_list}")
    print(f"Tasks: {tasks}")
    print(f"Score sigma: {args.score_sigma_begin} -> {args.score_sigma_end}")
    print(f"Denoise relax: {args.denoise_relax}")
    print(f"Clip denoiser input: {args.clip_denoiser_input}")
    print(f"Output: {result_root}\n")

    detail_rows = []
    matrix_final_rows = []
    matrix_best_rows = []

    for task in tasks:
        if task not in ["blur", "random_sampling"]:
            raise ValueError(f"Unknown task: {task}")

        task_name = "Deblurring" if task == "blur" else "Inpainting"
        final_task_psnrs = []
        best_task_psnrs = []

        for eta in eta_list:
            if task == "blur":
                r = 1.0
                lambda_val = lambda_deblur[eta]
                max_iter = 4800 if eta in [1, 2, 10] else 1200
            else:
                r = args.sampling_rate
                lambda_val = lambda_inpaint[eta]
                max_iter = 12000 if eta in [1, 2, 10] else 3000

            if args.max_iter_override is not None:
                max_iter = args.max_iter_override

            phi, adj_phi = get_observation_operators(
                operator=task,
                path_kernel=str(kernel_path),
                r=r,
            )

            final_psnrs = []
            best_psnrs = []

            print(
                f"\n=== {task_name} | eta={eta} | lambda={lambda_val} "
                f"| max_iter={max_iter} | r={r} ==="
            )

            for image_index, image_path in enumerate(image_paths, start=1):
                img_true = cv2.imread(str(image_path))
                if img_true is None:
                    raise RuntimeError(f"Could not read image: {image_path}")

                img_true = np.asarray(img_true, dtype=np.float32) / 255.0
                img_true = cv2.cvtColor(img_true, cv2.COLOR_BGR2GRAY)

                img_obsrv = phi(img_true)
                img_obsrv = apply_poisson_noise(img_obsrv, eta)
                x0 = img_obsrv / eta

                img_sol, trace = score_pds_poisson_iter(
                    x_0=x0,
                    x_obsrv=img_obsrv,
                    x_true=img_true,
                    phi=phi,
                    adj_phi=adj_phi,
                    score_denoiser=denoiser,
                    prox_GKL=prox_GKL,
                    proj_C=proj_C,
                    eval_psnr=eval_psnr,
                    eval_ssim=eval_ssim,
                    poisson_eta=eta,
                    lambda_val=lambda_val,
                    gamma1=args.gamma1,
                    gamma2=args.gamma2,
                    sigma_begin=args.score_sigma_begin,
                    sigma_end=args.score_sigma_end,
                    denoise_relax=args.denoise_relax,
                    max_iter=max_iter,
                    verbose_every=args.verbose_every,
                    clip_denoiser_input=args.clip_denoiser_input,
                )

                final_psnr = float(trace["psnr"][-1])
                final_ssim = float(trace["ssim"][-1])
                best_iter = int(np.argmax(trace["psnr"]) + 1)
                best_psnr = float(np.max(trace["psnr"]))
                best_ssim = float(trace["ssim"][best_iter - 1])
                final_update = float(trace["update"][-1])

                final_psnrs.append(final_psnr)
                best_psnrs.append(best_psnr)

                print(
                    f"image {image_index}/{len(image_paths)} {image_path.name}: "
                    f"final PSNR={final_psnr:.4f}, SSIM={final_ssim:.4f}, "
                    f"best PSNR={best_psnr:.4f} at iter={best_iter}, "
                    f"final update={final_update:.3e}"
                )

                stem = f"{task}_eta{eta}_img{image_index:02d}_{image_path.stem}"

                np.savez(
                    result_root / "traces" / f"TRACE_{stem}.npz",
                    psnr=trace["psnr"],
                    ssim=trace["ssim"],
                    update=trace["update"],
                )

                detail_rows.append(
                    {
                        "task": task_name,
                        "eta": eta,
                        "image": image_path.name,
                        "final_psnr": final_psnr,
                        "final_ssim": final_ssim,
                        "best_psnr": best_psnr,
                        "best_iter": best_iter,
                        "ssim_at_best_psnr": best_ssim,
                        "final_update": final_update,
                        "runtime_per_iter": float(trace["runtime_per_iter"]),
                        "lambda": lambda_val,
                        "max_iter": max_iter,
                        "gamma1": args.gamma1,
                        "gamma2": args.gamma2,
                        "r": r,
                        "score_sigma_begin": args.score_sigma_begin,
                        "score_sigma_end": args.score_sigma_end,
                        "denoise_relax": args.denoise_relax,
                        "clip_denoiser_input": args.clip_denoiser_input,
                    }
                )

            avg_final = float(np.mean(final_psnrs))
            avg_best = float(np.mean(best_psnrs))

            final_task_psnrs.append(avg_final)
            best_task_psnrs.append(avg_best)

            print(
                f"AVERAGE {task_name}, eta={eta}: "
                f"FINAL PSNR={avg_final:.4f}, BEST PSNR={avg_best:.4f}"
            )

        matrix_final_rows.append({
            "task": task_name,
            **{f"eta_{eta}": final_task_psnrs[i] for i, eta in enumerate(eta_list)},
        })

        matrix_best_rows.append({
            "task": task_name,
            **{f"eta_{eta}": best_task_psnrs[i] for i, eta in enumerate(eta_list)},
        })

    detail_csv = result_root / "table7_score_pds_detail.csv"
    matrix_final_csv = result_root / "table7_score_pds_matrix_final.csv"
    matrix_best_csv = result_root / "table7_score_pds_matrix_best.csv"

    with open(detail_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    with open(matrix_final_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(matrix_final_rows[0].keys()))
        writer.writeheader()
        writer.writerows(matrix_final_rows)

    with open(matrix_best_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(matrix_best_rows[0].keys()))
        writer.writeheader()
        writer.writerows(matrix_best_rows)

    print("\nSaved:")
    print(detail_csv)
    print(matrix_final_csv)
    print(matrix_best_csv)

    print("\nFinal Table VII-style Score-PDS matrix:")
    for row in matrix_final_rows:
        vals = [row[f"eta_{eta}"] for eta in eta_list]
        print(row["task"] + ": " + " & ".join(f"{v:.2f}" for v in vals))

    print("\nDiagnostic best-over-iterations matrix:")
    for row in matrix_best_rows:
        vals = [row[f"eta_{eta}"] for eta in eta_list]
        print(row["task"] + ": " + " & ".join(f"{v:.2f}" for v in vals))


if __name__ == "__main__":
    main()
