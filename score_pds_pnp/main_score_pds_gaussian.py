#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_noise_levels(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def save_chw_png(path: Path, img_chw: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.clip(img_chw, 0.0, 1.0)
    if img.ndim == 3:
        img = np.moveaxis(img, 0, -1)
    img_u8 = (255.0 * img).round().astype(np.uint8)
    # cv2 expects BGR; the PDS repo reads BGR and keeps channel order, so this is fine for debugging.
    cv2.imwrite(str(path), img_u8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental Score-PDS-PnP for Gaussian deblurring/inpainting.")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--task", choices=["deblur", "inpaint", "both"], default="deblur")
    parser.add_argument("--noise_levels", default="0.0025")
    parser.add_argument("--max_iter", type=int, default=1200)
    parser.add_argument("--max_images", type=int, default=7)
    parser.add_argument("--gamma1", type=float, default=0.5)
    parser.add_argument("--gamma2", type=float, default=0.99)
    parser.add_argument("--alpha", type=float, default=0.82)
    parser.add_argument("--sampling_rate", type=float, default=0.8)
    parser.add_argument("--score_sigma_begin", type=float, default=120.0, help="Use 120 for 120/255.")
    parser.add_argument("--score_sigma_end", type=float, default=10.0, help="Use 10 for 10/255.")
    parser.add_argument("--denoise_relax", type=float, default=1.0, help="Blend score denoiser: x=(1-theta)input + theta*D_score(input).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbose_every", type=int, default=50)
    parser.add_argument("--score_ckpt", default=None)
    parser.add_argument("--result_tag", default="score_pds")
    args = parser.parse_args()

    # Interpret CLI sigma values in /255 units when the user passes values >= 1.
    # Example: 2.55 -> 2.55/255 = 0.01, 1 -> 1/255.
    # Values already in [0,1), e.g. 0.01, are kept unchanged.
    def _cli_sigma_to_unit_interval(v: float) -> float:
        v = float(v)
        return v / 255.0 if v >= 1.0 else v

    args.score_sigma_begin = _cli_sigma_to_unit_interval(args.score_sigma_begin)
    args.score_sigma_end = _cli_sigma_to_unit_interval(args.score_sigma_end)

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

    # Make PDS utilities importable.
    sys.path.insert(0, str(pds_repo))
    # Make this package importable even when launched as a script.
    sys.path.insert(0, str(workspace / "score_pds_pnp"))

    from pnppds.operators import get_observation_operators, proj_C, proj_l2_ball  # type: ignore
    from utils.utils_noise import add_gaussian_noise  # type: ignore
    from utils.utils_eval import eval_psnr, eval_ssim  # type: ignore
    from score_pds.score_denoiser import ScoreDenoiser
    from score_pds.score_pds_solver import score_pds_gaussian_iter

    image_dir = pds_repo / "datasets" / "imagenet"
    kernel_path = pds_repo / "blur_models" / "blur_1.mat"
    paths = sorted(image_dir.glob("*.png"))[: args.max_images]
    if not paths:
        raise FileNotFoundError(f"No png images found in {image_dir}")

    tasks = ["blur", "random_sampling"] if args.task == "both" else (["blur"] if args.task == "deblur" else ["random_sampling"])
    noise_levels = parse_noise_levels(args.noise_levels)

    result_root = workspace / "score_pds_pnp" / "results" / f"{args.result_tag}_gaussian"
    result_root.mkdir(parents=True, exist_ok=True)
    (result_root / "images").mkdir(parents=True, exist_ok=True)
    (result_root / "traces").mkdir(parents=True, exist_ok=True)
    csv_path = result_root / "summary.csv"

    print("Loading score denoiser...")
    denoiser = ScoreDenoiser(score_repo=score_repo, checkpoint=score_ckpt, device=args.device)
    print("Score denoiser loaded.")

    rows: list[dict[str, object]] = []

    for nl in noise_levels:
        for deg_op in tasks:
            if deg_op == "blur":
                r = 1.0
            else:
                r = args.sampling_rate

            phi, adj_phi = get_observation_operators(operator=deg_op, path_kernel=str(kernel_path), r=r)
            Id, _ = get_observation_operators("Id", str(kernel_path), r)

            psnrs = []
            ssims = []
            for idx, path_img in enumerate(paths, start=1):
                img_true = cv2.imread(str(path_img))
                if img_true is None:
                    raise RuntimeError(f"Could not read image: {path_img}")
                img_true = np.asarray(img_true, dtype=np.float32) / 255.0
                img_true = np.moveaxis(img_true, -1, 0)  # CHW, BGR order as in original repo

                img_obsrv = phi(img_true)
                if deg_op in ["blur", "Id"]:
                    img_obsrv = add_gaussian_noise(img_obsrv, nl, Id)
                elif deg_op == "random_sampling":
                    img_obsrv = add_gaussian_noise(img_obsrv, nl, phi)
                x0 = np.copy(img_obsrv)

                print(
                    f"\n[{deg_op} sigma={nl}] image {idx}/{len(paths)}: {path_img.name}",
                    flush=True,
                )
                img_sol, trace = score_pds_gaussian_iter(
                    x_0=x0,
                    x_obsrv=img_obsrv,
                    x_true=img_true,
                    phi=phi,
                    adj_phi=adj_phi,
                    score_denoiser=denoiser,
                    proj_l2_ball=proj_l2_ball,
                    proj_C=proj_C,
                    eval_psnr=eval_psnr,
                    eval_ssim=eval_ssim,
                    gaussian_nl=nl,
                    sp_nl=0.0,
                    r=r,
                    gamma1=args.gamma1,
                    gamma2=args.gamma2,
                    alpha_n=args.alpha,
                    sigma_begin=args.score_sigma_begin,
                    sigma_end=args.score_sigma_end,
                    max_iter=args.max_iter,
                    verbose_every=args.verbose_every,
                    denoise_relax=args.denoise_relax,
                )

                psnr = float(trace.psnr[-1])
                ssim = float(trace.ssim[-1])
                psnrs.append(psnr)
                ssims.append(ssim)
                print(f"FINAL image {idx}: PSNR={psnr:.4f} SSIM={ssim:.4f}")

                stem = f"{deg_op}_sigma{nl}_img{idx:02d}_{path_img.stem}"
                save_chw_png(result_root / "images" / f"OBS_{stem}.png", img_obsrv)
                save_chw_png(result_root / "images" / f"REC_{stem}.png", img_sol)
                np.savez(
                    result_root / "traces" / f"TRACE_{stem}.npz",
                    psnr=trace.psnr,
                    ssim=trace.ssim,
                    update=trace.update,
                )

                rows.append(
                    {
                        "task": deg_op,
                        "noise": nl,
                        "image": path_img.name,
                        "psnr": psnr,
                        "ssim": ssim,
                        "runtime_per_iter": trace.runtime_per_iter,
                        "max_iter": args.max_iter,
                        "gamma1": args.gamma1,
                        "gamma2": args.gamma2,
                        "alpha": args.alpha,
                        "score_sigma_begin": args.score_sigma_begin,
                        "score_sigma_end": args.score_sigma_end,
                        "denoise_relax": args.denoise_relax,
                    }
                )

            print(
                f"\nAVERAGE {deg_op} sigma={nl}: "
                f"PSNR={np.mean(psnrs):.4f} SSIM={np.mean(ssims):.4f}",
                flush=True,
            )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved summary: {csv_path}")


if __name__ == "__main__":
    main()
