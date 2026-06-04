#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def save_chw_png(path: Path, img_chw: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.clip(img_chw, 0.0, 1.0)
    if img.ndim == 3:
        img = np.moveaxis(img, 0, -1)
    img_u8 = (255.0 * img).round().astype(np.uint8)
    cv2.imwrite(str(path), img_u8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce Table III protocol using Score-PDS-PnP."
    )

    parser.add_argument(
        "--workspace",
        default="/home/dc3000/shares/mdrive/PhD/projects_6oct2025/score_primal_dual_pnp_project",
    )
    parser.add_argument(
        "--kernel_dir",
        default="/home/dc3000/shares/mdrive/PhD/projects_6oct2025/PnP_PD_paper_hypo/2025-hypomono-IEEE-SPL/Pnp-PD/blur_models",
    )
    parser.add_argument(
        "--kernel_names",
        default="blur_1.mat,blur_2.mat,blur_3.mat,blur_4.mat,blur_5.mat,blur_6.mat,blur_7.mat,blur_8.mat,gaussian_1_6.mat,square_7.mat",
    )
    parser.add_argument(
        "--noise_levels",
        default="0.0025,0.005,0.01,0.02,0.04",
    )
    parser.add_argument("--max_images", type=int, default=7)
    parser.add_argument("--max_iter", type=int, default=1200)
    parser.add_argument("--gamma1", type=float, default=0.5)
    parser.add_argument("--gamma2", type=float, default=0.99)
    parser.add_argument("--sampling_rate", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbose_every", type=int, default=100)
    parser.add_argument("--score_ckpt", default=None)
    parser.add_argument("--result_tag", default="table3_score_pds")
    parser.add_argument("--save_images", action="store_true")

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
        raise FileNotFoundError(
            f"Score checkpoint not found: {score_ckpt}\n"
            "Run git lfs pull, or check score_pnp_ref/pretrained_models/score/imagenet256.pt"
        )

    # Make both repos importable.
    sys.path.insert(0, str(pds_repo))
    sys.path.insert(0, str(workspace / "score_pds_pnp"))

    from pnppds.operators import get_observation_operators, proj_C, proj_l2_ball  # type: ignore
    from utils.utils_noise import add_gaussian_noise  # type: ignore
    from utils.utils_eval import eval_psnr, eval_ssim  # type: ignore
    from score_pds.score_denoiser import ScoreDenoiser
    from score_pds.score_pds_solver import score_pds_gaussian_iter

    image_dir = pds_repo / "datasets" / "imagenet"
    image_paths = sorted(image_dir.glob("*.png"))[: args.max_images]

    if len(image_paths) != args.max_images:
        raise RuntimeError(
            f"Expected {args.max_images} images, found {len(image_paths)} in {image_dir}"
        )

    kernel_dir = Path(args.kernel_dir)
    kernel_names = parse_str_list(args.kernel_names)
    kernel_paths = []

    for name in kernel_names:
        p = kernel_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Missing kernel: {p}")
        kernel_paths.append(p)

    noise_levels = parse_float_list(args.noise_levels)

    # Same alpha values as Table IV, deblurring row.
    # For Score-PDS we keep alpha aligned with the paper protocol.
    alpha_by_noise = {
        0.0025: 0.82,
        0.005: 0.86,
        0.01: 0.92,
        0.02: 0.96,
        0.04: 1.00,
    }

    # Your current best Score-PDS deblurring schedule, but run with Table III max_iter=1200.
    # Values >1 are interpreted by the solver as /255 units.
    score_params_by_noise = {
        0.0025: {"score_sigma_begin": 2.55, "score_sigma_end": 2.55, "denoise_relax": 0.4},
        0.005:  {"score_sigma_begin": 2.55, "score_sigma_end": 2.55, "denoise_relax": 0.4},
        0.01:   {"score_sigma_begin": 80.0, "score_sigma_end": 2.55, "denoise_relax": 0.1},
        0.02:   {"score_sigma_begin": 80.0, "score_sigma_end": 2.55, "denoise_relax": 0.1},
        0.04:   {"score_sigma_begin": 80.0, "score_sigma_end": 2.55, "denoise_relax": 0.2},
    }

    result_root = workspace / "score_pds_pnp" / "results" / args.result_tag
    result_root.mkdir(parents=True, exist_ok=True)
    (result_root / "traces").mkdir(parents=True, exist_ok=True)
    if args.save_images:
        (result_root / "images").mkdir(parents=True, exist_ok=True)

    detail_rows: list[dict[str, object]] = []
    matrix_final_rows: list[dict[str, object]] = []
    matrix_best_rows: list[dict[str, object]] = []

    print("\nLoading score denoiser...")
    denoiser = ScoreDenoiser(score_repo=score_repo, checkpoint=score_ckpt, device=args.device)
    print("Score denoiser loaded.")

    print("\nRunning Score-PDS-PnP with Table III protocol")
    print(f"Images: {[p.name for p in image_paths]}")
    print(f"Kernels: {kernel_names}")
    print(f"Noise levels: {noise_levels}")
    print(f"Max iter: {args.max_iter}")
    print(f"Output: {result_root}\n")

    for nl in noise_levels:
        nl_key = round(float(nl), 6)

        if nl_key not in alpha_by_noise:
            raise ValueError(f"No alpha configured for noise level {nl}")

        if nl_key not in score_params_by_noise:
            raise ValueError(f"No Score-PDS params configured for noise level {nl}")

        alpha = alpha_by_noise[nl_key]
        score_params = score_params_by_noise[nl_key]

        final_kernel_psnrs = []
        best_kernel_psnrs = []

        for kernel_index, kernel_path in enumerate(kernel_paths, start=1):
            phi, adj_phi = get_observation_operators(
                operator="blur",
                path_kernel=str(kernel_path),
                r=args.sampling_rate,
            )
            Id, _ = get_observation_operators(
                operator="Id",
                path_kernel=str(kernel_path),
                r=args.sampling_rate,
            )

            final_psnrs = []
            best_psnrs = []

            print(
                f"\n=== sigma={nl} | kernel {kernel_index}/10: {kernel_path.name} "
                f"| alpha={alpha} | begin={score_params['score_sigma_begin']} "
                f"| end={score_params['score_sigma_end']} | relax={score_params['denoise_relax']} ==="
            )

            for image_index, image_path in enumerate(image_paths, start=1):
                img_true = cv2.imread(str(image_path))
                if img_true is None:
                    raise RuntimeError(f"Could not read image: {image_path}")

                # Original PDS convention: BGR, normalized, CHW.
                img_true = np.asarray(img_true, dtype=np.float32) / 255.0
                img_true = np.moveaxis(img_true, -1, 0)

                img_obsrv = phi(img_true)
                img_obsrv = add_gaussian_noise(img_obsrv, nl, Id)
                x0 = np.copy(img_obsrv)

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
                    r=args.sampling_rate,
                    gamma1=args.gamma1,
                    gamma2=args.gamma2,
                    alpha_n=alpha,
                    sigma_begin=score_params["score_sigma_begin"],
                    sigma_end=score_params["score_sigma_end"],
                    max_iter=args.max_iter,
                    verbose_every=args.verbose_every,
                    denoise_relax=score_params["denoise_relax"],
                )

                final_psnr = float(trace.psnr[-1])
                final_ssim = float(trace.ssim[-1])
                best_iter = int(np.argmax(trace.psnr) + 1)
                best_psnr = float(np.max(trace.psnr))
                best_ssim_at_best_psnr = float(trace.ssim[best_iter - 1])
                final_update = float(trace.update[-1])

                final_psnrs.append(final_psnr)
                best_psnrs.append(best_psnr)

                print(
                    f"image {image_index}/{len(image_paths)} {image_path.name}: "
                    f"final PSNR={final_psnr:.4f}, SSIM={final_ssim:.4f}, "
                    f"best PSNR={best_psnr:.4f} at iter={best_iter}, "
                    f"final update={final_update:.3e}"
                )

                stem = (
                    f"sigma{str(nl).replace('.', 'p')}_"
                    f"k{kernel_index:02d}_{kernel_path.stem}_"
                    f"img{image_index:02d}_{image_path.stem}"
                )

                np.savez(
                    result_root / "traces" / f"TRACE_{stem}.npz",
                    psnr=trace.psnr,
                    ssim=trace.ssim,
                    update=trace.update,
                )

                if args.save_images:
                    save_chw_png(result_root / "images" / f"OBS_{stem}.png", img_obsrv)
                    save_chw_png(result_root / "images" / f"REC_{stem}.png", img_sol)

                detail_rows.append(
                    {
                        "noise": nl,
                        "kernel_index": kernel_index,
                        "kernel_file": kernel_path.name,
                        "image_index": image_index,
                        "image": image_path.name,
                        "final_psnr": final_psnr,
                        "final_ssim": final_ssim,
                        "best_psnr": best_psnr,
                        "best_iter": best_iter,
                        "ssim_at_best_psnr": best_ssim_at_best_psnr,
                        "final_update": final_update,
                        "runtime_per_iter": float(trace.runtime_per_iter),
                        "max_iter": args.max_iter,
                        "gamma1": args.gamma1,
                        "gamma2": args.gamma2,
                        "alpha": alpha,
                        "score_sigma_begin": score_params["score_sigma_begin"],
                        "score_sigma_end": score_params["score_sigma_end"],
                        "denoise_relax": score_params["denoise_relax"],
                    }
                )

            avg_final = float(np.mean(final_psnrs))
            avg_best = float(np.mean(best_psnrs))

            final_kernel_psnrs.append(avg_final)
            best_kernel_psnrs.append(avg_best)

            print(
                f"AVERAGE sigma={nl}, kernel={kernel_index}, file={kernel_path.name}: "
                f"FINAL PSNR={avg_final:.4f}, BEST PSNR={avg_best:.4f}"
            )

        matrix_final_row = {
            "noise": nl,
            **{f"kernel_{i+1}": final_kernel_psnrs[i] for i in range(len(kernel_paths))},
            "average": float(np.mean(final_kernel_psnrs)),
        }

        matrix_best_row = {
            "noise": nl,
            **{f"kernel_{i+1}": best_kernel_psnrs[i] for i in range(len(kernel_paths))},
            "average": float(np.mean(best_kernel_psnrs)),
        }

        matrix_final_rows.append(matrix_final_row)
        matrix_best_rows.append(matrix_best_row)

        print("\nTABLE III STYLE ROW, FINAL ITERATION")
        print(
            f"sigma={nl}: "
            + " ".join(f"{v:.2f}" for v in final_kernel_psnrs)
            + f" | Average {matrix_final_row['average']:.2f}"
        )

        print("DIAGNOSTIC ROW, BEST OVER ITERATIONS")
        print(
            f"sigma={nl}: "
            + " ".join(f"{v:.2f}" for v in best_kernel_psnrs)
            + f" | Average {matrix_best_row['average']:.2f}"
        )

    detail_csv = result_root / "table3_score_pds_detail.csv"
    matrix_final_csv = result_root / "table3_score_pds_matrix_final.csv"
    matrix_best_csv = result_root / "table3_score_pds_matrix_best.csv"

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

    print("\nFinal Table III-style Score-PDS matrix:")
    for row in matrix_final_rows:
        vals = [row[f"kernel_{i}"] for i in range(1, len(kernel_paths) + 1)]
        print(
            f"sigma={row['noise']}: "
            + " & ".join(f"{v:.2f}" for v in vals)
            + f" & {row['average']:.2f}"
        )


if __name__ == "__main__":
    main()
