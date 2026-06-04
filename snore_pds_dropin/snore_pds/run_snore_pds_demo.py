from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .io_utils import load_image, save_image, load_kernel_txt, load_mask
from .operators import IdentityOperator, MaskOperator, BlurFFT, gaussian_kernel2d
from .snore_prior import GaussianSmoothingDenoiser, SNOREConfig
from .solver import SNOREPDSConfig, snore_pds_gaussian


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Smoke-test SNORE-PDS with a toy Gaussian-smoothing denoiser.")
    p.add_argument("--input", type=str, required=True, help="Clean image used to synthesize an observation.")
    p.add_argument("--output", type=str, default="results_snore_pds/demo_restored.png")
    p.add_argument("--task", type=str, default="deblur", choices=["denoise", "deblur", "inpaint"])
    p.add_argument("--kernel-txt", type=str, default=None, help="Optional text file containing a blur kernel.")
    p.add_argument("--mask", type=str, default=None, help="Optional binary mask image for inpainting.")
    p.add_argument("--noise-std", type=float, default=2.55/255.0)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--lambda-snore", type=float, default=1e-5)
    p.add_argument("--gamma1", type=float, default=0.25)
    p.add_argument("--gamma2", type=float, default=0.99)
    p.add_argument("--sigma-begin", type=float, default=5.0/255.0)
    p.add_argument("--sigma-end", type=float, default=2.55/255.0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)
    device = args.device
    x_true = load_image(args.input, device=device)
    B, C, H, W = x_true.shape

    if args.task == "denoise":
        op = IdentityOperator()
    elif args.task == "deblur":
        kernel = load_kernel_txt(args.kernel_txt, device=device) if args.kernel_txt else gaussian_kernel2d(9, 1.6, device=device, dtype=x_true.dtype)
        op = BlurFFT(kernel=kernel, image_size=(H, W))
    else:
        if args.mask is not None:
            mask = load_mask(args.mask, device=device, channels=C)
        else:
            mask = (torch.rand((1, 1, H, W), device=device) > 0.5).float().repeat(1, C, 1, 1)
        op = MaskOperator(mask)

    y_clean = op.forward(x_true)
    y = (y_clean + args.noise_std * torch.randn_like(y_clean)).clamp(0, 1)

    # This toy denoiser is only to verify the code path. Replace with your real denoiser.
    denoiser = GaussianSmoothingDenoiser(kernel_size=5, sigma=1.0).to(device).eval()

    cfg = SNOREPDSConfig(
        n_iter=args.iters,
        gamma1=args.gamma1,
        gamma2=args.gamma2,
        lambda_snore=args.lambda_snore,
        noise_std=args.noise_std,
        eps_factor=1.0,
        sigma_begin=args.sigma_begin,
        sigma_end=args.sigma_end,
        seed=args.seed,
        verbose=True,
        log_every=max(1, args.iters // 10),
        snore=SNOREConfig(mc_samples=1, residual_mode="clean", clip_denoised=True),
    )
    x_hat, logs = snore_pds_gaussian(y, op, denoiser, cfg, ground_truth=x_true)
    save_image(x_hat, args.output)
    print(f"Saved restored image to {args.output}")
    print(f"Final log: {logs[-1]}")


if __name__ == "__main__":
    main()
