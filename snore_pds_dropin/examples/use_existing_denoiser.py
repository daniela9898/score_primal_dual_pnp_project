"""
Example: using SNORE-PDS with your own denoiser from the existing repository.

Run from the root of score_primal_dual_pnp_project after copying this package:

    python examples/use_existing_denoiser.py

You must edit `load_your_denoiser()` to point to the denoiser-loading code that
already exists in your repo, e.g. DnCNN / DRUNet / score denoiser.
"""

from __future__ import annotations

import torch

from snore_pds import IdentityOperator, SNOREConfig, SNOREPDSConfig, snore_pds_gaussian
from snore_pds.io_utils import load_image, save_image


def load_your_denoiser(device: str):
    # TODO Daniela: replace this with the denoiser loader in your repo.
    # The object should support either:
    #   denoiser(x_noisy, sigma)
    # or
    #   denoiser(x_noisy)
    from snore_pds.snore_prior import GaussianSmoothingDenoiser
    return GaussianSmoothingDenoiser(kernel_size=5, sigma=1.0).to(device).eval()


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_true = load_image("data/example.png", device=device)  # change this path
    noise_std = 2.55 / 255.0
    op = IdentityOperator()
    y = (op.forward(x_true) + noise_std * torch.randn_like(x_true)).clamp(0, 1)

    denoiser = load_your_denoiser(device)
    cfg = SNOREPDSConfig(
        n_iter=300,
        gamma1=0.25,
        gamma2=0.99,
        lambda_snore=1e-5,
        noise_std=noise_std,
        sigma_begin=5.0 / 255.0,
        sigma_end=2.55 / 255.0,
        snore=SNOREConfig(mc_samples=1, residual_mode="clean", clip_denoised=True),
    )
    x_hat, logs = snore_pds_gaussian(y, op, denoiser, cfg, ground_truth=x_true)
    save_image(x_hat, "results_snore_pds/example_restored.png")
    print(logs[-1])


if __name__ == "__main__":
    main()
