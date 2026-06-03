from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ScorePDSTrace:
    psnr: np.ndarray
    ssim: np.ndarray
    update: np.ndarray
    runtime_per_iter: float


def log_sigma_schedule(begin: float, end: float, max_iter: int) -> np.ndarray:
    """Log-spaced sigma schedule in [0,1]. begin/end may be given as >1 values in /255 units."""
    b = float(begin)
    e = float(end)
    if b > 1.0:
        b /= 255.0
    if e > 1.0:
        e /= 255.0
    return np.logspace(np.log10(b), np.log10(e), int(max_iter)).astype(np.float32)


def score_pds_gaussian_iter(
    x_0: np.ndarray,
    x_obsrv: np.ndarray,
    x_true: np.ndarray,
    phi,
    adj_phi,
    score_denoiser,
    proj_l2_ball,
    proj_C,
    eval_psnr,
    eval_ssim,
    gaussian_nl: float,
    sp_nl: float,
    r: float,
    gamma1: float = 0.5,
    gamma2: float = 0.99,
    alpha_n: float = 0.82,
    sigma_begin: float = 120.0,
    sigma_end: float = 10.0,
    max_iter: int = 1200,
    verbose_every: int = 50,
    denoise_relax: float = 1.0,
) -> tuple[np.ndarray, ScorePDSTrace]:
    """Score-PDS-PnP for Gaussian constrained data fidelity.

    This is Algorithm 1 from the PDS repo, with the denoising step replaced by
    the score_pnp VP score denoiser.
    """
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
        denoiser_input = x_n - gamma1 * (adj_phi(y_n) + y2_n)
        x_score = score_denoiser.denoise_numpy_chw(denoiser_input, sigma=float(sigma_schedule[i]))
        x_n = (1.0 - denoise_relax) * denoiser_input + denoise_relax * x_score
        x_n = np.clip(x_n, 0.0, 1.0).astype(np.float32)

        y_tmp = y_n + gamma2 * phi(2.0 * x_n - x_prev)
        y2_tmp = y2_n + gamma2 * (2.0 * x_n - x_prev)

        y_n = y_tmp - gamma2 * proj_l2_ball(y_tmp / gamma2, alpha_n, gaussian_nl, sp_nl, x_obsrv, r)
        y2_n = y2_tmp - gamma2 * proj_C(y2_tmp / gamma2)

        denom = max(float(np.linalg.norm(x_prev.reshape(-1), 2)), 1e-12)
        update[i] = float(np.linalg.norm((x_n - x_prev).reshape(-1), 2) / denom)
        psnr[i] = float(eval_psnr(x_true, x_n))
        ssim[i] = float(eval_ssim(x_true, x_n))

        if verbose_every > 0 and ((i + 1) % verbose_every == 0 or i == 0 or i == max_iter - 1):
            print(
                f"iter {i+1:04d}/{max_iter} "
                f"sigma={float(sigma_schedule[i]):.5f} "
                f"PSNR={psnr[i]:.3f} SSIM={ssim[i]:.4f} update={update[i]:.3e}",
                flush=True,
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    runtime_per_iter = (time.process_time() - start) / max_iter
    return x_n, ScorePDSTrace(psnr=psnr, ssim=ssim, update=update, runtime_per_iter=runtime_per_iter)
