from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from .operators import LinearOperator
from .projections import moreau_prox_conjugate_box, moreau_prox_conjugate_indicator_l2_ball
from .snore_prior import SNOREConfig, snore_gradient, make_sigma_schedule


@dataclass
class SNOREPDSConfig:
    """Configuration for SNORE-PDS.

    This implements an experimental merge:
      - primal-dual splitting handles data/constraint terms;
      - SNORE gives a stochastic gradient prior.

    For Gaussian constrained restoration we solve approximately:
        min_x  lambda_snore R_SNORE(x)
        s.t.   ||Phi x - y||_2 <= eps,  x in [0,1]^K.

    The PDS condition inherited from the attached paper for the nonsmooth
    dual part is roughly gamma1*gamma2*(||Phi||^2+1)<1. The SNORE gradient
    adds a smooth/nonconvex stochastic term, so for stability choose smaller
    gamma1 or lambda_snore if curves explode.
    """

    n_iter: int = 300
    gamma1: float = 0.45
    gamma2: float = 0.99
    lambda_snore: float = 1e-4
    eps: Optional[float] = None
    eps_factor: float = 1.0
    noise_std: float = 2.55 / 255.0
    box: bool = True
    rho: float = 1.0
    sigma_begin: float = 5.0 / 255.0
    sigma_end: float = 2.55 / 255.0
    sigma_schedule: str = "geometric"
    snore: SNOREConfig = field(default_factory=SNOREConfig)
    seed: int = 0
    verbose: bool = True
    log_every: int = 25
    clip_each_iter: bool = False


def _safe_eps(y: torch.Tensor, noise_std: float, eps_factor: float) -> float:
    # For AWGN, epsilon ~= sigma * sqrt(K), per sample. Here K includes C*H*W.
    K = y[0].numel()
    return float(eps_factor * noise_std * (K ** 0.5))


def _check_pds_steps(config: SNOREPDSConfig, op: LinearOperator) -> None:
    lhs = config.gamma1 * config.gamma2 * (op.norm_bound() ** 2 + (1.0 if config.box else 0.0))
    if lhs >= 1.0:
        raise ValueError(
            f"Unsafe PDS steps: gamma1*gamma2*(||Phi||^2 + box) = {lhs:.4f} >= 1. "
            "Decrease gamma1 or gamma2."
        )


@torch.no_grad()
def snore_pds_gaussian(
    y: torch.Tensor,
    op: LinearOperator,
    denoiser: Callable,
    config: SNOREPDSConfig,
    x0: Optional[torch.Tensor] = None,
    ground_truth: Optional[torch.Tensor] = None,
    callback: Optional[Callable[[int, torch.Tensor, dict], None]] = None,
) -> tuple[torch.Tensor, list[dict]]:
    """SNORE-PDS for Gaussian constrained restoration.

    Args:
        y: observed/degraded image tensor [B,C,H,W] in [0,1].
        op: Phi linear operator.
        denoiser: callable D(x_noisy, sigma) or D(x_noisy).
        config: solver hyperparameters.
        x0: optional initialization; if None, Phi^* y is used.
        ground_truth: optional for diagnostics only.
        callback: optional function called as callback(iter, x, info).

    Returns:
        restored image x, list of logs.
    """
    _check_pds_steps(config, op)
    device = y.device
    gen = torch.Generator(device=device)
    gen.manual_seed(int(config.seed))

    x = op.adjoint(y).clone() if x0 is None else x0.clone()
    if config.box:
        x = x.clamp(0.0, 1.0)

    w_data = torch.zeros_like(y)
    w_box = torch.zeros_like(x)
    eps = float(config.eps) if config.eps is not None else _safe_eps(y, config.noise_std, config.eps_factor)
    sigmas = make_sigma_schedule(config.n_iter, config.sigma_begin, config.sigma_end, config.sigma_schedule)  # type: ignore[arg-type]

    logs: list[dict] = []
    prev_x = x.clone()
    for it in range(1, config.n_iter + 1):
        sn_cfg = config.snore
        sn_cfg = SNOREConfig(
            sigma=sigmas[it - 1],
            sigma_min=sn_cfg.sigma_min,
            mc_samples=sn_cfg.mc_samples,
            residual_mode=sn_cfg.residual_mode,
            clip_denoised=sn_cfg.clip_denoised,
            detach_denoiser=sn_cfg.detach_denoiser,
        )
        g_snore, sn_info = snore_gradient(x, denoiser, sn_cfg, generator=gen)

        primal_grad = op.adjoint(w_data)
        if config.box:
            primal_grad = primal_grad + w_box
        primal_grad = primal_grad + config.lambda_snore * g_snore

        x_tilde = x - config.gamma1 * primal_grad
        if config.clip_each_iter:
            x_tilde = x_tilde.clamp(0.0, 1.0)
        x_new = config.rho * x_tilde + (1.0 - config.rho) * x

        extrap = 2.0 * x_new - x
        w_data_tmp = w_data + config.gamma2 * op.forward(extrap)
        w_data = moreau_prox_conjugate_indicator_l2_ball(w_data_tmp, config.gamma2, center=y, radius=eps)

        if config.box:
            w_box_tmp = w_box + config.gamma2 * extrap
            w_box = moreau_prox_conjugate_box(w_box_tmp, config.gamma2, low=0.0, high=1.0)

        data_res = torch.linalg.norm((op.forward(x_new) - y).reshape(y.shape[0], -1), dim=1).mean()
        update = torch.linalg.norm((x_new - prev_x).reshape(x.shape[0], -1), dim=1).mean() / (
            torch.linalg.norm(prev_x.reshape(x.shape[0], -1), dim=1).mean().clamp_min(1e-12)
        )
        info = {
            "iter": it,
            "eps": eps,
            "sigma_snore": sn_info["sigma"],
            "data_residual": float(data_res.item()),
            "relative_update": float(update.item()),
            "snore_grad_norm": sn_info["grad_norm"],
            "denoise_residual": sn_info["denoise_residual"],
        }
        if ground_truth is not None:
            mse = torch.mean((x_new - ground_truth) ** 2).clamp_min(1e-12)
            info["psnr"] = float(-10.0 * torch.log10(mse).item())

        logs.append(info)
        if callback is not None:
            callback(it, x_new, info)
        if config.verbose and (it == 1 or it % config.log_every == 0 or it == config.n_iter):
            msg = (
                f"[SNORE-PDS] it={it:04d} sigma={info['sigma_snore']:.4g} "
                f"res={info['data_residual']:.4e} upd={info['relative_update']:.3e} "
                f"g={info['snore_grad_norm']:.3e}"
            )
            if "psnr" in info:
                msg += f" psnr={info['psnr']:.3f}"
            print(msg)

        prev_x = x
        x = x_new

    if config.box:
        x = x.clamp(0.0, 1.0)
    return x, logs


@torch.no_grad()
def snore_pds_l2_unconstrained(
    y: torch.Tensor,
    op: LinearOperator,
    denoiser: Callable,
    config: SNOREPDSConfig,
    x0: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, list[dict]]:
    """Simpler baseline: gradient descent on 0.5||Phi x-y||^2 + lambda R_SNORE(x).

    This is NOT the main proposed method, but useful as a debugging baseline.
    """
    device = y.device
    gen = torch.Generator(device=device)
    gen.manual_seed(int(config.seed))
    x = op.adjoint(y).clone() if x0 is None else x0.clone()
    sigmas = make_sigma_schedule(config.n_iter, config.sigma_begin, config.sigma_end, config.sigma_schedule)  # type: ignore[arg-type]
    logs: list[dict] = []
    for it in range(1, config.n_iter + 1):
        sn_cfg = config.snore
        sn_cfg = SNOREConfig(
            sigma=sigmas[it - 1],
            sigma_min=sn_cfg.sigma_min,
            mc_samples=sn_cfg.mc_samples,
            residual_mode=sn_cfg.residual_mode,
            clip_denoised=sn_cfg.clip_denoised,
            detach_denoiser=sn_cfg.detach_denoiser,
        )
        g_snore, sn_info = snore_gradient(x, denoiser, sn_cfg, generator=gen)
        data_grad = op.adjoint(op.forward(x) - y)
        x_new = x - config.gamma1 * (data_grad + config.lambda_snore * g_snore)
        if config.box:
            x_new = x_new.clamp(0.0, 1.0)
        data_res = torch.linalg.norm((op.forward(x_new) - y).reshape(y.shape[0], -1), dim=1).mean()
        logs.append({"iter": it, "data_residual": float(data_res.item()), **sn_info})
        x = x_new
    return x, logs
