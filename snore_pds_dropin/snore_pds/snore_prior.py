from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Sequence

import torch

ResidualMode = Literal["clean", "noisy"]


@dataclass
class SNOREConfig:
    """Configuration for the SNORE stochastic denoising-gradient estimator.

    sigma is expressed in the same scale as image pixels. For images in [0,1],
    sigma=2.55/255 corresponds to a denoiser trained at noise level 2.55/255.

    residual_mode="clean" implements the practical SNORE-style estimator
        grad ~= (x - D_sigma(x + sigma * eps)) / sigma^2
    which injects noise only inside the denoiser input.

    residual_mode="noisy" implements the more literal score/Tweedie residual
        grad ~= (x + sigma * eps - D_sigma(x + sigma * eps)) / sigma^2
    and is noisier in practice.
    """

    sigma: float = 2.55 / 255.0
    sigma_min: float = 1e-4
    mc_samples: int = 1
    residual_mode: ResidualMode = "clean"
    clip_denoised: bool = False
    detach_denoiser: bool = True


def make_sigma_schedule(
    n_iter: int,
    sigma_begin: float,
    sigma_end: float,
    mode: Literal["constant", "linear", "geometric"] = "geometric",
) -> list[float]:
    if n_iter <= 0:
        return []
    if mode == "constant" or n_iter == 1:
        return [float(sigma_begin)] * n_iter
    if mode == "linear":
        return torch.linspace(float(sigma_begin), float(sigma_end), n_iter).tolist()
    if mode == "geometric":
        sigma_begin = max(float(sigma_begin), 1e-12)
        sigma_end = max(float(sigma_end), 1e-12)
        return torch.exp(torch.linspace(torch.log(torch.tensor(sigma_begin)), torch.log(torch.tensor(sigma_end)), n_iter)).tolist()
    raise ValueError(f"unknown sigma schedule mode: {mode}")


def _call_denoiser(denoiser: Callable, x_noisy: torch.Tensor, sigma: float) -> torch.Tensor:
    """Call denoisers with several common signatures.

    Supported signatures:
      denoiser(x, sigma_float)
      denoiser(x, sigma_tensor)
      denoiser(x)

    For DRUNet-like denoisers that expect a noise map concatenated to x, wrap
    the model yourself and expose one of the above signatures.
    """
    try:
        return denoiser(x_noisy, sigma)
    except TypeError:
        pass
    try:
        sigma_t = torch.tensor(float(sigma), device=x_noisy.device, dtype=x_noisy.dtype)
        return denoiser(x_noisy, sigma_t)
    except TypeError:
        pass
    return denoiser(x_noisy)


@torch.no_grad()
def snore_gradient(
    x: torch.Tensor,
    denoiser: Callable,
    config: SNOREConfig,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, dict]:
    """Monte-Carlo estimator of the SNORE regularization gradient.

    Returns:
        grad: tensor with same shape as x
        info: small diagnostics dict
    """
    sigma = max(float(config.sigma), float(config.sigma_min))
    grads = []
    denoised_list = []
    for _ in range(int(config.mc_samples)):
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        x_noisy = x + sigma * noise
        d = _call_denoiser(denoiser, x_noisy, sigma)
        if config.clip_denoised:
            d = d.clamp(0.0, 1.0)
        base = x_noisy if config.residual_mode == "noisy" else x
        grads.append((base - d) / (sigma * sigma))
        denoised_list.append(d)
    grad = torch.stack(grads, dim=0).mean(dim=0)
    dmean = torch.stack(denoised_list, dim=0).mean(dim=0)
    info = {
        "sigma": sigma,
        "grad_norm": float(torch.linalg.norm(grad.detach().reshape(grad.shape[0], -1), dim=1).mean().item()),
        "denoise_residual": float(torch.mean(torch.abs(x.detach() - dmean.detach())).item()),
    }
    return grad, info


class GaussianSmoothingDenoiser(torch.nn.Module):
    """Tiny toy denoiser for smoke tests only.

    This is NOT a publishable prior. Use your real DnCNN/DRUNet/score denoiser.
    """

    def __init__(self, kernel_size: int = 5, sigma: float = 1.0):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        ax = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        yy, xx = torch.meshgrid(ax, ax, indexing="ij")
        k = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        k = k / k.sum()
        self.register_buffer("kernel", k[None, None])
        self.pad = kernel_size // 2

    def forward(self, x: torch.Tensor, sigma: float | torch.Tensor | None = None) -> torch.Tensor:
        B, C, H, W = x.shape
        k = self.kernel.to(device=x.device, dtype=x.dtype).repeat(C, 1, 1, 1)
        return torch.nn.functional.conv2d(x, k, padding=self.pad, groups=C)
