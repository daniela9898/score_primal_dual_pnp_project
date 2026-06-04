from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr(x: torch.Tensor, ref: torch.Tensor, data_range: float = 1.0, eps: float = 1e-12) -> float:
    mse = torch.mean((x.detach() - ref.detach()) ** 2).clamp_min(eps)
    return float(20.0 * torch.log10(torch.tensor(data_range, device=x.device, dtype=x.dtype)) - 10.0 * torch.log10(mse))


def ssim_torch(x: torch.Tensor, ref: torch.Tensor, data_range: float = 1.0, window_size: int = 11) -> float:
    """Minimal global SSIM-ish implementation for quick diagnostics.

    For paper tables, prefer skimage.metrics.structural_similarity or piq/torchmetrics.
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    pad = window_size // 2
    weight = torch.ones((x.shape[1], 1, window_size, window_size), device=x.device, dtype=x.dtype)
    weight = weight / (window_size * window_size)

    mu_x = F.conv2d(x, weight, padding=pad, groups=x.shape[1])
    mu_y = F.conv2d(ref, weight, padding=pad, groups=ref.shape[1])
    sigma_x = F.conv2d(x * x, weight, padding=pad, groups=x.shape[1]) - mu_x**2
    sigma_y = F.conv2d(ref * ref, weight, padding=pad, groups=ref.shape[1]) - mu_y**2
    sigma_xy = F.conv2d(x * ref, weight, padding=pad, groups=x.shape[1]) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
    return float(ssim_map.mean().item())
