from __future__ import annotations

import torch


def project_box(x: torch.Tensor, low: float = 0.0, high: float = 1.0) -> torch.Tensor:
    return torch.clamp(x, min=low, max=high)


def project_l2_ball(z: torch.Tensor, center: torch.Tensor, radius: float, eps: float = 1e-12) -> torch.Tensor:
    """Project each batch element of z onto an l2 ball centered at center.

    z and center shape: [B,C,H,W]. radius may be a float or tensor broadcastable to [B].
    """
    center = center.to(device=z.device, dtype=z.dtype)
    diff = z - center
    flat = diff.reshape(diff.shape[0], -1)
    norm = torch.linalg.norm(flat, dim=1).clamp_min(eps)
    if not torch.is_tensor(radius):
        radius_t = torch.full_like(norm, float(radius))
    else:
        radius_t = radius.to(device=z.device, dtype=z.dtype).reshape(-1)
        if radius_t.numel() == 1:
            radius_t = radius_t.expand_as(norm)
    scale = torch.minimum(torch.ones_like(norm), radius_t / norm)
    while scale.ndim < diff.ndim:
        scale = scale[..., None]
    return center + diff * scale


def moreau_prox_conjugate_indicator_l2_ball(w: torch.Tensor, gamma: float, center: torch.Tensor, radius: float) -> torch.Tensor:
    """prox_{gamma * indicator_B^*}(w) via Moreau:
        prox_{gamma h^*}(w) = w - gamma * prox_{h/gamma}(w/gamma)
    For an indicator, prox is projection onto the set.
    """
    return w - gamma * project_l2_ball(w / gamma, center=center, radius=radius)


def moreau_prox_conjugate_box(w: torch.Tensor, gamma: float, low: float = 0.0, high: float = 1.0) -> torch.Tensor:
    return w - gamma * project_box(w / gamma, low=low, high=high)
