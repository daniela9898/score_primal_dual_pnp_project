from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class LinearOperator:
    """Small interface for linear forward operators Phi and adjoints Phi^*."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def norm_bound(self) -> float:
        """Return a safe upper bound on ||Phi||_op.

        The PDS step condition used in the attached PnP-PDS paper is
            gamma1 * gamma2 * (||Phi||^2 + 1) < 1
        for the Gaussian constrained version with an extra box dual.
        """
        return 1.0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class IdentityOperator(LinearOperator):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        return y

    def norm_bound(self) -> float:
        return 1.0


@dataclass
class MaskOperator(LinearOperator):
    """Random or fixed-mask inpainting operator.

    mask must broadcast to the image tensor shape, typically [1,1,H,W] or [B,C,H,W].
    We implement Phi x = mask * x and Phi^* y = mask * y, so ||Phi|| <= 1.
    """

    mask: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mask.to(device=x.device, dtype=x.dtype) * x

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        return self.mask.to(device=y.device, dtype=y.dtype) * y

    def norm_bound(self) -> float:
        return 1.0


@dataclass
class BlurFFT(LinearOperator):
    """Circular convolution using FFT.

    kernel: 2D tensor [kh, kw] or [1,1,kh,kw]. It is padded and shifted to image_size.
    This class assumes the same blur kernel for every batch/channel.
    """

    kernel: torch.Tensor
    image_size: Tuple[int, int]
    _otf_cache: Optional[torch.Tensor] = None
    _otf_shape: Optional[Tuple[int, int, torch.device, torch.dtype]] = None

    def _otf(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.image_size
        key = (H, W, x.device, x.dtype)
        if self._otf_cache is not None and self._otf_shape == key:
            return self._otf_cache

        k = self.kernel.to(device=x.device, dtype=x.dtype)
        if k.ndim == 4:
            k = k[0, 0]
        if k.ndim != 2:
            raise ValueError("kernel must be 2D or [1,1,kh,kw]")
        kh, kw = k.shape
        pad = torch.zeros((H, W), device=x.device, dtype=x.dtype)
        pad[:kh, :kw] = k
        # Put the kernel center at (0, 0) for circular convolution.
        pad = torch.roll(pad, shifts=(-kh // 2, -kw // 2), dims=(0, 1))
        otf = torch.fft.rfft2(pad)
        self._otf_cache = otf
        self._otf_shape = key
        return otf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        otf = self._otf(x)
        X = torch.fft.rfft2(x, dim=(-2, -1))
        y = torch.fft.irfft2(X * otf[None, None, ...], s=x.shape[-2:], dim=(-2, -1))
        return y

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        otf = self._otf(y)
        Y = torch.fft.rfft2(y, dim=(-2, -1))
        x = torch.fft.irfft2(Y * torch.conj(otf)[None, None, ...], s=y.shape[-2:], dim=(-2, -1))
        return x

    def norm_bound(self) -> float:
        # If kernel is normalized, this is usually <= 1. We return 1 as a safe default.
        return 1.0


@dataclass
class DownsampleOperator(LinearOperator):
    """Simple decimation downsampler and zero-insertion adjoint.

    This is useful for quick SR experiments, not a full degradation model.
    """

    scale: int = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., :: self.scale, :: self.scale]

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        B, C, h, w = y.shape
        out = torch.zeros((B, C, h * self.scale, w * self.scale), device=y.device, dtype=y.dtype)
        out[..., :: self.scale, :: self.scale] = y
        return out

    def norm_bound(self) -> float:
        return 1.0


def gaussian_kernel2d(size: int, sigma: float, device=None, dtype=None) -> torch.Tensor:
    """Utility for toy/demo blur kernels."""
    if size % 2 == 0:
        raise ValueError("size must be odd")
    ax = torch.arange(size, device=device, dtype=dtype) - size // 2
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return k / k.sum()
