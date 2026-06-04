"""
SNORE-PDS: experimental stochastic denoising regularization inside a
primal-dual splitting solver for constrained image restoration.

This is research code meant to be dropped into Daniela's
score_primal_dual_pnp_project repository.
"""

from .operators import LinearOperator, IdentityOperator, MaskOperator, BlurFFT, DownsampleOperator
from .snore_prior import snore_gradient, SNOREConfig, make_sigma_schedule
from .solver import SNOREPDSConfig, snore_pds_gaussian, snore_pds_l2_unconstrained
from .metrics import psnr, ssim_torch

__all__ = [
    "LinearOperator",
    "IdentityOperator",
    "MaskOperator",
    "BlurFFT",
    "DownsampleOperator",
    "snore_gradient",
    "SNOREConfig",
    "make_sigma_schedule",
    "SNOREPDSConfig",
    "snore_pds_gaussian",
    "snore_pds_l2_unconstrained",
    "psnr",
    "ssim_torch",
]

__version__ = "0.2.0-table5"
