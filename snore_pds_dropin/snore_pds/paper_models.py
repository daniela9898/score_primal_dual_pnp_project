from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """DnCNN-like architecture used by the PnP-PDS paper.

    Matches the public paper repo: 20 conv layers, 64 channels, no BN,
    LeakyReLU activations, residual skip x_out = conv(x) + x.
    """

    def __init__(self, ch_in: int = 3, ch_out: int = 3, ch: int = 64, depth: int = 20):
        super().__init__()
        self.depth = int(depth)
        self.in_conv = nn.Conv2d(ch_in, ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_list = nn.ModuleList(
            [nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True) for _ in range(self.depth - 2)]
        )
        self.out_conv = nn.Conv2d(ch, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.nl_list = nn.ModuleList([nn.LeakyReLU() for _ in range(self.depth - 1)])

    def forward(self, x: torch.Tensor, sigma: Optional[float] = None) -> torch.Tensor:
        h = self.in_conv(x)
        h = self.nl_list[0](h)
        for i in range(self.depth - 2):
            h = self.conv_list[i](h)
            h = self.nl_list[i + 1](h)
        return self.out_conv(h) + x


class DeepInvDnCNN(nn.Module):
    """DnCNN architecture used in score_pnp/deepinv for dncnn_sigma2_color.pth."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3, depth: int = 20, nf: int = 64, bias: bool = True):
        super().__init__()
        self.depth = int(depth)
        self.in_conv = nn.Conv2d(in_channels, nf, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv_list = nn.ModuleList(
            [nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=bias) for _ in range(self.depth - 2)]
        )
        self.out_conv = nn.Conv2d(nf, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.nl_list = nn.ModuleList([nn.ReLU() for _ in range(self.depth - 1)])

    def forward(self, x: torch.Tensor, sigma: Optional[float] = None) -> torch.Tensor:
        h = self.in_conv(x)
        h = self.nl_list[0](h)
        for i in range(self.depth - 2):
            h = self.conv_list[i](h)
            h = self.nl_list[i + 1](h)
        return self.out_conv(h) + x


def _torch_load(path: str | Path, map_location="cpu"):
    """torch.load wrapper compatible with old and new PyTorch versions."""
    path = str(path)
    kwargs = {"map_location": map_location}
    try:
        sig = inspect.signature(torch.load)
        if "weights_only" in sig.parameters:
            kwargs["weights_only"] = False
    except Exception:
        pass
    return torch.load(path, **kwargs)


def _extract_state_dict(obj):
    """Accept a state_dict, a dict containing one, a nn.Module, or DataParallel."""
    if isinstance(obj, nn.Module):
        if hasattr(obj, "module") and isinstance(obj.module, nn.Module):
            return obj.module.state_dict()
        return obj.state_dict()
    if isinstance(obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "network", "params"]:
            if key in obj:
                val = obj[key]
                if isinstance(val, nn.Module):
                    return val.module.state_dict() if hasattr(val, "module") else val.state_dict()
                if isinstance(val, dict):
                    return val
        # Looks like a raw state_dict.
        if obj and all(isinstance(k, str) for k in obj.keys()):
            return obj
    raise TypeError(f"Could not extract state_dict from checkpoint object of type {type(obj)}")


def _clean_state_dict_keys(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        kk = str(k)
        for prefix in ["module.", "model.", "denoiser.", "network."]:
            if kk.startswith(prefix):
                kk = kk[len(prefix) :]
        out[kk] = v
    return out


def load_paper_dncnn(ckpt: str | Path, channels: int = 3, device: str | torch.device = "cpu", paper_repo: str | Path | None = None) -> nn.Module:
    """Load the FNE DnCNN from the Convergent PnP-PDS paper repo.

    The public repo saved older PyTorch objects in some releases, so we add
    paper_repo to sys.path before torch.load when provided.
    """
    if paper_repo:
        p = str(Path(paper_repo).resolve())
        if p not in sys.path:
            sys.path.insert(0, p)

    ckpt = Path(ckpt)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Paper DnCNN checkpoint not found: {ckpt}")
    raw = _torch_load(ckpt, map_location="cpu")
    sd = _clean_state_dict_keys(_extract_state_dict(raw))
    net = SimpleCNN(ch_in=channels, ch_out=channels, ch=64, depth=20)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[warn] Paper DnCNN load_state_dict strict=False: missing={len(missing)}, unexpected={len(unexpected)}")
        if missing[:5]:
            print(f"       first missing: {missing[:5]}")
        if unexpected[:5]:
            print(f"       first unexpected: {unexpected[:5]}")
    return net.to(device).eval()


def load_score_dncnn(ckpt: str | Path, channels: int = 3, device: str | torch.device = "cpu") -> nn.Module:
    """Load the DeepInverse/score_pnp DnCNN checkpoint."""
    ckpt = Path(ckpt)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Score-PnP DnCNN checkpoint not found: {ckpt}")
    raw = _torch_load(ckpt, map_location="cpu")
    sd = _clean_state_dict_keys(_extract_state_dict(raw))
    net = DeepInvDnCNN(in_channels=channels, out_channels=channels, depth=20, nf=64)
    net.load_state_dict(sd, strict=True)
    return net.to(device).eval()


class ScorePNPVPDenoiser(nn.Module):
    """Wrapper around score_pnp's VP diffusion MMSE denoiser.

    It imports `guided_diffusion.unet.create_vp_model` from a local clone of
    https://github.com/wustl-cig/score_pnp. The model's forward(x, sigma)
    already performs the parameter matching from sigma to diffusion timestep.
    """

    def __init__(self, score_repo: str | Path, ckpt: str | Path, device: str | torch.device = "cpu", image_size: int = 256):
        super().__init__()
        score_repo = Path(score_repo).resolve()
        if not score_repo.exists():
            raise FileNotFoundError(f"score_pnp repo not found: {score_repo}")
        if str(score_repo) not in sys.path:
            sys.path.insert(0, str(score_repo))
        from guided_diffusion.unet import create_vp_model  # type: ignore

        self.model = create_vp_model(
            image_size=int(image_size),
            num_channels=256,
            num_res_blocks=2,
            channel_mult="",
            learn_sigma=True,
            class_cond=False,
            use_checkpoint=False,
            attention_resolutions="32,16,8",
            num_heads=4,
            num_head_channels=64,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            dropout=0.0,
            resblock_updown=True,
            use_fp16=False,
            use_new_attention_order=False,
            pretrained_check_point=str(ckpt),
            noise_perturbation_type="vp",
        )
        self.model.to(device).eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor, sigma: float | torch.Tensor | None = None) -> torch.Tensor:
        if sigma is None:
            sigma = 0.01
        if isinstance(sigma, torch.Tensor):
            sigma_val = float(sigma.detach().flatten()[0].item())
        else:
            sigma_val = float(sigma)
        # score_pnp models were trained on [-1,1]-style diffusion data in many configs,
        # but the public wrapper is used with normalize=False in their experiments.
        # We follow their forward contract: input and output are tensors in the same scale.
        return self.model(x, sigma_val).clamp(0.0, 1.0)


def load_score_pnp_denoiser(
    ckpt: str | Path,
    score_repo: str | Path,
    channels: int = 3,
    device: str | torch.device = "cpu",
    denoiser_type: str = "auto",
    score_model_image_size: int = 256,
) -> nn.Module:
    ckpt = Path(ckpt)
    name = ckpt.name.lower()
    denoiser_type = denoiser_type.lower()
    if denoiser_type == "auto":
        if "dncnn" in name:
            denoiser_type = "dncnn"
        elif "drunet" in name:
            denoiser_type = "drunet"
        else:
            denoiser_type = "vp_score"
    if denoiser_type == "dncnn":
        return load_score_dncnn(ckpt, channels=channels, device=device)
    if denoiser_type in {"vp", "vp_score", "score", "diffusion"}:
        return ScorePNPVPDenoiser(score_repo=score_repo, ckpt=ckpt, device=device, image_size=score_model_image_size)
    raise ValueError(f"Unsupported score_pnp denoiser_type={denoiser_type!r}. Use auto, dncnn, or vp_score.")
