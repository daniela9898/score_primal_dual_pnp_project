from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_VP_MODEL_CONFIG = {
    "image_size": 256,
    "num_channels": 256,
    "num_res_blocks": 2,
    "channel_mult": "",
    "learn_sigma": True,
    "class_cond": False,
    "use_checkpoint": False,
    "attention_resolutions": "32,16,8",
    "num_heads": 4,
    "num_head_channels": 64,
    "num_heads_upsample": -1,
    "use_scale_shift_norm": True,
    "dropout": 0.0,
    "resblock_updown": True,
    "use_fp16": False,
    "use_new_attention_order": False,
    "noise_perturbation_type": "vp",
}


class ScoreDenoiser:
    """Thin wrapper around the score_pnp VP score denoiser.

    The score_pnp UNet forward method accepts `(x, sigma)` and returns the
    MMSE/Tweedie denoised estimate. Inputs and outputs are tensors in [0, 1]
    with shape BCHW.
    """

    def __init__(
        self,
        score_repo: str | Path,
        checkpoint: str | Path,
        device: str = "cuda",
        model_config: dict[str, Any] | None = None,
        clamp_output: bool = True,
    ) -> None:
        self.score_repo = Path(score_repo).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.device = torch.device(device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu")
        self.clamp_output = clamp_output

        if not self.score_repo.exists():
            raise FileNotFoundError(f"score_pnp repo not found: {self.score_repo}")
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"score checkpoint not found: {self.checkpoint}")

        # Import from the local score_pnp reference repo.
        sys.path.insert(0, str(self.score_repo))
        from guided_diffusion.unet import create_vp_model  # type: ignore

        cfg = dict(DEFAULT_VP_MODEL_CONFIG)
        if model_config:
            cfg.update(model_config)
        cfg["pretrained_check_point"] = str(self.checkpoint)

        # score_pnp's create_vp_model calls torch.load internally. With PyTorch >=2.6,
        # old checkpoints may fail unless weights_only=False is explicitly used.
        original_torch_load = torch.load

        def torch_load_compat(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        torch.load = torch_load_compat  # type: ignore[assignment]
        try:
            self.model = create_vp_model(**cfg)
        finally:
            torch.load = original_torch_load  # type: ignore[assignment]

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def denoise_torch(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        """Denoise BCHW tensor in [0, 1]."""
        x = x.to(self.device, dtype=torch.float32)
        # The VP score wrapper expects sigma in [0, 1], e.g. 120/255.
        sigma_float = float(sigma)
        out = self.model(x, sigma_float)
        out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
        if self.clamp_output:
            out = out.clamp(0.0, 1.0)
        return out

    def denoise_numpy_chw(self, x_chw: np.ndarray, sigma: float) -> np.ndarray:
        """Denoise a single CHW numpy image in [0, 1]."""
        x = np.asarray(x_chw, dtype=np.float32)
        if x.ndim != 3:
            raise ValueError(f"Expected CHW array, got shape {x.shape}")
        xt = torch.from_numpy(x).unsqueeze(0)
        yt = self.denoise_torch(xt, sigma=sigma)
        return yt.squeeze(0).detach().cpu().numpy().astype(np.float32)
