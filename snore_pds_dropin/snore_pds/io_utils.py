from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from PIL import Image
import numpy as np


def load_image(path: str | Path, device: str = "cpu", grayscale: bool = False) -> torch.Tensor:
    img = Image.open(path)
    img = img.convert("L" if grayscale else "RGB")
    arr = np.asarray(img).astype("float32") / 255.0
    if grayscale:
        arr = arr[None, ...]
    else:
        arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr)[None].to(device)


def save_image(x: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = x.detach().clamp(0, 1).cpu()[0]
    if x.shape[0] == 1:
        arr = (x[0].numpy() * 255.0 + 0.5).astype("uint8")
        img = Image.fromarray(arr, mode="L")
    else:
        arr = (x.numpy().transpose(1, 2, 0) * 255.0 + 0.5).astype("uint8")
        img = Image.fromarray(arr, mode="RGB")
    img.save(path)


def load_kernel_txt(path: str | Path, device: str = "cpu") -> torch.Tensor:
    arr = np.loadtxt(path).astype("float32")
    k = torch.from_numpy(arr).to(device)
    return k / k.sum()


def load_mask(path: str | Path, device: str = "cpu", channels: int = 3) -> torch.Tensor:
    img = Image.open(path).convert("L")
    arr = np.asarray(img).astype("float32") / 255.0
    mask = torch.from_numpy((arr > 0.5).astype("float32"))[None, None].to(device)
    if channels != 1:
        mask = mask.repeat(1, channels, 1, 1)
    return mask
