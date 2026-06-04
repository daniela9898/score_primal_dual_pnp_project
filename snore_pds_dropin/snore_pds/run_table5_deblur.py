from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .metrics import psnr, ssim_torch
from .paper_models import load_paper_dncnn, load_score_pnp_denoiser
from .projections import moreau_prox_conjugate_box, moreau_prox_conjugate_indicator_l2_ball
from .snore_prior import SNOREConfig
from .solver import SNOREPDSConfig, snore_pds_gaussian
from .operators import LinearOperator, gaussian_kernel2d
from .io_utils import save_image


KNOWN_7 = [
    "ILSVRC2012_val_00000491.JPEG.png",
    "ILSVRC2012_val_00000524.JPEG.png",
    "ILSVRC2012_val_00000664.JPEG.png",
    "ILSVRC2012_val_00000976.JPEG.png",
    "ILSVRC2012_val_00001260.JPEG.png",
    "ILSVRC2012_val_00001668.JPEG.png",
    "ILSVRC2012_val_00001737.JPEG.png",
]


class PaperBlurTorch(LinearOperator):
    """Torch implementation of the blur/adjoint in the PnP-PDS paper repo.

    It follows pnppds/operators.py from the public repo:
      forward: circular pad by (l//2+1, l//2), fft2 kernel, crop [..., l:, l:]
      adjoint: circular pad by (l//2, l//2), fft2 conj(kernel), crop [..., :-l+1, :-l+1]
    """

    def __init__(self, kernel: torch.Tensor):
        if kernel.ndim == 4:
            kernel = kernel[0, 0]
        if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
            raise ValueError("PaperBlurTorch expects a square 2D kernel")
        self.kernel = kernel.detach().float()

    def _fft_conv_padded(self, x: torch.Tensor, adjoint: bool = False) -> torch.Tensor:
        l = int(self.kernel.shape[0])
        k = self.kernel.to(device=x.device, dtype=x.dtype)
        if not adjoint:
            # np.pad with ((l//2+1,l//2),(l//2+1,l//2)), wrap
            pad = (l // 2 + 1, l // 2, l // 2 + 1, l // 2)  # left, right, top, bottom
            xp = F.pad(x, pad, mode="circular")
            K = torch.fft.fft2(k, s=xp.shape[-2:])
            Y = torch.fft.ifft2(K[None, None, ...] * torch.fft.fft2(xp, dim=(-2, -1)), dim=(-2, -1)).real
            return Y[..., l:, l:]
        else:
            pad = (l // 2, l // 2, l // 2, l // 2)
            xp = F.pad(x, pad, mode="circular")
            K = torch.fft.fft2(k, s=xp.shape[-2:])
            Y = torch.fft.ifft2(torch.conj(K)[None, None, ...] * torch.fft.fft2(xp, dim=(-2, -1)), dim=(-2, -1)).real
            return Y[..., : -l + 1, : -l + 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._fft_conv_padded(x, adjoint=False)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        return self._fft_conv_padded(y, adjoint=True)

    def norm_bound(self) -> float:
        return 1.0


def _center_crop_or_resize(img: Image.Image, crop_size: int) -> Image.Image:
    if crop_size <= 0:
        return img
    w, h = img.size
    if w < crop_size or h < crop_size:
        # Rare for ImageNet crops. Keep aspect simple and deterministic.
        img = img.resize((max(w, crop_size), max(h, crop_size)), Image.BICUBIC)
        w, h = img.size
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    return img.crop((left, top, left + crop_size, top + crop_size))


def load_image_center_crop(path: str | Path, crop_size: int, device: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = _center_crop_or_resize(img, crop_size)
    arr = np.asarray(img).astype("float32") / 255.0
    arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr)[None].to(device)


def _candidate_paths(root: Path, name: str) -> Iterable[Path]:
    p = Path(name)
    if p.is_absolute():
        yield p
    yield root / name
    yield root / f"{name}.png"
    yield root / f"{name}.jpg"
    yield root / f"{name}.JPEG"
    yield root / f"{name}.JPEG.png"
    if name.endswith(".JPEG"):
        yield root / f"{name}.png"
    if name.endswith(".jpg") or name.endswith(".png") or name.endswith(".JPEG"):
        yield root / Path(name).name


def resolve_images(dataset_root: str | Path, image_list: str | Path | None, num_images: int) -> list[Path]:
    root = Path(dataset_root)
    names: list[str] = []
    if image_list and Path(image_list).is_file():
        names = [ln.strip() for ln in Path(image_list).read_text().splitlines() if ln.strip() and not ln.strip().startswith("#")]
    elif (root / "imagenet_val_7.txt").is_file():
        names = [ln.strip() for ln in (root / "imagenet_val_7.txt").read_text().splitlines() if ln.strip()]
    elif (root / "imagenet_val.txt").is_file():
        names = [ln.strip() for ln in (root / "imagenet_val.txt").read_text().splitlines() if ln.strip()]
    else:
        names = KNOWN_7

    paths: list[Path] = []
    missing: list[str] = []
    for name in names:
        found = None
        for cand in _candidate_paths(root, name):
            if cand.is_file():
                found = cand
                break
        if found is not None and found not in paths:
            paths.append(found)
        else:
            missing.append(name)
        if len(paths) >= num_images:
            break

    if len(paths) < num_images:
        all_imgs = sorted(
            [p for ext in ("*.png", "*.jpg", "*.jpeg", "*.JPEG") for p in root.glob(ext)]
        )
        for p in all_imgs:
            if p not in paths:
                paths.append(p)
            if len(paths) >= num_images:
                break

    if missing:
        print(f"[warn] Some listed images were missing/skipped, first few: {missing[:5]}")
    if not paths:
        raise FileNotFoundError(f"No images found under {root}")
    if len(paths) < num_images:
        print(f"[warn] Requested {num_images} images but found only {len(paths)}")
    return paths[:num_images]


def load_kernel(args, device: str) -> torch.Tensor:
    # Priority 1: explicit path.
    candidates: list[Path] = []
    if args.kernel_file:
        candidates.append(Path(args.kernel_file))
    if args.paper_repo:
        candidates.append(Path(args.paper_repo) / "blur_models" / "blur_1.mat")
    if args.score_pnp_repo:
        candidates.append(Path(args.score_pnp_repo) / "data" / "blur_kernels" / "Levin09.npy")
        candidates.append(Path(args.score_pnp_repo) / "blur_kernels" / "Levin09.npy")

    for path in candidates:
        if path.is_file():
            suffix = path.suffix.lower()
            if suffix == ".mat":
                import scipy.io

                d = scipy.io.loadmat(path)
                key = "blur" if "blur" in d else next(k for k in d.keys() if not k.startswith("__"))
                arr = np.asarray(d[key]).astype("float32")
                print(f"[kernel] loaded {path} key={key} shape={arr.shape}")
                k = torch.from_numpy(arr)
                return (k / k.sum()).to(device)
            if suffix == ".npy":
                arr = np.load(path, allow_pickle=True)
                idx = ord(args.kernel_id.lower()[0]) - ord("a") if args.kernel_id.isalpha() else int(args.kernel_id)
                # DeepInverse Levin09.npy is often [10,kh,kw] or object array.
                if arr.dtype == object:
                    ker = np.asarray(arr[idx]).astype("float32")
                elif arr.ndim == 3:
                    ker = arr[idx].astype("float32")
                elif arr.ndim == 4:
                    ker = arr[idx, 0].astype("float32")
                elif arr.ndim == 2:
                    ker = arr.astype("float32")
                else:
                    raise ValueError(f"Unsupported npy kernel shape {arr.shape} in {path}")
                print(f"[kernel] loaded {path} kernel_id={args.kernel_id} shape={ker.shape}")
                k = torch.from_numpy(ker)
                return (k / k.sum()).to(device)
            if suffix in {".txt", ".csv"}:
                arr = np.loadtxt(path).astype("float32")
                k = torch.from_numpy(arr)
                return (k / k.sum()).to(device)

    print("[warn] Could not find paper kernel. Falling back to toy Gaussian 9x9 sigma=1.6. This is NOT exact Table V.")
    return gaussian_kernel2d(9, 1.6, device=device, dtype=torch.float32)


def make_observation(x_true: torch.Tensor, op: LinearOperator, noise_std: float) -> torch.Tensor:
    y_clean = op.forward(x_true)
    # Match the paper repo: np.random.seed(1234) inside add_gaussian_noise for each image.
    gen = torch.Generator(device=x_true.device)
    gen.manual_seed(1234)
    noise = noise_std * torch.randn(y_clean.shape, device=x_true.device, dtype=x_true.dtype, generator=gen)
    return y_clean + noise


def pnp_pds_paper_gaussian(
    y: torch.Tensor,
    op: LinearOperator,
    denoiser,
    n_iter: int,
    gamma1: float,
    gamma2: float,
    eps: float,
    x0: Optional[torch.Tensor] = None,
    gt: Optional[torch.Tensor] = None,
    verbose: bool = True,
    log_every: int = 100,
) -> tuple[torch.Tensor, list[dict]]:
    x = op.adjoint(y).clone() if x0 is None else x0.clone()
    x = x.clamp(0.0, 1.0)
    w_data = torch.zeros_like(y)
    w_box = torch.zeros_like(x)
    logs: list[dict] = []
    prev = x.clone()
    for it in range(1, n_iter + 1):
        inp = x - gamma1 * (op.adjoint(w_data) + w_box)
        x_new = denoiser(inp).clamp(0.0, 1.0)
        extrap = 2.0 * x_new - x
        w_data_tmp = w_data + gamma2 * op.forward(extrap)
        w_data = moreau_prox_conjugate_indicator_l2_ball(w_data_tmp, gamma2, center=y, radius=eps)
        w_box_tmp = w_box + gamma2 * extrap
        w_box = moreau_prox_conjugate_box(w_box_tmp, gamma2, low=0.0, high=1.0)
        rel = torch.linalg.norm((x_new - prev).reshape(x.shape[0], -1), dim=1).mean() / torch.linalg.norm(prev.reshape(x.shape[0], -1), dim=1).mean().clamp_min(1e-12)
        res = torch.linalg.norm((op.forward(x_new) - y).reshape(y.shape[0], -1), dim=1).mean()
        info = {"iter": it, "relative_update": float(rel.item()), "data_residual": float(res.item())}
        if gt is not None:
            info["psnr"] = psnr(x_new, gt)
        logs.append(info)
        if verbose and (it == 1 or it % log_every == 0 or it == n_iter):
            msg = f"[PnP-PDS-paper] it={it:04d} res={info['data_residual']:.4e} upd={info['relative_update']:.3e}"
            if "psnr" in info:
                msg += f" psnr={info['psnr']:.3f}"
            print(msg)
        prev = x
        x = x_new
    return x.clamp(0.0, 1.0), logs


def autodetect_ckpt(root: str | Path | None, patterns: list[str]) -> Optional[Path]:
    if not root:
        return None
    root = Path(root)
    if not root.exists():
        return None
    files = [p for ext in ("*.pth", "*.pt", "*.ckpt") for p in root.rglob(ext)]
    for pat in patterns:
        for p in files:
            if pat.lower() in p.name.lower() or pat.lower() in str(p).lower():
                return p
    return files[0] if files else None


def main():
    parser = argparse.ArgumentParser(description="Run Table V-style Gaussian deblurring at sigma=0.0025 for PnP-PDS/SNORE-PDS.")
    parser.add_argument("--method", choices=["pnp_pds_paper", "snore_pds"], required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--image-list", default=None)
    parser.add_argument("--num-images", type=int, default=7)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--task", default="deblur", choices=["deblur"])
    parser.add_argument("--kernel-id", default="a")
    parser.add_argument("--kernel-file", default=None)
    parser.add_argument("--noise-std", type=float, default=0.0025)
    parser.add_argument("--eps-alpha", type=float, default=0.82)
    parser.add_argument("--iters", type=int, default=1200)
    parser.add_argument("--gamma1", type=float, default=0.5)
    parser.add_argument("--gamma2", type=float, default=0.99)
    parser.add_argument("--denoiser", choices=["paper_dncnn", "score_pnp"], required=True)
    parser.add_argument("--paper-repo", default="refs/pds_pnp_ref")
    parser.add_argument("--denoiser-ckpt", default=None)
    parser.add_argument("--score-pnp-repo", default="refs/score_pnp_ref")
    parser.add_argument("--score-pnp-ckpt", default=None)
    parser.add_argument("--score-denoiser-type", default="auto", choices=["auto", "dncnn", "vp_score", "score", "diffusion"])
    parser.add_argument("--score-model-image-size", type=int, default=256)
    parser.add_argument("--lambda-snore", type=float, default=1e-6)
    parser.add_argument("--sigma-begin", type=float, default=0.01)
    parser.add_argument("--sigma-end", type=float, default=0.0025)
    parser.add_argument("--sigma-schedule", choices=["constant", "linear", "geometric"], default="geometric")
    parser.add_argument("--snore-mc-samples", type=int, default=1)
    parser.add_argument("--snore-residual-mode", choices=["clean", "noisy"], default="clean")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--save-every-image", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; using CPU")
        args.device = "cpu"
    device = args.device
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    paths = resolve_images(args.dataset_root, args.image_list, args.num_images)
    print(f"[data] using {len(paths)} image(s):")
    for p in paths:
        print(f"  - {p}")

    kernel = load_kernel(args, device=device)
    op = PaperBlurTorch(kernel)

    ckpt = None
    if args.denoiser == "paper_dncnn":
        ckpt = Path(args.denoiser_ckpt) if args.denoiser_ckpt else autodetect_ckpt(args.paper_repo, ["DnCNN_nobn_nch_3_nlev_0.01", "dncnn", "0.01"])
        if ckpt is None:
            raise FileNotFoundError("Could not autodetect paper DnCNN checkpoint. Pass --denoiser-ckpt refs/pds_pnp_ref/nn/DnCNN_nobn_nch_3_nlev_0.01.pth")
        print(f"[denoiser] loading paper DnCNN: {ckpt}")
        denoiser = load_paper_dncnn(ckpt, channels=3, device=device, paper_repo=args.paper_repo)
    else:
        ckpt = Path(args.score_pnp_ckpt) if args.score_pnp_ckpt else autodetect_ckpt(args.score_pnp_repo, ["imagenet256", "dncnn_sigma2_color", "drunet_color", "score", "ffhq_10m"])
        if ckpt is None:
            raise FileNotFoundError("Could not autodetect score_pnp checkpoint. Pass --score-pnp-ckpt PATH")
        print(f"[denoiser] loading score_pnp denoiser: {ckpt} type={args.score_denoiser_type}")
        denoiser = load_score_pnp_denoiser(
            ckpt=ckpt,
            score_repo=args.score_pnp_repo,
            channels=3,
            device=device,
            denoiser_type=args.score_denoiser_type,
            score_model_image_size=args.score_model_image_size,
        )

    rows = []
    psnrs = []
    ssims = []
    for idx, path in enumerate(paths, start=1):
        print(f"\n=== image {idx}/{len(paths)}: {path.name} ===")
        x_true = load_image_center_crop(path, args.crop_size, device=device)
        y = make_observation(x_true, op, args.noise_std)
        eps = float(args.eps_alpha * args.noise_std * math.sqrt(x_true[0].numel()))
        x0 = y.clamp(0.0, 1.0)
        start = time.time()
        if args.method == "pnp_pds_paper":
            x_hat, logs = pnp_pds_paper_gaussian(
                y=y,
                op=op,
                denoiser=denoiser,
                n_iter=args.iters,
                gamma1=args.gamma1,
                gamma2=args.gamma2,
                eps=eps,
                x0=x0,
                gt=x_true,
                verbose=True,
                log_every=args.log_every,
            )
        else:
            cfg = SNOREPDSConfig(
                n_iter=args.iters,
                gamma1=args.gamma1,
                gamma2=args.gamma2,
                lambda_snore=args.lambda_snore,
                eps=eps,
                noise_std=args.noise_std,
                eps_factor=args.eps_alpha,
                box=True,
                rho=1.0,
                sigma_begin=args.sigma_begin,
                sigma_end=args.sigma_end,
                sigma_schedule=args.sigma_schedule,
                snore=SNOREConfig(mc_samples=args.snore_mc_samples, residual_mode=args.snore_residual_mode, clip_denoised=True),
                seed=args.seed + idx - 1,
                verbose=True,
                log_every=args.log_every,
            )
            x_hat, logs = snore_pds_gaussian(y=y, op=op, denoiser=denoiser, config=cfg, x0=x0, ground_truth=x_true)
        elapsed = time.time() - start
        p = psnr(x_hat, x_true)
        s = ssim_torch(x_hat, x_true)
        pin = psnr(y.clamp(0, 1), x_true)
        psnrs.append(p)
        ssims.append(s)
        row = {
            "index": idx,
            "filename": path.name,
            "method": args.method,
            "denoiser": args.denoiser,
            "noise_std": args.noise_std,
            "eps_alpha": args.eps_alpha,
            "eps": eps,
            "iters": args.iters,
            "gamma1": args.gamma1,
            "gamma2": args.gamma2,
            "lambda_snore": args.lambda_snore if args.method == "snore_pds" else "",
            "psnr_input": pin,
            "psnr": p,
            "ssim_fast": s,
            "seconds": elapsed,
        }
        rows.append(row)
        print(f"[result] {path.name}: input_psnr={pin:.3f} recon_psnr={p:.3f} ssim_fast={s:.4f} time={elapsed:.1f}s")
        save_image(x_hat, outdir / f"recon_{idx:02d}_{path.stem}.png")
        save_image(y.clamp(0, 1), outdir / f"obs_{idx:02d}_{path.stem}.png")
        if args.save_every_image:
            save_image(x_true, outdir / f"gt_{idx:02d}_{path.stem}.png")
        with open(outdir / f"logs_{idx:02d}_{path.stem}.json", "w") as f:
            json.dump(logs, f, indent=2)

    avg = {
        "average_psnr": float(np.mean(psnrs)),
        "average_ssim_fast": float(np.mean(ssims)),
        "num_images": len(rows),
        "args": vars(args),
    }
    with open(outdir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(outdir / "summary.json", "w") as f:
        json.dump({"summary": avg, "rows": rows}, f, indent=2)
    print("\n# ------------")
    print(f"# Average PSNR: {avg['average_psnr']:.4f} dB")
    print(f"# Average SSIM-fast: {avg['average_ssim_fast']:.4f}")
    print(f"# Saved to: {outdir}")
    print("# ------------")


if __name__ == "__main__":
    main()
