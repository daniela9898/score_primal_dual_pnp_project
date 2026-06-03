#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def check(path: Path, description: str) -> bool:
    ok = path.exists()
    print(f"[{'OK' if ok else 'MISSING'}] {description}: {path}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()
    ws = Path(args.workspace).resolve()
    pds = ws / "refs" / "pds_pnp_ref"
    score = ws / "refs" / "score_pnp_ref"

    required = [
        (pds, "PDS reference repo"),
        (score, "score_pnp reference repo"),
        (pds / "datasets" / "imagenet", "PDS 7 ImageNet test images"),
        (pds / "blur_models" / "blur_1.mat", "PDS blur kernel blur_1.mat"),
        (pds / "nn", "PDS denoiser folder"),
        (score / "guided_diffusion" / "unet.py", "score_pnp guided_diffusion/unet.py"),
        (score / "pretrained_models" / "score" / "imagenet256.pt", "VP score checkpoint imagenet256.pt"),
    ]
    oks = [check(p, d) for p, d in required]
    if not all(oks):
        raise SystemExit("Some assets are missing. Download/check the missing paths above.")
    print("All required assets are present.")


if __name__ == "__main__":
    main()
