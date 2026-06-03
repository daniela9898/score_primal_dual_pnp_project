#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def patch_file(path: Path, replacements: list[tuple[str, str]]) -> bool:
    text = path.read_text()
    original = text
    for old, new in replacements:
        text = text.replace(old, new)
    if text != original:
        path.write_text(text)
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pds_repo", required=True)
    args = parser.parse_args()

    pds_repo = Path(args.pds_repo).resolve()
    denoiser_py = pds_repo / "models" / "denoiser.py"
    test_py = pds_repo / "pnppds" / "test_pnppds.py"

    if not denoiser_py.exists():
        raise FileNotFoundError(denoiser_py)
    if not test_py.exists():
        raise FileNotFoundError(test_py)

    # PyTorch >=2.6 loads with weights_only=True by default; old checkpoints need weights_only=False.
    changed_denoiser = patch_file(
        denoiser_py,
        [
            (
                "torch.load(file_name, map_location=lambda storage, loc: storage)",
                "torch.load(file_name, map_location=lambda storage, loc: storage, weights_only=False)",
            )
        ],
    )

    text = test_py.read_text()
    original = text
    if "import os" not in text.split("\n", 1)[0] and "import os" not in text[:300]:
        text = text.replace("import cv2, datetime, glob, json", "import cv2, datetime, glob, json, os")
    # Filename extraction: Windows-only -> portable.
    text = text.replace("filename = (path_img[path_img.rfind('\\\\'):])[1:]", "filename = os.path.basename(path_img)")
    # Save paths: Windows-only -> portable. Handles the common original strings.
    text = text.replace("path_result + '\\\\GROUND_TRUTH_' + path_saveimg_base", "os.path.join(path_result, 'GROUND_TRUTH_' + path_saveimg_base)")
    text = text.replace("path_result + '\\\\OBSERVATION_' + path_saveimg_base", "os.path.join(path_result, 'OBSERVATION_' + path_saveimg_base)")
    text = text.replace("path_result + '\\\\RESULT_' + path_saveimg_base", "os.path.join(path_result, 'RESULT_' + path_saveimg_base)")
    text = text.replace("np.save(path_result + '\\\\DATA_' + path_saveimg_base, datas)", "np.save(os.path.join(path_result, 'DATA_' + path_saveimg_base), datas)")
    text = text.replace("path_result + '\\GROUND_TRUTH_' + path_saveimg_base", "os.path.join(path_result, 'GROUND_TRUTH_' + path_saveimg_base)")
    text = text.replace("path_result + '\\OBSERVATION_' + path_saveimg_base", "os.path.join(path_result, 'OBSERVATION_' + path_saveimg_base)")
    text = text.replace("path_result + '\\RESULT_' + path_saveimg_base", "os.path.join(path_result, 'RESULT_' + path_saveimg_base)")
    text = text.replace("np.save(path_result + '\\DATA_' + path_saveimg_base, datas)", "np.save(os.path.join(path_result, 'DATA_' + path_saveimg_base), datas)")
    changed_test = text != original
    if changed_test:
        test_py.write_text(text)

    print(f"Patched {denoiser_py}: {changed_denoiser}")
    print(f"Patched {test_py}: {changed_test}")


if __name__ == "__main__":
    main()
