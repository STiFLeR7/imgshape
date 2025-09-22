# src/imgshape/resize.py
"""
Robust image resizing utilities for imgshape v2.2.0

Features:
- Single-image and batch resizing
- Supports int (square) or "WxH" size strings
- Aspect-ratio preserving (with padding) or direct resize
- Keeps optional folder structure when saving batch
- Structured return values for easier logging and testing
"""

import os
from pathlib import Path
from typing import Union, Tuple, List, Dict, Optional

from PIL import Image, ImageOps


def _parse_size(size: Union[int, str, Tuple[int, int]]) -> Tuple[int, int]:
    """Normalize size input into (width, height)."""
    if isinstance(size, int):
        return (size, size)
    if isinstance(size, str) and "x" in size.lower():
        w, h = size.lower().split("x")
        return (int(w), int(h))
    if isinstance(size, (tuple, list)) and len(size) == 2:
        return (int(size[0]), int(size[1]))
    raise ValueError("Invalid size format. Use int, (W,H), or 'WIDTHxHEIGHT'.")


def resize_image(
    img_path: Union[str, Path],
    size: Union[int, str, Tuple[int, int]],
    fmt: str = "jpg",
    keep_aspect: bool = True,
    save_dir: Optional[Union[str, Path]] = None,
    keep_original: bool = False,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> Dict[str, Union[str, bool]]:
    """
    Resize a single image and save it.

    Returns
    -------
    dict with keys:
      - input_path
      - output_path
      - kept_original (bool)
      - size (W,H)
    """
    img_path = Path(img_path)
    if not img_path.exists():
        return {"error": "file_not_found", "input_path": str(img_path)}

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        return {"error": "unreadable_file", "input_path": str(img_path), "detail": str(e)}

    target_size = _parse_size(size)
    filename = img_path.stem

    # Resize
    if keep_aspect:
        img = ImageOps.pad(img, target_size, color=pad_color)
    else:
        img = img.resize(target_size, Image.BILINEAR)

    # Build output path
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{filename}.{fmt}"
    else:
        out_path = img_path.parent / f"{filename}_resized.{fmt}"

    try:
        img.save(out_path, fmt.upper())
    except Exception as e:
        return {"error": "save_failed", "input_path": str(img_path), "detail": str(e)}

    return {
        "input_path": str(img_path),
        "output_path": str(out_path),
        "kept_original": keep_original,
        "size": target_size,
    }


def batch_resize(
    folder_path: Union[str, Path],
    size: Union[int, str, Tuple[int, int]],
    fmt: str = "jpg",
    keep_structure: bool = True,
    save_dir: Optional[Union[str, Path]] = None,
    keep_original: bool = False,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> List[Dict[str, Union[str, bool]]]:
    """
    Resize all images in a folder (recursively).

    Returns
    -------
    list of dicts, one per image, each same as resize_image return.
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        return [{"error": "folder_not_found", "folder_path": str(folder_path)}]

    results: List[Dict[str, Union[str, bool]]] = []
    supported_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    for root, _, files in os.walk(folder_path):
        for file in files:
            if not file.lower().endswith(supported_exts):
                continue
            img_path = Path(root) / file

            # Build output directory if keeping structure
            target_dir = None
            if save_dir:
                if keep_structure:
                    rel_path = Path(root).relative_to(folder_path)
                    target_dir = Path(save_dir) / rel_path
                else:
                    target_dir = Path(save_dir)

            res = resize_image(
                img_path,
                size,
                fmt=fmt,
                keep_aspect=True,
                save_dir=target_dir,
                keep_original=keep_original,
                pad_color=pad_color,
            )
            results.append(res)

    return results


if __name__ == "__main__":
    # Example usage
    folder = "./sample_images"
    out = batch_resize(folder, "256x256", fmt="png", save_dir="./resized_out")
    print(f"Resized {len(out)} images")
