# src/imgshape/shape.py
"""
shape.py — shape extraction utilities for imgshape v2.2.0

Provides robust functions to extract image shapes (H, W, C) for single
images or batches. Handles both paths and PIL.Image instances.
"""

from __future__ import annotations
from typing import Tuple, List, Union, Dict
from pathlib import Path
from PIL import Image


SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def get_shape(path_or_img: Union[str, Path, Image.Image]) -> Tuple[int, int, int]:
    """
    Return (H, W, C) for a single image.

    Parameters
    ----------
    path_or_img : str | Path | PIL.Image
        File path to image or an already loaded PIL.Image.

    Returns
    -------
    tuple[int, int, int]
        (height, width, channels)

    Raises
    ------
    FileNotFoundError
        If path does not exist.
    OSError
        If image cannot be opened.
    """
    if isinstance(path_or_img, Image.Image):
        w, h = path_or_img.size
        c = len(path_or_img.getbands())
        return (h, w, c)

    p = Path(path_or_img)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path_or_img}")

    with Image.open(p) as img:
        w, h = img.size
        c = len(img.getbands())
        return (h, w, c)


def get_shape_batch(
    dir_path: Union[str, Path], recursive: bool = False, include_errors: bool = False
) -> List[Union[Tuple[int, int, int], Dict[str, str]]]:
    """
    Return list of shapes for all images in a directory.

    Parameters
    ----------
    dir_path : str | Path
        Directory containing images.
    recursive : bool, default False
        If True, also scan subdirectories.
    include_errors : bool, default False
        If True, append dicts with error info instead of skipping unreadable files.

    Returns
    -------
    list
        List of (H, W, C) tuples, or dicts if include_errors=True.
    """
    p = Path(dir_path)
    if not p.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    results: List[Union[Tuple[int, int, int], Dict[str, str]]] = []

    files = p.rglob("*") if recursive else p.iterdir()
    for file in files:
        if not file.is_file() or file.suffix.lower() not in SUPPORTED_EXTS:
            continue
        try:
            shape = get_shape(file)
            results.append(shape)
        except Exception as e:
            if include_errors:
                results.append({"file": str(file), "error": str(e)})
            # else skip
            continue
    return results


if __name__ == "__main__":
    # Example usage
    folder = "./sample_images"
    try:
        shapes = get_shape_batch(folder, recursive=True, include_errors=True)
        for s in shapes[:5]:
            print(s)
    except Exception as e:
        print("Error:", e)
