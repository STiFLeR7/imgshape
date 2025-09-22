"""
shape.py — shape extraction utilities for imgshape v2.2.0
"""

from __future__ import annotations
from typing import Tuple, List
from pathlib import Path
from PIL import Image


def get_shape(path_or_img) -> Tuple[int, int, int]:
    """
    Return (H, W, C) for a single image.
    Accepts path string or PIL.Image.
    """
    if isinstance(path_or_img, Image.Image):
        w, h = path_or_img.size
        c = len(path_or_img.getbands())
        return (h, w, c)
    path = Path(path_or_img)
    with Image.open(path) as img:
        w, h = img.size
        c = len(img.getbands())
        return (h, w, c)


def get_shape_batch(dir_path: str) -> List[Tuple[int, int, int]]:
    """
    Return list of shapes for all images in a directory.
    """
    p = Path(dir_path)
    shapes = []
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    for file in p.iterdir():
        if file.suffix.lower() in exts:
            try:
                shapes.append(get_shape(file))
            except Exception:
                continue
    return shapes
