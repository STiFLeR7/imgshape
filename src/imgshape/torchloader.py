"""
viz.py — visualization helpers for imgshape v2.2.0
"""

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image


def plot_shape_distribution(dataset_path: str, out_path: str = "shape_distribution.png") -> str:
    """
    Plot histogram of (width,height) shapes for a dataset directory.
    """
    p = Path(dataset_path)
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    shapes = []
    for f in p.iterdir():
        if f.suffix.lower() in exts:
            try:
                with Image.open(f) as img:
                    shapes.append(f"{img.width}x{img.height}")
            except Exception:
                continue
    counter = Counter(shapes)
    if not counter:
        return ""

    plt.figure(figsize=(8, 4))
    plt.bar(counter.keys(), counter.values())
    plt.xticks(rotation=90)
    plt.title("Image Shape Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path
