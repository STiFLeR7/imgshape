# src/imgshape/viz.py
"""
Visualization utilities for imgshape v2.1.0

Provides histograms, scatter plots, and simple distribution summaries
for dataset image sizes and shapes.

All functions can display interactively OR save to disk.
"""

import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Dict, Tuple, Optional

from imgshape.shape import get_shape_batch


# -------------------------------
# Helpers
# -------------------------------

def _get_image_paths(folder_path: str) -> List[str]:
    """Return a list of image file paths from a folder (common extensions only)."""
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.gif"]
    files: List[str] = []
    for ext in image_extensions:
        files.extend(glob.glob(os.path.join(folder_path, ext)))
    return files


def _extract_dims(shapes_dict: Dict[str, Tuple[int, int, int]]) -> Tuple[List[int], List[int], List[int]]:
    """Extract width, height, channels from a dict of {path: (h,w,c)}."""
    widths, heights, channels = [], [], []
    for shape in shapes_dict.values():
        if len(shape) == 3:
            h, w, c = shape
            widths.append(w)
            heights.append(h)
            channels.append(c)
    return widths, heights, channels


# -------------------------------
# Plot functions
# -------------------------------

def plot_shape_distribution(folder_path: str, save: bool = False, out_dir: str = "output") -> Optional[str]:
    """
    Plot histogram distribution of image widths and heights.

    Args:
        folder_path: directory of images
        save: whether to save PNG to disk
        out_dir: folder to save in

    Returns:
        str path to saved file if save=True, else None
    """
    image_paths = _get_image_paths(folder_path)
    shapes_dict = get_shape_batch(image_paths)

    widths, heights, _ = _extract_dims(shapes_dict)

    plt.figure(figsize=(10, 5))
    plt.hist(widths, bins=15, alpha=0.6, label="Widths", color="#5DADE2")
    plt.hist(heights, bins=15, alpha=0.6, label="Heights", color="#F5B041")
    plt.xlabel("Pixels")
    plt.ylabel("Frequency")
    plt.title("🧮 Image Size Distribution")
    plt.legend()
    plt.tight_layout()

    if save:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "shape_distribution.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        return out_path
    else:
        plt.show()
        return None


def plot_image_dimensions(folder_path: str, save: bool = False, out_dir: str = "output") -> Optional[str]:
    """
    Scatter plot of width vs. height for all images.

    Returns path if saved, else None.
    """
    image_paths = _get_image_paths(folder_path)
    shapes_dict = get_shape_batch(image_paths)

    widths, heights, _ = _extract_dims(shapes_dict)

    plt.figure(figsize=(6, 6))
    plt.scatter(widths, heights, alpha=0.6, c="#48C9B0", edgecolors="black")
    plt.xlabel("Width (px)")
    plt.ylabel("Height (px)")
    plt.title("🖼️ Image Dimension Scatter Plot")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "dimension_scatter.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        return out_path
    else:
        plt.show()
        return None


def plot_channel_distribution(folder_path: str, save: bool = False, out_dir: str = "output") -> Optional[str]:
    """
    Bar chart of channel counts (RGB vs Grayscale).

    Returns path if saved, else None.
    """
    image_paths = _get_image_paths(folder_path)
    shapes_dict = get_shape_batch(image_paths)
    _, _, channels = _extract_dims(shapes_dict)

    counts = Counter(channels)
    labels, values = list(counts.keys()), list(counts.values())

    plt.figure(figsize=(5, 4))
    plt.bar(labels, values, color="#9B59B6", alpha=0.7)
    plt.xticks(labels, [f"{c} channels" for c in labels])
    plt.ylabel("Count")
    plt.title("🌈 Channel Distribution")
    plt.tight_layout()

    if save:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "channel_distribution.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        return out_path
    else:
        plt.show()
        return None


def plot_dataset_distribution(folder_path: str, save: bool = False, out_dir: str = "output") -> Dict[str, str]:
    """
    Convenience function: generate all distribution plots (size hist, scatter, channel).

    Returns dict of paths if saved, else empty dict.
    """
    results = {}
    results["size_hist"] = plot_shape_distribution(folder_path, save=save, out_dir=out_dir)
    results["scatter"] = plot_image_dimensions(folder_path, save=save, out_dir=out_dir)
    results["channels"] = plot_channel_distribution(folder_path, save=save, out_dir=out_dir)
    return {k: v for k, v in results.items() if v}
