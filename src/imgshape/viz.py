# src/imgshape/viz.py
"""
Visualization utilities for imgshape v2.2.0

Provides histograms, scatter plots, and simple distribution summaries
for dataset image sizes and shapes.

All functions are defensive: they handle empty folders, unreadable files,
and can either display plots or save them to disk. Functions return the
output path (if saved) or None when shown interactively. A convenience
function returns a dict of generated artefacts.
"""

from __future__ import annotations
import os
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Dict, Tuple, Optional, Union
import logging

from imgshape.shape import get_shape_batch

logger = logging.getLogger("imgshape.viz")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


# -------------------------------
# Helpers
# -------------------------------

def _ensure_outdir(out_dir: Union[str, Path]) -> Path:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _shapes_from_folder(folder_path: Union[str, Path], recursive: bool = True, include_errors: bool = False):
    """
    Wrapper around get_shape_batch which accepts folder path and returns lists of widths, heights, channels.
    Returns (widths, heights, channels, errors)
    """
    p = Path(folder_path)
    if not p.exists():
        logger.warning("Folder does not exist: %s", folder_path)
        return [], [], [], [{"error": "folder_not_found", "path": str(folder_path)}]

    try:
        shapes = get_shape_batch(str(folder_path), recursive=recursive, include_errors=include_errors)
    except Exception as e:
        logger.warning("get_shape_batch failed: %s", e)
        return [], [], [], [{"error": "shape_batch_failed", "detail": str(e)}]

    widths: List[int] = []
    heights: List[int] = []
    channels: List[int] = []
    errors: List[Dict] = []

    for item in shapes:
        if isinstance(item, tuple) and len(item) == 3:
            h, w, c = item
            widths.append(w)
            heights.append(h)
            channels.append(c)
        elif isinstance(item, dict):
            errors.append(item)

    return widths, heights, channels, errors


# -------------------------------
# Plot functions
# -------------------------------

def plot_shape_distribution(
    folder_path: Union[str, Path],
    save: bool = False,
    out_dir: str = "output",
    figsize: Tuple[int, int] = (10, 4),
    bins: int = 20,
    recursive: bool = True,
) -> Optional[str]:
    """
    Plot histogram of width and height distributions for a dataset directory.

    Returns path if saved, else None (plots interactively).
    """
    widths, heights, _, errors = _shapes_from_folder(folder_path, recursive=recursive)
    if not widths and not heights:
        logger.info("No images found in %s", folder_path)
        # Create a tiny empty figure with message
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No images found to plot", ha="center", va="center", fontsize=12)
        ax.axis("off")
        if save:
            outp = _ensure_outdir(out_dir) / "shape_distribution.png"
            fig.savefig(outp, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return str(outp)
        else:
            plt.show()
            plt.close(fig)
            return None

    fig, ax = plt.subplots(figsize=figsize)
    # If only a single sample, show both as single-bar hist with annotations
    if len(widths) == 1 and len(heights) == 1:
        w, h = widths[0], heights[0]
        ax.bar([0, 1], [1, 1], tick_label=[f"W:{w}px", f"H:{h}px"], color=["#5DADE2", "#F5B041"], alpha=0.8)
        ax.set_ylabel("Count")
        ax.set_title("Single-sample Image Shape")
    else:
        ax.hist(widths, bins=bins, alpha=0.6, label="Widths (px)")
        ax.hist(heights, bins=bins, alpha=0.6, label="Heights (px)")
        ax.set_xlabel("Pixels")
        ax.set_ylabel("Frequency")
        ax.set_title("Image Width/Height Distribution")
        ax.legend()

    plt.tight_layout()
    if save:
        outp = _ensure_outdir(out_dir) / "shape_distribution.png"
        fig.savefig(outp, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved shape distribution to %s", outp)
        return str(outp)
    else:
        plt.show()
        plt.close(fig)
        return None


def plot_image_dimensions(
    folder_path: Union[str, Path],
    save: bool = False,
    out_dir: str = "output",
    figsize: Tuple[int, int] = (6, 6),
    recursive: bool = True,
) -> Optional[str]:
    """
    Scatter plot of image (width, height) pairs. Annotates single samples for clarity.
    """
    widths, heights, _, errors = _shapes_from_folder(folder_path, recursive=recursive)
    fig, ax = plt.subplots(figsize=figsize)

    if not widths:
        logger.info("No images found in %s", folder_path)
        ax.text(0.5, 0.5, "No images found to plot", ha="center", va="center", fontsize=12)
        ax.axis("off")
    elif len(widths) == 1:
        w, h = widths[0], heights[0]
        ax.scatter([w], [h], s=300, alpha=0.9, edgecolors="black", zorder=3)
        ax.annotate(f"{w}×{h}", (w, h), textcoords="offset points", xytext=(10, 10))
        margin = max(100, int(0.05 * max(w, h)))
        ax.set_xlim(w - margin, w + margin)
        ax.set_ylim(h - margin, h + margin)
        ax.set_xlabel("Width (px)")
        ax.set_ylabel("Height (px)")
    else:
        ax.scatter(widths, heights, alpha=0.6, c="#48C9B0", edgecolors="black")
        ax.set_xlabel("Width (px)")
        ax.set_ylabel("Height (px)")

    ax.set_title("Image Dimension Scatter Plot")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save:
        outp = _ensure_outdir(out_dir) / "dimension_scatter.png"
        fig.savefig(outp, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved dimension scatter to %s", outp)
        return str(outp)
    else:
        plt.show()
        plt.close(fig)
        return None


def plot_channel_distribution(
    folder_path: Union[str, Path],
    save: bool = False,
    out_dir: str = "output",
    figsize: Tuple[int, int] = (5, 4),
    recursive: bool = True,
) -> Optional[str]:
    """
    Bar chart showing counts of channel types (1, 3, 4, ...).
    """
    _, _, channels, errors = _shapes_from_folder(folder_path, recursive=recursive)
    counts = Counter(channels)
    if not counts:
        logger.info("No channel information found in %s", folder_path)
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No images found", ha="center", va="center")
        ax.axis("off")
        if save:
            outp = _ensure_outdir(out_dir) / "channel_distribution.png"
            fig.savefig(outp, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return str(outp)
        else:
            plt.show()
            plt.close(fig)
            return None

    labels = [f"{k} ch" for k in counts.keys()]
    values = list(counts.values())

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(labels, values, color="#9B59B6", alpha=0.8)
    ax.set_ylabel("Count")
    ax.set_title("Channel Distribution")
    plt.tight_layout()

    if save:
        outp = _ensure_outdir(out_dir) / "channel_distribution.png"
        fig.savefig(outp, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved channel distribution to %s", outp)
        return str(outp)
    else:
        plt.show()
        plt.close(fig)
        return None


def plot_dataset_distribution(
    folder_path: Union[str, Path],
    save: bool = False,
    out_dir: str = "output",
    recursive: bool = True,
) -> Dict[str, Optional[str]]:
    """
    Convenience function: generate all distribution plots (size hist, scatter, channel).

    Returns a dict of saved file paths (if save=True) or None values (if shown interactively).
    """
    results: Dict[str, Optional[str]] = {}
    results["size_hist"] = plot_shape_distribution(folder_path, save=save, out_dir=out_dir, recursive=recursive)
    results["scatter"] = plot_image_dimensions(folder_path, save=save, out_dir=out_dir, recursive=recursive)
    results["channels"] = plot_channel_distribution(folder_path, save=save, out_dir=out_dir, recursive=recursive)
    # Remove None entries if saving to disk
    if save:
        return {k: v for k, v in results.items() if v}
    return results
