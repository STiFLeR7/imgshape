# src/imgshape/viz.py
"""
viz.py — imgshape v3 interactive visualization utilities.

Supports:
- Interactive histograms and scatter plots (Plotly)
- Fallback to Matplotlib for headless environments
- Non-fatal imports: no hard dependency on Plotly

Usage:
    from imgshape.viz import plot_shape_distribution
    fig = plot_shape_distribution("dataset/")
    fig.show()  # works interactively in Jupyter/Streamlit
"""

from __future__ import annotations
import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
from PIL import Image

# optional imports
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

logger = logging.getLogger("imgshape.viz")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


def _get_image_shapes(dataset_path: str, max_samples: int = 1000) -> List[Tuple[int, int]]:
    """
    Collect image shapes from a dataset folder.
    Returns list of (width, height) tuples.
    """
    shapes = []
    p = Path(dataset_path)
    if not p.exists():
        logger.warning("Dataset path does not exist: %s", dataset_path)
        return []

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    for img_path in p.rglob("*"):
        if img_path.suffix.lower() in exts:
            try:
                with Image.open(img_path) as im:
                    shapes.append(im.size)  # (width, height)
            except Exception as e:
                logger.debug("Failed to open %s: %s", img_path, e)
        if len(shapes) >= max_samples:
            break
    return shapes


def plot_shape_distribution(
    dataset_path: str,
    save: bool = False,
    output_path: Optional[str] = None,
    interactive: Optional[bool] = None,
):
    """
    Plot the width/height distribution of images in the dataset.
    Returns a figure object (Plotly or Matplotlib) for embedding.

    Parameters
    ----------
    dataset_path : str
        Path to dataset root
    save : bool, optional
        Save static figure to file if True
    output_path : str, optional
        If save=True, specify where to write PNG/HTML
    interactive : bool, optional
        Force Plotly (True) or Matplotlib (False).
        If None, auto-detects based on Plotly availability.
    """
    shapes = _get_image_shapes(dataset_path)
    if not shapes:
        logger.warning("No images found for visualization in %s", dataset_path)
        return None

    widths, heights = zip(*shapes)
    aspect_ratios = np.array(widths) / np.array(heights)
    total = len(shapes)

    use_plotly = interactive if interactive is not None else HAS_PLOTLY

    if use_plotly:
        # --- Plotly Interactive ---
        logger.info("Using Plotly for interactive visualization.")
        df = {
            "width": widths,
            "height": heights,
            "aspect_ratio": aspect_ratios,
        }

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=widths,
                y=heights,
                mode="markers",
                marker=dict(
                    size=6,
                    color=aspect_ratios,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Aspect Ratio"),
                ),
                name="Images",
            )
        )
        fig.update_layout(
            title=f"📊 Image Shape Distribution — {Path(dataset_path).name} ({total} samples)",
            xaxis_title="Width (px)",
            yaxis_title="Height (px)",
            template="plotly_white",
            hovermode="closest",
        )

        # Save as HTML if requested
        if save:
            output_path = output_path or str(Path(dataset_path).joinpath("shape_distribution.html"))
            try:
                fig.write_html(output_path)
                logger.info("Saved interactive HTML plot to %s", output_path)
            except Exception as e:
                logger.error("Failed to save HTML plot: %s", e)
        return fig

    elif HAS_MPL:
        # --- Matplotlib fallback ---
        logger.info("Using Matplotlib fallback (non-interactive).")
        plt.figure(figsize=(7, 6))
        plt.scatter(widths, heights, alpha=0.6, c=aspect_ratios, cmap="viridis")
        plt.xlabel("Width (px)")
        plt.ylabel("Height (px)")
        plt.title(f"Image Shape Distribution — {Path(dataset_path).name} ({total} samples)")
        plt.colorbar(label="Aspect Ratio")

        if save:
            output_path = output_path or str(Path(dataset_path).joinpath("shape_distribution.png"))
            plt.savefig(output_path, dpi=120, bbox_inches="tight")
            logger.info("Saved static PNG plot to %s", output_path)

        plt.close()
        return None
    else:
        logger.warning("Neither Plotly nor Matplotlib available. Install one for visualization.")
        return None


def plot_entropy_distribution(entropy_values: List[float], title: str = "Entropy Distribution", interactive: Optional[bool] = None):
    """
    Plot entropy value distribution (useful for dataset analysis).
    Returns Plotly Figure if available, else None.
    """
    if not entropy_values:
        logger.warning("Empty entropy values; skipping plot.")
        return None

    use_plotly = interactive if interactive is not None else HAS_PLOTLY
    if use_plotly:
        import plotly.express as px

        fig = px.histogram(
            x=entropy_values,
            nbins=30,
            title=title,
            labels={"x": "Entropy", "y": "Frequency"},
            color_discrete_sequence=["#636EFA"],
        )
        fig.update_layout(template="plotly_white")
        return fig
    elif HAS_MPL:
        plt.figure(figsize=(6, 4))
        plt.hist(entropy_values, bins=30, color="#636EFA", alpha=0.8)
        plt.title(title)
        plt.xlabel("Entropy")
        plt.ylabel("Frequency")
        plt.close()
        return None
    else:
        logger.warning("Plotly/Matplotlib not installed; cannot plot entropy distribution.")
        return None
