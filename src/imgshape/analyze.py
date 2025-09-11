# src/imgshape/analyze.py
from PIL import Image, ImageStat, ImageFilter
import numpy as np
from collections import Counter
from typing import Dict, Any, List
from statistics import mean, pstdev
import os, glob


def get_entropy(image_path: str) -> float:
    """Returns entropy of an image (Shannon entropy over grayscale)."""
    img = Image.open(image_path).convert("L")
    return img.entropy()


def get_edge_density(image_path: str) -> float:
    """Returns edge pixel ratio after edge detection."""
    img = Image.open(image_path).convert("L")
    edges = img.filter(ImageFilter.FIND_EDGES)
    arr = np.array(edges)
    edge_pixels = np.sum(arr > 50)  # threshold
    total_pixels = arr.size
    return edge_pixels / total_pixels


def get_dominant_color(image_path: str) -> str:
    """Returns the dominant color (as hex) in the image."""
    img = Image.open(image_path).convert("RGB").resize((50, 50))
    pixels = np.array(img).reshape(-1, 3)
    counts = Counter(map(tuple, pixels))
    dominant = counts.most_common(1)[0][0]
    return '#%02x%02x%02x' % dominant


def analyze_type(image_path: str) -> Dict[str, Any]:
    """Performs a lightweight image type analysis (single image)."""
    entropy = get_entropy(image_path)
    edge_density = get_edge_density(image_path)
    dominant_color = get_dominant_color(image_path)

    # Heuristic type guesser (can improve later)
    if entropy < 3.0 and edge_density < 0.01:
        guess = "document/scan"
    elif edge_density > 0.07:
        guess = "object-rich"
    elif entropy > 5.0:
        guess = "natural image"
    else:
        guess = "uncertain"

    return {
        "entropy": round(entropy, 2),
        "edge_density": round(edge_density, 3),
        "dominant_color": dominant_color,
        "guess_type": guess,
    }


# ----------------------
# NEW: dataset-level API
# ----------------------

def analyze_dataset(folder_path: str) -> Dict[str, Any]:
    """
    Aggregate dataset-level statistics for images under folder_path.
    Returns a dict with at least: image_count, source_dir, entropy stats, channels, etc.
    """
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.gif")
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder_path, "**", e), recursive=True))

    stats: Dict[str, Any] = {
        "image_count": len(files),
        "source_dir": folder_path,
        "shape_distribution": {},
        "class_balance": {},
    }

    if not files:
        return stats

    entropies: List[float] = []
    edge_densities: List[float] = []
    colorfulness: List[str] = []
    channels_seen: List[int] = []

    for f in files:
        try:
            info = analyze_type(f)
            if "entropy" in info:
                entropies.append(info["entropy"])
            if "edge_density" in info:
                edge_densities.append(info["edge_density"])
            if "dominant_color" in info:
                colorfulness.append(info["dominant_color"])
            # crude channel inference
            try:
                img = Image.open(f)
                channels_seen.append(len(img.getbands()))
            except Exception:
                pass
        except Exception:
            continue

    if entropies:
        stats["entropy_mean"] = round(mean(entropies), 3)
        stats["entropy_std"] = round(pstdev(entropies), 3) if len(entropies) > 1 else 0.0
    if edge_densities:
        stats["edge_density_mean"] = round(mean(edge_densities), 3)
    if colorfulness:
        # dominant color mode (most frequent)
        counts = Counter(colorfulness)
        stats["dominant_color_mode"] = max(counts, key=counts.get)
    if channels_seen:
        stats["channels"] = max(set(channels_seen), key=channels_seen.count)

    # simple shape distribution
    try:
        from imgshape.shape import get_shape
        wh = []
        for fp in files:
            try:
                s = get_shape(fp)
                if s and len(s) >= 2:
                    h, w = s[0], s[1]
                    wh.append((w, h))
            except Exception:
                continue
        if wh:
            uniq = Counter(wh)
            stats["shape_distribution"] = {
                "unique_shapes": {f"{w}x{h}": c for (w, h), c in uniq.items()},
                "most_common": f"{max(uniq, key=uniq.get)[0]}x{max(uniq, key=uniq.get)[1]}",
            }
    except Exception:
        pass

    return stats
