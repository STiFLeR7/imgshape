# src/imgshape/analyze.py

import os, glob
from typing import Dict, Any, List
from statistics import mean, pstdev
from imgshape.shape import get_shape
from imgshape.analyze import analyze_type  # reuse your per-image analyzer

def analyze_dataset(folder_path: str) -> Dict[str, Any]:
    """Aggregate dataset-level stats across all images in a folder."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.gif")
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder_path, "**", e), recursive=True))

    stats: Dict[str, Any] = {
        "image_count": len(files),
        "source_dir": folder_path,
        "shape_distribution": {},
        "class_balance": {},  # hook for future label balance checks
    }

    if not files:
        return stats

    entropies, colorfulness_vals, channels_list = [], [], []

    for f in files:
        try:
            info = analyze_type(f)
            if not isinstance(info, dict):
                continue
            if "entropy" in info:
                entropies.append(info["entropy"])
            if "colorfulness" in info:
                colorfulness_vals.append(info["colorfulness"])
            if "channels" in info:
                channels_list.append(info["channels"])
        except Exception:
            continue

    if entropies:
        stats["entropy_mean"] = round(mean(entropies), 3)
        stats["entropy_std"] = round(pstdev(entropies), 3) if len(entropies) > 1 else 0
    if colorfulness_vals:
        stats["colorfulness_mean"] = round(mean(colorfulness_vals), 3)
    if channels_list:
        stats["channels"] = max(set(channels_list), key=channels_list.count)

    # simple shape distribution summary
    try:
        shapes = [get_shape(f) for f in files]
        wh_pairs = [(s[1], s[0]) for s in shapes if s and len(s) >= 2]
        stats["shape_distribution"] = {
            "unique_shapes": {str(s): wh_pairs.count(s) for s in set(wh_pairs)},
            "most_common": max(set(wh_pairs), key=wh_pairs.count) if wh_pairs else None,
        }
    except Exception:
        pass

    return stats
