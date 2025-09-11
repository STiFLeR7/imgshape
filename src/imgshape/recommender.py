# src/imgshape/recommender.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Iterable
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import math
import glob


def _open_image_from_input(inp: Any) -> Optional[Image.Image]:
    """Accept PIL.Image, path-like, bytes/bytearray, BytesIO, or file-like and return PIL.Image (RGB)."""
    if inp is None:
        return None

    # Already a PIL image
    try:
        if isinstance(inp, Image.Image):
            return inp.convert("RGB")
    except Exception:
        pass

    # Path-like (string or Path)
    if isinstance(inp, (str, Path)):
        try:
            return Image.open(str(inp)).convert("RGB")
        except Exception:
            return None

    # Bytes or bytearray
    if isinstance(inp, (bytes, bytearray)):
        try:
            return Image.open(BytesIO(inp)).convert("RGB")
        except Exception:
            return None

    # File-like (has read)
    if hasattr(inp, "read"):
        try:
            # Some file-likes (Streamlit) are consumed — ensure we can seek
            try:
                inp.seek(0)
            except Exception:
                pass
            data = inp.read()
            if not data:
                return None
            return Image.open(BytesIO(data)).convert("RGB")
        except Exception:
            return None

    return None


def _shape_from_image(pil: Image.Image) -> Optional[Tuple[int, int, int]]:
    """Return (height, width, channels) from PIL image."""
    if pil is None:
        return None
    try:
        w, h = pil.size
        channels = len(pil.getbands())
        return (h, w, channels)
    except Exception:
        return None


def _entropy_from_image(pil: Image.Image) -> Optional[float]:
    """Compute simple Shannon entropy on grayscale histogram (base-2)."""
    if pil is None:
        return None
    try:
        gray = pil.convert("L")
        hist = gray.histogram()
        total = sum(hist)
        if total == 0:
            return 0.0
        entropy = 0.0
        for c in hist:
            if c == 0:
                continue
            p = c / total
            entropy -= p * math.log2(p)
        return round(float(entropy), 3)
    except Exception:
        return None


def _defaults_for_channels(channels: int) -> Tuple[list, list]:
    if channels == 1:
        return [0.5], [0.5]
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def _choose_resize_by_min_side(min_side: int) -> Tuple[Tuple[int, int], str, str]:
    """Return (size, method, suggested_model)"""
    if min_side >= 224:
        return (224, 224), "bilinear", "MobileNet/ResNet"
    if min_side >= 96:
        return (96, 96), "bilinear", "EfficientNet-B0 (small)"
    if min_side <= 32:
        return (32, 32), "nearest", "TinyNet/MNIST/CIFAR"
    return (128, 128), "bilinear", "General Use"


def recommend_preprocessing(input_obj: Any) -> Dict[str, Any]:
    """
    Suggest preprocessing steps.

    Accepts:
      - stats dict (dataset stats) -> uses stats.get(...)
      - PIL.Image.Image, path (str/Path), bytes/BytesIO, file-like -> compute local stats

    Returns:
      {
        "resize": {"size": [w,h], "method": "bilinear"},
        "normalize": {"mean": [...], "std": [...]},
        "mode": "RGB" or "Grayscale",
        "entropy": float or None,
        "suggested_model": str
      }
    """
    # If a dict-like stats object is provided, use its values where possible
    if isinstance(input_obj, dict):
        stats = input_obj
        entropy = stats.get("entropy_mean") or stats.get("entropy") or None
        channels = stats.get("channels") or stats.get("channels_mode") or None

        # try to extract a representative shape
        rep_h = rep_w = None
        sd = stats.get("shape_distribution") or {}
        if isinstance(sd, dict):
            uniq = sd.get("unique_shapes") or {}
            if uniq:
                # look for first "WxH" or "HxW" string key
                for k in uniq.keys():
                    if "x" in k:
                        parts = k.split("x")
                        if len(parts) == 2:
                            try:
                                # try both orders
                                a, b = int(parts[0]), int(parts[1])
                                # prefer interpreting as width x height if width>height
                                if a >= b:
                                    rep_w, rep_h = a, b
                                else:
                                    rep_w, rep_h = b, a
                                break
                            except Exception:
                                continue

        if rep_h is None or rep_w is None:
            rep_h = stats.get("height") or stats.get("avg_height")
            rep_w = stats.get("width") or stats.get("avg_width")

        if rep_h and rep_w:
            min_side = min(int(rep_h), int(rep_w))
        else:
            min_side = 224

        size, method, suggested = _choose_resize_by_min_side(min_side)
        channels = int(channels) if channels else 3
        mean, std = _defaults_for_channels(channels)

        return {
            "resize": {"size": [size[0], size[1]], "method": method},
            "normalize": {"mean": mean, "std": std},
            "mode": "RGB" if channels == 3 else "Grayscale",
            "entropy": entropy,
            "suggested_model": suggested,
        }

    # Otherwise treat input as an image-like and compute minimal stats
    pil = _open_image_from_input(input_obj)
    if pil is None:
        return {"error": "Unsupported input for recommend_preprocessing. Provide stats dict, path, PIL.Image, bytes, or file-like."}

    shape = _shape_from_image(pil)
    entropy = _entropy_from_image(pil)
    channels = shape[2] if shape and len(shape) >= 3 else len(pil.getbands()) if pil else 3
    h = shape[0] if shape else None
    w = shape[1] if shape else None

    min_side = min(w, h) if (w and h) else 224
    size, method, suggested = _choose_resize_by_min_side(min_side)
    mean, std = _defaults_for_channels(channels)

    return {
        "resize": {"size": [size[0], size[1]], "method": method},
        "normalize": {"mean": mean, "std": std},
        "mode": "RGB" if channels == 3 else "Grayscale",
        "entropy": entropy,
        "suggested_model": suggested,
    }


def recommend_dataset(dataset_input: Any) -> Dict[str, Any]:
    """
    Lightweight dataset-level recommender.
    Accepts either:
      - dataset stats dict (from analyze_dataset)
      - path to dataset folder (str/Path)
    Returns dataset-level recommendation summary.
    """
    # if dict-like, forward
    if isinstance(dataset_input, dict):
        return recommend_preprocessing(dataset_input)

    # If path-like, attempt to derive a simple dataset stat summary
    if isinstance(dataset_input, (str, Path)):
        path = Path(dataset_input)
        if not path.exists():
            return {"error": f"path not found: {dataset_input}"}

        # collect basic stats (small/simple pass)
        images = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.gif"):
            images.extend(glob.glob(str(path / "**" / ext), recursive=True))

        if not images:
            return {"error": "no images found in dataset path"}

        # compute a small sample (first N) to avoid heavy work
        sample = images[:50]
        entropies = []
        channels_seen = []
        widths = []
        heights = []
        for p in sample:
            try:
                img = _open_image_from_input(p)
                if img is None:
                    continue
                h, w, c = _shape_from_image(img)
                entropies.append(_entropy_from_image(img) or 0.0)
                channels_seen.append(c)
                widths.append(w)
                heights.append(h)
            except Exception:
                continue

        stats = {
            "image_count": len(images),
            "entropy_mean": round(sum(entropies) / len(entropies), 3) if entropies else None,
            "channels": max(set(channels_seen), key=channels_seen.count) if channels_seen else None,
            "avg_width": int(sum(widths) / len(widths)) if widths else None,
            "avg_height": int(sum(heights) / len(heights)) if heights else None,
        }
        return recommend_preprocessing(stats)

    return {"error": "Unsupported input for recommend_dataset"}
