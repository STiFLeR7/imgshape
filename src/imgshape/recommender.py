# src/imgshape/recommender.py
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Iterable, List
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import math
import glob
import logging

logger = logging.getLogger(__name__)

# -----------------------------------------------------------
# Helpers: open image, shape, entropy, defaults and sizing
# -----------------------------------------------------------
def _open_image_from_input(inp: Any) -> Optional[Image.Image]:
    """Accept PIL.Image, path-like, bytes/bytearray, BytesIO, or file-like and return PIL.Image (RGB)."""
    if inp is None:
        return None

    # Already a PIL image
    try:
        if isinstance(inp, Image.Image):
            return inp.convert("RGB")
    except Exception:
        # defensive: continue to other handlers
        logger.debug("open_image: not a PIL.Image (%s)", type(inp), exc_info=True)

    # Path-like (string or Path)
    if isinstance(inp, (str, Path)):
        try:
            return Image.open(str(inp)).convert("RGB")
        except Exception:
            logger.debug("open_image: path open failed for %s", inp, exc_info=True)
            return None

    # Bytes or bytearray
    if isinstance(inp, (bytes, bytearray)):
        try:
            return Image.open(BytesIO(inp)).convert("RGB")
        except Exception:
            logger.debug("open_image: bytes open failed", exc_info=True)
            return None

    # File-like (has read)
    if hasattr(inp, "read"):
        try:
            try:
                inp.seek(0)
            except Exception:
                pass
            data = inp.read()
            if not data:
                return None
            return Image.open(BytesIO(data)).convert("RGB")
        except Exception:
            logger.debug("open_image: file-like open failed", exc_info=True)
            return None

    return None


def _shape_from_image(pil: Image.Image) -> Optional[Tuple[int, int, int]]:
    """Return (height, width, channels) from PIL image or None."""
    if pil is None:
        return None
    try:
        w, h = pil.size
        channels = len(pil.getbands())
        return (h, w, channels)
    except Exception:
        logger.debug("shape_from_image failed", exc_info=True)
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
        logger.debug("entropy_from_image failed", exc_info=True)
        return None


def _defaults_for_channels(channels: int) -> Tuple[List[float], List[float]]:
    """Return (mean, std) for given channel count."""
    if channels == 1:
        return [0.5], [0.5]
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def _choose_resize_by_min_side(min_side: int) -> Tuple[Tuple[int, int], str, str]:
    """
    Return (size, method, suggested_model).

    `size` is returned as (width, height) for consistency with common image APIs.
    """
    try:
        ms = int(min_side)
    except Exception:
        ms = 224

    if ms >= 224:
        return (224, 224), "bilinear", "MobileNet/ResNet"
    if ms >= 96:
        return (96, 96), "bilinear", "EfficientNet-B0 (small)"
    if ms <= 32:
        return (32, 32), "nearest", "TinyNet/MNIST/CIFAR"
    return (128, 128), "bilinear", "General Use"


# -----------------------------------------------------------
# Public API: recommend_preprocessing and recommend_dataset
# -----------------------------------------------------------
def recommend_preprocessing(input_obj: Any) -> Dict[str, Any]:
    """
    Suggest preprocessing steps.

    Accepts:
      - Mapping-like stats (dataset stats) -> uses mapping.get(...)
      - PIL.Image.Image, path (str/Path), bytes/BytesIO, file-like -> compute local stats

    Returns a dict:
      {
        "resize": {"size": [width, height], "method": "bilinear"},
        "normalize": {"mean": [...], "std": [...]},
        "mode": "RGB" or "Grayscale",
        "entropy": float or None,
        "suggested_model": str
      }

    On unsupported input returns: {"error": "...message..."}
    """
    # 1) mapping-like stats input (Dataset-level stats)
    if isinstance(input_obj, Mapping):
        stats = input_obj  # type: Mapping

        # safe accessors with fallbacks
        entropy = stats.get("entropy_mean") or stats.get("entropy") or None
        channels = stats.get("channels") or stats.get("channels_mode") or None

        # Representative shape detection: tries shape_distribution -> avg -> fallback
        rep_h = rep_w = None
        sd = stats.get("shape_distribution") or {}
        if isinstance(sd, Mapping):
            uniq = sd.get("unique_shapes") or {}
            if isinstance(uniq, Mapping) and uniq:
                for k in uniq.keys():
                    try:
                        ks = str(k)
                        if "x" in ks:
                            a_str, b_str = ks.split("x", 1)
                            a, b = int(a_str), int(b_str)
                            # decide which is width/height based on typical orientation
                            if a >= b:
                                rep_w, rep_h = a, b
                            else:
                                rep_w, rep_h = b, a
                            break
                    except Exception:
                        continue

        if rep_h is None or rep_w is None:
            rep_h = stats.get("height") or stats.get("avg_height") or None
            rep_w = stats.get("width") or stats.get("avg_width") or None

        if rep_h and rep_w:
            try:
                min_side = min(int(rep_h), int(rep_w))
            except Exception:
                min_side = 224
        else:
            min_side = 224

        size_tuple, method, suggested = _choose_resize_by_min_side(min_side)

        try:
            channels = int(channels) if channels is not None else 3
        except Exception:
            channels = 3

        mean, std = _defaults_for_channels(channels)

        # Return size as [width, height]
        width, height = size_tuple

        return {
            "augmentation_plan": None,
            "resize": {"size": [width, height], "method": method},
            "normalize": {"mean": mean, "std": std},
            "mode": "RGB" if channels == 3 else "Grayscale",
            "entropy": entropy,
            "suggested_model": suggested,
        }

    # 2) image-like input -> compute minimal stats from PIL image
    pil = _open_image_from_input(input_obj)
    if pil is None:
        tname = type(input_obj).__name__ if input_obj is not None else "None"
        return {"error": f"Unsupported input for recommend_preprocessing (type={tname}). Provide stats mapping, path, PIL.Image, bytes, or file-like."}

    shape = _shape_from_image(pil)
    entropy = _entropy_from_image(pil)

    # shape returns (h, w, c)
    if shape:
        h, w, c = shape
    else:
        h = w = c = None

    min_side = min(w, h) if (w and h) else 224
    size_tuple, method, suggested = _choose_resize_by_min_side(min_side)

    channels = c if c is not None else len(pil.getbands()) if pil else 3
    try:
        channels = int(channels)
    except Exception:
        channels = 3

    mean, std = _defaults_for_channels(channels)
    width, height = size_tuple

    return {
        "augmentation_plan": None,
        "resize": {"size": [width, height], "method": method},
        "normalize": {"mean": mean, "std": std},
        "mode": "RGB" if channels == 3 else "Grayscale",
        "entropy": entropy,
        "suggested_model": suggested,
    }


def _find_images_in_path(path: Path, exts: Iterable[str]) -> List[str]:
    results: List[str] = []
    for ext in exts:
        results.extend(glob.glob(str(path / "**" / ext), recursive=True))
    return results


def recommend_dataset(dataset_input: Any, sample_limit: int = 50) -> Dict[str, Any]:
    """
    Lightweight dataset-level recommender.

    Accepts:
      - Mapping-like stats (forwarded to recommend_preprocessing)
      - path to dataset folder (str/Path) -> scans recursively for common image extensions

    Returns dataset-level recommendation summary (calls recommend_preprocessing on aggregated stats)
    """
    # if mapping-like, forward
    if isinstance(dataset_input, Mapping):
        return recommend_preprocessing(dataset_input)

    # If path-like, attempt to derive a simple dataset stat summary
    if isinstance(dataset_input, (str, Path)):
        path = Path(dataset_input)
        if not path.exists():
            return {"error": f"path not found: {dataset_input}"}

        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.gif", "*.webp")
        images = _find_images_in_path(path, exts)

        if not images:
            return {"error": "no images found in dataset path"}

        # sample first N for a light-weight summary
        sample_files = images[:sample_limit]

        entropies: List[float] = []
        channels_seen: List[int] = []
        widths: List[int] = []
        heights: List[int] = []

        for p in sample_files:
            try:
                img = _open_image_from_input(p)
                if img is None:
                    continue
                shp = _shape_from_image(img)
                if not shp:
                    continue
                hh, ww, cc = shp
                ent = _entropy_from_image(img) or 0.0
                entropies.append(ent)
                channels_seen.append(cc)
                widths.append(ww)
                heights.append(hh)
            except Exception:
                logger.debug("Error processing sample image %s", p, exc_info=True)
                continue

        stats: Dict[str, Any] = {
            "image_count": len(images),
            "entropy_mean": round(sum(entropies) / len(entropies), 3) if entropies else None,
            "channels": max(set(channels_seen), key=channels_seen.count) if channels_seen else None,
            "avg_width": int(sum(widths) / len(widths)) if widths else None,
            "avg_height": int(sum(heights) / len(heights)) if heights else None,
            "shape_distribution": {"unique_shapes": {f"{w}x{h}": 1 for w, h in zip(widths, heights)}},
        }

        return recommend_preprocessing(stats)

    return {"error": "Unsupported input for recommend_dataset"}
