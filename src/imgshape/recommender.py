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


# -------------------------
# Low-level helpers
# -------------------------
def _open_image_from_input(inp: Any) -> Optional[Image.Image]:
    """Accept PIL.Image, path-like, bytes/bytearray, BytesIO, or file-like and return PIL.Image (RGB)."""
    if inp is None:
        return None

    try:
        if isinstance(inp, Image.Image):
            return inp.convert("RGB")
    except Exception:
        logger.debug("open_image: not PIL.Image", exc_info=True)

    if isinstance(inp, (str, Path)):
        try:
            return Image.open(str(inp)).convert("RGB")
        except Exception:
            logger.debug("open_image: path open failed for %s", inp, exc_info=True)
            return None

    if isinstance(inp, (bytes, bytearray)):
        try:
            return Image.open(BytesIO(inp)).convert("RGB")
        except Exception:
            logger.debug("open_image: bytes open failed", exc_info=True)
            return None

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
    `size` is (width, height).
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


# -------------------------
# Utility helpers (for callers)
# -------------------------
def safe_get(obj: Any, key: str, default=None):
    """
    Safe accessor: if obj is mapping-like use .get, otherwise try getattr, otherwise default.
    """
    try:
        if isinstance(obj, Mapping):
            return obj.get(key, default)
        return getattr(obj, key, default)
    except Exception:
        return default


def ensure_stats(obj: Any) -> Dict[str, Any]:
    """
    Convert common inputs to a stats-like dict.
    - If obj is a mapping, return it (shallow-copied).
    - If obj is a PIL.Image or path-like, compute minimal stats and return a mapping.
    - Otherwise return empty dict.
    """
    try:
        if isinstance(obj, Mapping):
            return dict(obj)
    except Exception:
        pass

    pil = _open_image_from_input(obj)
    if pil is None:
        return {}

    shp = _shape_from_image(pil) or (None, None, None)
    h, w, c = shp
    entropy = _entropy_from_image(pil)
    channels = int(c) if c else 3

    stats = {
        "image_count": 1,
        "entropy_mean": entropy,
        "channels": channels,
        "avg_width": int(w) if w else None,
        "avg_height": int(h) if h else None,
        "shape_distribution": {"unique_shapes": {f"{w}x{h}": 1}} if w and h else {},
    }
    return stats


# -------------------------
# Augmentation plan generator (basic heuristics)
# -------------------------
def _generate_augmentation_plan_from_stats(stats: Mapping) -> Dict[str, Any]:
    """
    Return an augmentation_plan dict shaped like:
    {
      "order": ["RandomHorizontalFlip", ...],
      "augmentations": [
         {"name": "RandomHorizontalFlip", "params": {"p": 0.5}, "reason": "...", "score": 0.7},
         ...
      ]
    }
    Basic heuristics:
      - If image count small/one-shot -> be conservative
      - If entropy high -> color/brightness augmentations OK
      - If min_side >= 96 -> spatial augmentations allowed
      - If channels == 3 -> color augmentations possible
    """
    plan: Dict[str, Any] = {"order": [], "augmentations": []}
    # safe reads
    entropy = safe_get(stats, "entropy_mean", None)
    channels = safe_get(stats, "channels", 3)
    avg_w = safe_get(stats, "avg_width", None)
    avg_h = safe_get(stats, "avg_height", None)
    image_count = safe_get(stats, "image_count", 0) or 0

    # interpret shape
    try:
        if avg_w and avg_h:
            min_side = min(int(avg_w), int(avg_h))
        else:
            min_side = 224
    except Exception:
        min_side = 224

    # heuristics & scoring
    def add_aug(name: str, params: Dict[str, Any], reason: str, base_score: float):
        plan["order"].append(name)
        plan["augmentations"].append({"name": name, "params": params, "reason": reason, "score": round(float(base_score), 2)})

    # always safe: small geometric flips if natural images
    if min_side >= 32:
        add_aug(
            "RandomHorizontalFlip",
            {"p": 0.5},
            "Common orientation variance; usually safe for many datasets",
            0.7,
        )

    # small rotations if image count moderate and not tiny images
    if min_side >= 96 and image_count > 5:
        add_aug(
            "RandomRotation",
            {"degrees": 15},
            "Small rotations to increase orientation robustness",
            0.6,
        )

    # color jitter if RGB and entropy indicates variety
    try:
        if int(channels) == 3 and (entropy is None or (entropy is not None and entropy >= 3.0)):
            add_aug(
                "ColorJitter",
                {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.05},
                "Color variations; useful for natural images with decent entropy",
                0.5,
            )
    except Exception:
        pass

    # brightness/contrast for low-entropy images (e.g., mostly dark)
    if entropy is not None and entropy < 2.0:
        add_aug(
            "RandomAdjustSharpness",
            {"sharpness_factor": 1.2},
            "Low-entropy images might benefit from contrast/sharpness augmentation",
            0.45,
        )

    # crop/rescale if big images
    if min_side >= 224:
        add_aug(
            "RandomResizedCrop",
            {"size": 224, "scale": [0.8, 1.0]},
            "Large images: random resized crop helps scale invariance",
            0.6,
        )

    # if no augmentation added (very small dataset), provide a conservative default
    if not plan["augmentations"]:
        add_aug("RandomHorizontalFlip", {"p": 0.5}, "Default conservative augmentation", 0.4)

    # ensure order is stable and unique (preserve insertion order but dedupe names)
    seen = set()
    ordered = []
    uniq_aug = []
    for a in plan["order"]:
        if a not in seen:
            seen.add(a)
            ordered.append(a)
    deduped_augs = []
    seen = set()
    for a in plan["augmentations"]:
        if a["name"] not in seen:
            seen.add(a["name"])
            deduped_augs.append(a)

    plan["order"] = ordered
    plan["augmentations"] = deduped_augs
    return plan


# -------------------------
# Public API: recommend_preprocessing & recommend_dataset
# -------------------------
def recommend_preprocessing(input_obj: Any) -> Dict[str, Any]:
    """
    Suggest preprocessing steps and augmentation plan.

    Returns keys:
      - augmentation_plan: dict (order + augmentations)
      - resize: {"size": [w,h], "method": "..."}
      - normalize: {"mean": [...], "std": [...]}
      - mode: "RGB" or "Grayscale"
      - entropy: float|None
      - suggested_model: str
      - (or) error: message
    """
    # Case A: mapping-like stats
    if isinstance(input_obj, Mapping):
        stats = input_obj  # type: Mapping
        entropy = safe_get(stats, "entropy_mean", None)
        channels = safe_get(stats, "channels", 3)

        # shape detection
        rep_h = rep_w = None
        sd = safe_get(stats, "shape_distribution", {}) or {}
        if isinstance(sd, Mapping):
            uniq = sd.get("unique_shapes") or {}
            if isinstance(uniq, Mapping) and uniq:
                for k in uniq.keys():
                    try:
                        ks = str(k)
                        if "x" in ks:
                            a_str, b_str = ks.split("x", 1)
                            a, b = int(a_str), int(b_str)
                            if a >= b:
                                rep_w, rep_h = a, b
                            else:
                                rep_w, rep_h = b, a
                            break
                    except Exception:
                        continue

        if rep_h is None or rep_w is None:
            rep_h = safe_get(stats, "height") or safe_get(stats, "avg_height") or None
            rep_w = safe_get(stats, "width") or safe_get(stats, "avg_width") or None

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

        augmentation_plan = _generate_augmentation_plan_from_stats(stats)

        width, height = size_tuple
        return {
            "augmentation_plan": augmentation_plan,
            "resize": {"size": [width, height], "method": method},
            "normalize": {"mean": mean, "std": std},
            "mode": "RGB" if channels == 3 else "Grayscale",
            "entropy": entropy,
            "suggested_model": suggested,
        }

    # Case B: image-like input
    pil = _open_image_from_input(input_obj)
    if pil is None:
        tname = type(input_obj).__name__ if input_obj is not None else "None"
        return {"error": f"Unsupported input for recommend_preprocessing (type={tname}). Provide stats mapping, path, PIL.Image, bytes, or file-like."}

    shape = _shape_from_image(pil)
    entropy = _entropy_from_image(pil)
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

    # build light stats for augmentation plan generation
    stats_for_plan = {
        "image_count": 1,
        "entropy_mean": entropy,
        "channels": channels,
        "avg_width": int(w) if w else None,
        "avg_height": int(h) if h else None,
        "shape_distribution": {"unique_shapes": {f"{w}x{h}": 1}} if w and h else {},
    }
    augmentation_plan = _generate_augmentation_plan_from_stats(stats_for_plan)

    width, height = size_tuple
    return {
        "augmentation_plan": augmentation_plan,
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
    Lightweight dataset-level recommender. If passed a mapping, forwarded to recommend_preprocessing.
    If passed a path, scans sample files and aggregates minimal stats, then forwards.
    """
    if isinstance(dataset_input, Mapping):
        return recommend_preprocessing(dataset_input)

    if isinstance(dataset_input, (str, Path)):
        path = Path(dataset_input)
        if not path.exists():
            return {"error": f"path not found: {dataset_input}"}

        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.gif", "*.webp")
        images = _find_images_in_path(path, exts)
        if not images:
            return {"error": "no images found in dataset path"}

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
