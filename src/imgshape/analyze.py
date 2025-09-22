# src/imgshape/analyze.py
"""
Robust analyze module for imgshape v2.2.0

Provides:
- analyze_type(input_obj): analyze a single image-like input and return stats/detection.
- analyze_dataset(dataset_path_or_iterable, sample_limit=200): scan a dataset and return aggregated stats.

Input types supported:
- PIL.Image.Image
- str / pathlib.Path (path to file)
- bytes / bytearray
- file-like object (has .read())
- URL strings (http/https) only if allow_network=True
Returns a dict. On error returns {"error": "..."}.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Iterable, List
from pathlib import Path
from io import BytesIO

from PIL import Image, UnidentifiedImageError, ImageStat

import math
from collections import Counter

logger = logging.getLogger("imgshape.analyze")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


# ---- helpers: open and normalize inputs ----
def _open_image_from_input(inp: Any, allow_network: bool = False) -> Optional[Image.Image]:
    """
    Robustly open a PIL.Image from a variety of inputs:
      - PIL.Image.Image -> return .convert("RGB")
      - bytes / bytearray -> BytesIO -> PIL
      - str / Path -> file path -> PIL if exists (NETWORK ops disabled by default)
      - file-like (has read) -> read bytes -> PIL
    Returns None if it cannot be opened.
    """
    if inp is None:
        return None

    # If already a PIL image, just convert and return
    try:
        if isinstance(inp, Image.Image):
            return inp.convert("RGB")
    except Exception:
        logger.debug("Input is not PIL.Image or conversion failed", exc_info=True)

    # Bytes-like
    try:
        if isinstance(inp, (bytes, bytearray)):
            return Image.open(BytesIO(inp)).convert("RGB")
    except Exception:
        logger.debug("Failed to open bytes input as image", exc_info=True)

    # Path-like (string or Path)
    try:
        if isinstance(inp, (str, Path)):
            s = str(inp)
            p = Path(s)
            if p.exists() and p.is_file():
                return Image.open(p).convert("RGB")
            # If it's a URL and network allowed, fetch it (disabled by default)
            if (s.startswith("http://") or s.startswith("https://")) and allow_network:
                try:
                    import requests

                    r = requests.get(s, timeout=5)
                    r.raise_for_status()
                    return Image.open(BytesIO(r.content)).convert("RGB")
                except Exception:
                    logger.debug("Failed to fetch URL image", exc_info=True)
    except Exception:
        logger.debug("Failed to handle string/path input", exc_info=True)

    # File-like objects (e.g., Gradio file object may present differently)
    try:
        if hasattr(inp, "read"):
            try:
                pos = None
                try:
                    pos = inp.tell()
                except Exception:
                    pos = None
                data = inp.read()
                if data:
                    if isinstance(data, str):
                        data = data.encode("utf-8")
                    return Image.open(BytesIO(data)).convert("RGB")
                if pos is not None:
                    try:
                        inp.seek(pos)
                    except Exception:
                        pass
            except Exception:
                logger.debug("Error reading file-like object in _open_image_from_input", exc_info=True)
    except Exception:
        logger.debug("Input does not support read()", exc_info=True)

    return None


def _safe_mode_and_channels(pil: Image.Image) -> Dict[str, Any]:
    """
    Return small dict describing mode, channels, size and basic stats.
    """
    try:
        w, h = pil.size
        bands = pil.getbands() or ()
        channels = len(bands)
        mode = pil.mode
        try:
            stat = ImageStat.Stat(pil)
            means = stat.mean if hasattr(stat, "mean") else []
            stddev = stat.stddev if hasattr(stat, "stddev") else []
        except Exception:
            means = []
            stddev = []
        return {
            "width": int(w),
            "height": int(h),
            "channels": int(channels),
            "mode": mode,
            "means": [float(x) for x in means] if means else [],
            "stddev": [float(x) for x in stddev] if stddev else [],
        }
    except Exception:
        logger.debug("_safe_mode_and_channels failed", exc_info=True)
        return {}


def _entropy_from_image(pil: Image.Image) -> Optional[float]:
    """Shannon entropy computed on grayscale histogram; robust to exceptions."""
    if pil is None:
        return None
    try:
        gray = pil.convert("L")
        hist = gray.histogram()
        total = sum(hist)
        if total == 0:
            return 0.0
        ent = 0.0
        for c in hist:
            if c == 0:
                continue
            p = c / total
            ent -= p * math.log2(p)
        return round(float(ent), 3)
    except Exception:
        logger.debug("entropy computation failed", exc_info=True)
        return None


# ---- Public single-image analyzer ----
def analyze_type(input_obj: Any, allow_network: bool = False) -> Dict[str, Any]:
    """
    Analyze a single image-like input and return a dict with:
      - meta (width, height, channels, mode, means, stddev)
      - entropy
      - suggestions (size/model)
    Always returns a dict — on failure returns {'error': '...'}.
    """
    try:
        pil = _open_image_from_input(input_obj, allow_network=allow_network)
        if pil is None:
            return {"error": "Unsupported input for analyze_type. Provide path, PIL.Image, bytes, or file-like."}

        meta = _safe_mode_and_channels(pil)
        entropy = _entropy_from_image(pil)
        meta["entropy"] = entropy

        suggestions = {}
        if meta.get("channels", 3) == 1:
            suggestions["mode"] = "grayscale"
        else:
            suggestions["mode"] = "rgb"

        w = meta.get("width")
        h = meta.get("height")
        if w and h:
            min_side = min(int(w), int(h))
            if min_side >= 224:
                suggestions["suggested_size"] = [224, 224]
                suggestions["suggested_model"] = "ResNet/MobileNet"
            elif min_side >= 96:
                suggestions["suggested_size"] = [96, 96]
                suggestions["suggested_model"] = "EfficientNet-B0 / MobileNetV2"
            else:
                suggestions["suggested_size"] = [32, 32]
                suggestions["suggested_model"] = "TinyNet/CIFAR-style"
        else:
            suggestions["suggested_size"] = [128, 128]
            suggestions["suggested_model"] = "General-purpose"

        result = {"meta": meta, "suggestions": suggestions}
        return result
    except Exception as exc:
        logger.exception("Unexpected error in analyze_type: %s", exc)
        return {"error": "Internal analyzer failure", "detail": str(exc)}


# ---- Public dataset analyzer ----
def analyze_dataset(dataset_input: Any, sample_limit: int = 200) -> Dict[str, Any]:
    """
    Analyze a dataset (directory path or iterable of image-like objects).
    Returns aggregated statistics.
    """
    try:
        items: List[Any] = []
        if isinstance(dataset_input, (str, Path)):
            p = Path(dataset_input)
            if not p.exists() or not p.is_dir():
                return {"error": "Dataset path invalid or not a directory."}
            exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")
            paths = []
            for e in exts:
                paths.extend(sorted(p.glob(e)))
            paths = sorted(dict.fromkeys(paths))
            items = paths[: sample_limit]
        elif isinstance(dataset_input, Iterable):
            items = list(dataset_input)[:sample_limit]
        else:
            return {"error": "Unsupported dataset input. Provide path or iterable of images."}

        if not items:
            return {"error": "No images found in dataset."}

        image_count = 0
        shape_counter = Counter()
        entropy_values = []
        channels_counter = Counter()
        sample_summaries = []

        for it in items:
            pil = _open_image_from_input(it)
            if pil is None:
                continue
            image_count += 1
            w, h = pil.size
            shape_counter[f"{w}x{h}"] += 1
            channels_counter[len(pil.getbands())] += 1
            ent = _entropy_from_image(pil)
            if ent is not None:
                entropy_values.append(ent)
            if len(sample_summaries) < 5:
                summ = analyze_type(pil)
                sample_summaries.append(summ)

        if image_count == 0:
            return {"error": "No readable images in dataset."}

        avg_entropy = round(float(sum(entropy_values) / len(entropy_values)), 3) if entropy_values else None

        out = {
            "image_count": image_count,
            "unique_shapes": dict(shape_counter),
            "channels_distribution": dict(channels_counter),
            "avg_entropy": avg_entropy,
            "sample_summaries": sample_summaries,
        }
        return out
    except Exception as exc:
        logger.exception("Unexpected error in analyze_dataset: %s", exc)
        return {"error": "Internal dataset analyzer failure", "detail": str(exc)}
