# src/imgshape/analyze.py
"""
analyze.py — robust image/dataset analysis for imgshape v2.2.0

Provides:
- analyze_type(input_obj): analyze a single image-like input and return stats/suggestions.
- analyze_dataset(dataset_path_or_iterable, sample_limit=200): scan a dataset and return aggregated stats.

Input types supported:
- PIL.Image.Image
- str / pathlib.Path (path to file or directory)
- bytes / bytearray
- file-like object (has .read())
- URL strings (http/https) if allow_network=True

Returns a dict. On error, always returns {"error": "..."}.
"""

from __future__ import annotations
import logging
import math
from typing import Any, Dict, Optional, Iterable, List
from pathlib import Path
from io import BytesIO
from collections import Counter

from PIL import Image, ImageStat, UnidentifiedImageError

logger = logging.getLogger("imgshape.analyze")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


# ----------------------
# Input handling helpers
# ----------------------
def _open_image_from_input(inp: Any, allow_network: bool = False) -> Optional[Image.Image]:
    """Robustly open a PIL.Image from a variety of inputs.

    Resolves strings through multiple strategies so relative paths in test/CI are handled.
    """
    if inp is None:
        return None

    # Already a PIL image
    try:
        if isinstance(inp, Image.Image):
            return inp.convert("RGB")
    except Exception:
        logger.debug("Not a PIL.Image", exc_info=True)

    # Raw bytes / bytearray
    try:
        if isinstance(inp, (bytes, bytearray)):
            return Image.open(BytesIO(inp)).convert("RGB")
    except Exception:
        logger.debug("Failed opening bytes", exc_info=True)

    # Path-like or string: attempt a few fallbacks
    try:
        if isinstance(inp, (str, Path)):
            s = str(inp)
            candidates = []

            # 1) direct path
            candidates.append(Path(s))

            # 2) cwd relative
            try:
                candidates.append(Path.cwd() / s)
            except Exception:
                pass

            # 3) repo relative: two levels up from this file (src/imgshape -> repo root)
            try:
                repo_root = Path(__file__).resolve().parents[2]
                candidates.append(repo_root / s)
            except Exception:
                pass

            # 4) common assets folder sibling to package
            try:
                pkg_assets = Path(__file__).resolve().parents[1] / "assets" / s
                candidates.append(pkg_assets)
            except Exception:
                pass

            # 5) try progressively moving up parent directories for relative paths
            try:
                p = Path(s)
                if not p.is_absolute():
                    curr = Path.cwd()
                    for _ in range(4):
                        candidates.append(curr / s)
                        curr = curr.parent
            except Exception:
                pass

            # Deduplicate and try
            seen = set()
            for cand in candidates:
                try:
                    candp = Path(cand)
                    if str(candp) in seen:
                        continue
                    seen.add(str(candp))
                    if candp.exists() and candp.is_file():
                        try:
                            return Image.open(candp).convert("RGB")
                        except UnidentifiedImageError:
                            logger.debug("File exists but is not an image: %s", candp)
                            return None
                        except Exception:
                            logger.debug("Failed opening image at candidate: %s", candp, exc_info=True)
                            continue
                except Exception:
                    logger.debug("Candidate evaluation failed for: %s", cand, exc_info=True)
            # Network fetch (optional)
            if (s.startswith("http://") or s.startswith("https://")) and allow_network:
                try:
                    import requests

                    r = requests.get(s, timeout=8)
                    r.raise_for_status()
                    return Image.open(BytesIO(r.content)).convert("RGB")
                except Exception:
                    logger.debug("Failed fetching URL: %s", s, exc_info=True)
    except Exception:
        logger.debug("Path/URL handling failed", exc_info=True)

    # File-like objects (has .read)
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
                logger.debug("File-like read failed", exc_info=True)
    except Exception:
        logger.debug("Object not readable", exc_info=True)

    return None


def _safe_meta(pil: Image.Image) -> Dict[str, Any]:
    """Extract width, height, channels, mode, means, stddev from an image safely."""
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
            means, stddev = [], []
        return {
            "width": int(w),
            "height": int(h),
            "channels": int(channels),
            "mode": mode,
            "means": [float(x) for x in means] if means else [],
            "stddev": [float(x) for x in stddev] if stddev else [],
        }
    except Exception:
        logger.debug("_safe_meta failed", exc_info=True)
        return {}


def _entropy_from_image(pil: Image.Image) -> Optional[float]:
    """Compute Shannon entropy on grayscale histogram."""
    if pil is None:
        return None
    try:
        gray = pil.convert("L")
        hist = gray.histogram()
        total = sum(hist)
        if total == 0:
            return 0.0
        ent = -sum((c / total) * math.log2(c / total) for c in hist if c > 0)
        return round(float(ent), 3)
    except Exception:
        logger.debug("entropy computation failed", exc_info=True)
        return None


def _guess_image_type(meta: Dict[str, Any], entropy: Optional[float]) -> str:
    """
    Simple heuristic to set a guess_type label:
      - 'photograph': typically high entropy, 3 channels, varied means
      - 'natural' : mid entropy photographic-like
      - 'diagram'  : low entropy, possibly few colors
      - 'icon'     : very low entropy and small size
      - 'unknown'  : fallback
    The tests only require the presence of a key, so keep this conservative.
    """
    try:
        if entropy is None:
            return "unknown"
        ch = int(meta.get("channels", 3))
        w = int(meta.get("width") or 0)
        h = int(meta.get("height") or 0)
        min_side = min(w, h) if w and h else 0

        if entropy >= 6.5 and ch == 3 and min_side >= 128:
            return "photograph"
        if 4.0 <= entropy < 6.5:
            return "natural"
        if entropy < 3.0:
            if min_side <= 64:
                return "icon"
            return "diagram"
        return "unknown"
    except Exception:
        return "unknown"


# ----------------------
# Public analyzers
# ----------------------
def analyze_type(input_obj: Any, allow_network: bool = False) -> Dict[str, Any]:
    """
    Analyze a single image-like input and return dict with:
      - meta (width, height, channels, mode, means, stddev)
      - entropy
      - suggestions (size/model)
      - guess_type (heuristic string)
    Always returns a dict (error dict on failure).
    """
    try:
        pil = _open_image_from_input(input_obj, allow_network=allow_network)
        if pil is None:
            logger.debug("analyze_type: unsupported input for input_obj=%r", input_obj)
            return {"error": "Unsupported input for analyze_type"}

        meta = _safe_meta(pil)
        ent = _entropy_from_image(pil)
        meta["entropy"] = ent

        suggestions: Dict[str, Any] = {}
        suggestions["mode"] = "grayscale" if meta.get("channels", 3) == 1 else "rgb"

        w, h = meta.get("width"), meta.get("height")
        if w and h:
            min_side = min(w, h)
            if min_side >= 224:
                suggestions.update({"suggested_size": [224, 224], "suggested_model": "ResNet/MobileNet"})
            elif min_side >= 96:
                suggestions.update({"suggested_size": [96, 96], "suggested_model": "EfficientNet-B0 / MobileNetV2"})
            else:
                suggestions.update({"suggested_size": [32, 32], "suggested_model": "TinyNet/CIFAR-style"})
        else:
            suggestions.update({"suggested_size": [128, 128], "suggested_model": "General-purpose"})

        guess = _guess_image_type(meta, ent)

        return {"meta": meta, "entropy": ent, "suggestions": suggestions, "guess_type": guess}
    except Exception as exc:
        logger.exception("Unexpected error in analyze_type: %s", exc)
        return {"error": "Internal analyzer failure", "detail": str(exc)}


def analyze_dataset(dataset_input: Any, sample_limit: int = 200) -> Dict[str, Any]:
    """
    Analyze a dataset (directory path or iterable of image-like objects).
    Returns aggregated stats including counts, entropy, shapes, channels.
    """
    try:
        items: List[Any] = []
        if isinstance(dataset_input, (str, Path)):
            p = Path(dataset_input).expanduser()
            # If provided path isn't absolute, try some sensible fallbacks
            if not p.exists():
                candidates = [p, Path.cwd() / p]
                try:
                    repo_candidate = Path(__file__).resolve().parents[2] / p
                    candidates.append(repo_candidate)
                except Exception:
                    pass
                found_dir = None
                for cand in candidates:
                    if cand.exists() and cand.is_dir():
                        found_dir = cand
                        break
                if found_dir is None:
                    return {"error": f"Dataset path invalid: {dataset_input}"}
                p = found_dir
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in exts:
                    items.append(f)
            items = items[:sample_limit]
        elif isinstance(dataset_input, Iterable):
            items = list(dataset_input)[:sample_limit]
        else:
            return {"error": "Unsupported dataset input type"}

        if not items:
            return {"error": "No images found in dataset"}

        image_count = 0
        shape_counter, channels_counter = Counter(), Counter()
        entropy_vals: List[float] = []
        sample_summaries: List[Dict[str, Any]] = []
        unreadable = 0

        for it in items:
            pil = _open_image_from_input(it)
            if pil is None:
                unreadable += 1
                continue
            image_count += 1
            w, h = pil.size
            shape_counter[f"{w}x{h}"] += 1
            channels_counter[len(pil.getbands())] += 1
            ent = _entropy_from_image(pil)
            if ent is not None:
                entropy_vals.append(ent)
            if len(sample_summaries) < 5:
                # store small sample analysis (use analyze_type on PIL to keep format)
                sample_summaries.append(analyze_type(pil))

        if image_count == 0:
            return {"error": "No readable images in dataset"}

        avg_entropy = round(sum(entropy_vals) / len(entropy_vals), 3) if entropy_vals else None
        most_common_shape, most_common_count = shape_counter.most_common(1)[0] if shape_counter else (None, 0)

        return {
            "image_count": image_count,
            "unique_shapes": dict(shape_counter),
            "most_common_shape": most_common_shape,
            "most_common_shape_count": most_common_count,
            "channels_distribution": dict(channels_counter),
            "avg_entropy": avg_entropy,
            "sample_summaries": sample_summaries,
            "unreadable_count": unreadable,
            "sampled_paths_count": len(items),
            # convenience fields for compatibility checks
            "shapes": [tuple(map(int, s.split("x"))) for s in shape_counter.keys()] if shape_counter else [],
            "channels": list(channels_counter.keys()),
            "min_entropy": min(entropy_vals) if entropy_vals else None,
            "max_entropy": max(entropy_vals) if entropy_vals else None,
        }
    except Exception as exc:
        logger.exception("Unexpected error in analyze_dataset: %s", exc)
        return {"error": "Internal dataset analyzer failure", "detail": str(exc)}
