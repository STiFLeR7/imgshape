# src/imgshape/analyze.py
"""
analyze.py — robust image/dataset analysis for imgshape v3 (backwards-compatible)

Improvements over v2:
- analyze_dataset distinguishes between a single-file path and a directory.
- Scanner ignores obvious temp/cache/hidden folders (e.g. .streamlit uploads/tmp/cache).
- Defensive behavior: never raises; returns structured error dicts.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Iterable, List, Tuple
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
# Helpers: robustly open images
# ----------------------
def _open_image_from_path(path: Path) -> Optional[Image.Image]:
    """Try to open a path with PIL.Image.open(path). Convert to RGB if possible."""
    try:
        if not path.exists() or not path.is_file():
            return None
        img = Image.open(path)
        try:
            return img.convert("RGB")
        except Exception:
            return img
    except UnidentifiedImageError:
        logger.debug("PIL could not identify image: %s", path)
        return None
    except Exception:
        logger.debug("Failed opening path %s", path, exc_info=True)
        return None


def _open_image_from_bytes(data: bytes) -> Optional[Image.Image]:
    try:
        img = Image.open(BytesIO(data))
        try:
            return img.convert("RGB")
        except Exception:
            return img
    except UnidentifiedImageError:
        return None
    except Exception:
        logger.debug("Failed opening image from bytes", exc_info=True)
        return None


def _string_candidates(s: str) -> List[Path]:
    """
    Create prioritized candidate paths from the provided string.
    Order:
      1. Exact path as given
      2. expanduser()
      3. cwd / s
      4. package repo-root relative (two levels up from this file)
      5. assets/ + s next to package
      6. ascend parents from cwd (small depth)
    """
    candidates: List[Path] = []
    try:
        p = Path(s)
    except Exception:
        p = None

    if p is not None:
        candidates.append(p)
        try:
            candidates.append(p.expanduser())
        except Exception:
            pass

    try:
        candidates.append(Path.cwd() / s)
    except Exception:
        pass

    # repo root (two levels up from src/imgshape)
    try:
        repo_root = Path(__file__).resolve().parents[2]
        candidates.append(repo_root / s)
    except Exception:
        pass

    # package assets folder
    try:
        pkg_assets = Path(__file__).resolve().parents[1] / "assets" / s
        candidates.append(pkg_assets)
    except Exception:
        pass

    # try climb up from cwd (helpful for tests running in tmpdir)
    try:
        cur = Path.cwd()
        for _ in range(4):
            candidates.append(cur / s)
            cur = cur.parent
    except Exception:
        pass

    # Deduplicate while preserving order
    seen = set()
    out: List[Path] = []
    for c in candidates:
        try:
            key = str(c.resolve(strict=False))
        except Exception:
            key = str(c)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _open_image_from_input(inp: Any, allow_network: bool = False) -> Optional[Image.Image]:
    """Open image from PIL.Image, bytes, file-like, path string, or Path."""
    if inp is None:
        return None

    # PIL image
    try:
        if isinstance(inp, Image.Image):
            try:
                return inp.convert("RGB")
            except Exception:
                return inp
    except Exception:
        logger.debug("Not a PIL image", exc_info=True)

    # bytes-like
    try:
        if isinstance(inp, (bytes, bytearray)):
            return _open_image_from_bytes(bytes(inp))
    except Exception:
        logger.debug("Not bytes", exc_info=True)

    # Path-like / string
    try:
        if isinstance(inp, (str, Path)):
            s = str(inp)
            # Network fetch if allowed
            if (s.startswith("http://") or s.startswith("https://")) and allow_network:
                try:
                    import requests

                    r = requests.get(s, timeout=8)
                    r.raise_for_status()
                    return _open_image_from_bytes(r.content)
                except Exception:
                    logger.debug("Failed to fetch URL %s", s, exc_info=True)

            # Try candidate file paths
            candidates = _string_candidates(s)
            logger.debug("Trying candidates for %s -> %s", s, candidates)
            for cand in candidates:
                img = _open_image_from_path(cand)
                if img is not None:
                    return img
    except Exception:
        logger.debug("String/Path handling failed", exc_info=True)

    # file-like readable (has read)
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
                    return _open_image_from_bytes(data)
                if pos is not None:
                    try:
                        inp.seek(pos)
                    except Exception:
                        pass
            except Exception:
                logger.debug("File-like read failed", exc_info=True)
    except Exception:
        logger.debug("Input is not file-like", exc_info=True)

    return None


# ----------------------
# Analysis helpers
# ----------------------
def _safe_meta(pil: Image.Image) -> Dict[str, Any]:
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
        logger.debug("Failed to extract meta", exc_info=True)
        return {}


def _entropy_from_image(pil: Image.Image) -> Optional[float]:
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
        logger.debug("Entropy computation failed", exc_info=True)
        return None


def _guess_image_type(meta: Dict[str, Any], entropy: Optional[float]) -> str:
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


def _is_probable_temp_or_hidden(path: Path) -> bool:
    """
    Heuristic to skip Streamlit/temp/cache/hidden folders when scanning datasets.
    Avoids counting upload caches and developer temp files.
    """
    s = str(path).lower()
    # skip hidden filenames / directories starting with '.' or containing common temp/cache names
    if any(part.startswith(".") for part in path.parts):
        return True
    for token in (".streamlit", "uploads", "tmp", "temp", "cache", "thumbs"):
        if token in s:
            return True
    return False


# ----------------------
# Public API
# ----------------------
def analyze_type(input_obj: Any, allow_network: bool = False) -> Dict[str, Any]:
    """
    Analyze single image-like input and return:
      {"meta": {...}, "entropy": float, "suggestions": {...}, "guess_type": str}
    Returns an error dict on failure (never raises).
    """
    try:
        pil = _open_image_from_input(input_obj, allow_network=allow_network)
        if pil is None:
            logger.debug("analyze_type: unsupported input: %r", input_obj)
            return {"error": "Unsupported input for analyze_type"}

        meta = _safe_meta(pil)
        ent = _entropy_from_image(pil)
        if meta is not None:
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

    Behavior improvements:
    - If dataset_input is a file path, treat it as a single-image dataset.
    - Skip obvious cache/temp/hidden folders to avoid counting upload caches.
    """
    try:
        items: List[Any] = []

        # If a string/path, try to resolve candidates and decide if it's a file or directory.
        if isinstance(dataset_input, (str, Path)):
            p = Path(dataset_input).expanduser()
            # if exact file -> treat as single-image dataset
            if p.exists() and p.is_file():
                # single file input -> return a dataset-like summary for the single image
                pil = _open_image_from_input(p)
                if pil is None:
                    return {"error": f"Could not open file: {str(p)}"}
                meta = _safe_meta(pil)
                ent = _entropy_from_image(pil)
                sample_summary = analyze_type(pil)
                return {
                    "image_count": 1,
                    "unique_shapes": {f"{meta.get('width')}x{meta.get('height')}": 1} if meta else {},
                    "most_common_shape": f"{meta.get('width')}x{meta.get('height')}" if meta else None,
                    "most_common_shape_count": 1,
                    "channels_distribution": {meta.get("channels"): 1} if meta else {},
                    "avg_entropy": ent,
                    "sample_summaries": [sample_summary],
                    "unreadable_count": 0,
                    "sampled_paths_count": 1,
                    "shapes": [(int(meta.get("width")), int(meta.get("height")))] if meta else [],
                    "channels": [meta.get("channels")] if meta else [],
                    "min_entropy": ent,
                    "max_entropy": ent,
                }

            # if it's not a file, try to find a directory candidate
            if p.exists() and p.is_dir():
                base_dir = p
            else:
                # fallbacks: cwd/p, repo-relative, assets/
                found = None
                candidates = _string_candidates(str(p))
                for c in candidates:
                    try:
                        if c.exists() and c.is_dir():
                            found = c
                            break
                    except Exception:
                        continue
                if found is None:
                    return {"error": f"Dataset path invalid: {dataset_input}"}
                base_dir = found

            exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
            # Walk and collect image files, but skip probable temp/hidden folders
            for f in sorted(base_dir.rglob("*")):
                try:
                    if not f.is_file():
                        continue
                    # skip hidden / temp paths heuristically
                    if _is_probable_temp_or_hidden(f) or _is_probable_temp_or_hidden(f.parent):
                        logger.debug("Skipping probable temp/hidden file: %s", f)
                        continue
                    if f.suffix and f.suffix.lower() in exts:
                        items.append(f)
                except Exception:
                    logger.debug("Error inspecting file during dataset scan: %s", f, exc_info=True)
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
            # convenience fields
            "shapes": [tuple(map(int, s.split("x"))) for s in shape_counter.keys()] if shape_counter else [],
            "channels": list(channels_counter.keys()),
            "min_entropy": min(entropy_vals) if entropy_vals else None,
            "max_entropy": max(entropy_vals) if entropy_vals else None,
        }
    except Exception as exc:
        logger.exception("Unexpected error in analyze_dataset: %s", exc)
        return {"error": "Internal dataset analyzer failure", "detail": str(exc)}
