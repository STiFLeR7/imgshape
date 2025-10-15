# src/imgshape/recommender.py
"""
Robust recommender for imgshape v3 (backwards-compatible with v2 functions).

Exports:
- recommend_preprocessing(input_obj, user_prefs=None)
- recommend_dataset(dataset_input, sample_limit=200, user_prefs=None)
- RecommendEngine(profile=None)  <-- new class used by GUI / server

The RecommendEngine class provides convenience methods expected by the v3 GUI/server:
- recommend_from_bytes(bytes_or_buffer)
- recommend_from_image(PIL.Image)
- recommend_from_analysis(dict)
- recommend_from_dataset(path)
"""

from __future__ import annotations

import math
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Mapping, Union
from io import BytesIO
from PIL import Image
from collections import Counter
import yaml

logger = logging.getLogger("imgshape.recommender")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


# -----------------------
# Internal helpers
# -----------------------
def _open_image_from_input(inp: Any) -> Optional[Image.Image]:
    """Open various input forms as a PIL.Image (RGB)."""
    if inp is None:
        return None
    try:
        if isinstance(inp, Image.Image):
            return inp.convert("RGB")
    except Exception:
        logger.debug("Not a PIL image object", exc_info=True)

    try:
        if isinstance(inp, (bytes, bytearray)):
            return Image.open(BytesIO(inp)).convert("RGB")
    except Exception:
        logger.debug("Failed to open bytes", exc_info=True)

    try:
        if isinstance(inp, (str, Path)):
            p = Path(inp)
            if p.exists() and p.is_file():
                return Image.open(p).convert("RGB")
    except Exception:
        logger.debug("Failed to open path", exc_info=True)

    try:
        if hasattr(inp, "read"):
            try:
                inp.seek(0)
            except Exception:
                pass
            data = inp.read()
            if not data:
                return None
            if isinstance(data, str):
                data = data.encode("utf-8")
            return Image.open(BytesIO(data)).convert("RGB")
    except Exception:
        logger.debug("Failed to open file-like object", exc_info=True)

    return None


def _shape_from_image(pil: Image.Image) -> Optional[Tuple[int, int, int]]:
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
        logger.debug("entropy_from_image failed", exc_info=True)
        return None


def _defaults_for_channels(channels: int):
    if channels == 1:
        return [0.5], [0.5]
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def interpret_prefs(prefs: Optional[List[str]]) -> Dict[str, str]:
    out = {"bias": "neutral"}
    if not prefs:
        return out
    s = " ".join(prefs).lower()
    if any(x in s for x in ("fast", "latency", "speed")):
        out["bias"] = "fast"
    if any(x in s for x in ("small", "tiny", "edge", "mobile")):
        out["bias"] = "small"
    if any(x in s for x in ("high", "quality", "best", "accuracy", "precise")):
        out["bias"] = "quality"
    return out


def _deterministic_fallback_preprocessing(user_prefs: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "error": "fallback",
        "message": "Could not compute recommendation; returning safe defaults.",
        "user_prefs": user_prefs or [],
        "bias": interpret_prefs(user_prefs).get("bias", "neutral"),
        "augmentation_plan": {
            "order": ["RandomHorizontalFlip"],
            "augmentations": [
                {"name": "RandomHorizontalFlip", "params": {"p": 0.5}, "reason": "Default conservative augmentation", "score": 0.4}
            ],
        },
        "resize": {"size": [224, 224], "method": "bilinear", "suggested_model": "ResNet/MobileNet"},
        "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "mode": "RGB",
    }


def _deterministic_fallback_dataset(user_prefs: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "error": "fallback",
        "message": "Could not analyse dataset; returning fallback summary.",
        "user_prefs": user_prefs or [],
        "dataset_summary": {"image_count": 0, "unique_shapes": {}, "notes": "fallback"},
        "representative_preprocessing": _deterministic_fallback_preprocessing(user_prefs),
    }


def _stats_from_pil(pil: Image.Image) -> Dict[str, Any]:
    shp = _shape_from_image(pil)
    stats: Dict[str, Any] = {}
    if shp:
        h, w, c = shp
        stats["avg_width"] = int(w)
        stats["avg_height"] = int(h)
        stats["channels"] = int(c)
        stats["shape_distribution"] = {f"{w}x{h}": 1}
    else:
        stats["channels"] = 3
    stats["entropy_mean"] = _entropy_from_image(pil)
    stats["image_count"] = 1
    return stats


def _generate_augmentation_plan_from_stats(stats: Mapping) -> Dict[str, Any]:
    plan: Dict[str, Any] = {"order": [], "augmentations": []}
    entropy = stats.get("entropy_mean")
    channels = int(stats.get("channels", 3))
    avg_w = stats.get("avg_width")
    avg_h = stats.get("avg_height")
    image_count = int(stats.get("image_count", 0) or 0)

    try:
        min_side = min(int(avg_w), int(avg_h)) if avg_w and avg_h else 224
    except Exception:
        min_side = 224

    def add(name: str, params: dict, reason: str, score: float):
        if name not in plan["order"]:
            plan["order"].append(name)
            plan["augmentations"].append({"name": name, "params": params, "reason": reason, "score": round(float(score), 2)})

    if min_side >= 32:
        add("RandomHorizontalFlip", {"p": 0.5}, "Default safe flip for many datasets", 0.7)

    if min_side >= 96 and image_count > 5:
        add("RandomRotation", {"degrees": 15}, "Small rotations to increase orientation robustness", 0.6)

    if channels == 3 and (entropy is None or entropy >= 3.0):
        add("ColorJitter", {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.05}, "Color variations", 0.5)

    if entropy is not None and entropy < 2.0:
        add("RandomAdjustSharpness", {"sharpness_factor": 1.2}, "Low-entropy image adjustments", 0.45)

    if min_side >= 224:
        add("RandomResizedCrop", {"size": 224, "scale": [0.8, 1.0]}, "Resize crop for big images", 0.6)

    if not plan["augmentations"]:
        add("RandomHorizontalFlip", {"p": 0.5}, "Conservative default", 0.4)

    return plan


# -----------------------
# Core functional API (backwards-compatible)
# -----------------------
def recommend_preprocessing(input_obj: Any, user_prefs: Optional[List[str]] = None) -> Dict[str, Any]:
    """Recommend preprocessing for a single image (path/PIL/bytes/file-like)."""
    try:
        pil = _open_image_from_input(input_obj)
        if pil is None:
            logger.warning("recommend_preprocessing: could not open input as image, returning fallback")
            return _deterministic_fallback_preprocessing(user_prefs)

        stats = _stats_from_pil(pil)
        plan = _generate_augmentation_plan_from_stats(stats)
        pref_meta = interpret_prefs(user_prefs or [])

        try:
            min_side = min(int(stats.get("avg_width", 224)), int(stats.get("avg_height", 224)))
        except Exception:
            min_side = 224

        bias = pref_meta.get("bias", "neutral")

        # bias-influenced sizing
        if bias == "fast":
            if min_side >= 96:
                size = [96, 96]; method = "bilinear"; suggested_model = "EfficientNet-B0 (fast)"
            else:
                size = [64, 64]; method = "nearest"; suggested_model = "TinyNet (very fast)"
        elif bias == "small":
            size = [96, 96] if min_side >= 96 else [64, 64]; method = "bilinear"; suggested_model = "MobileNetV2 (edge)"
        elif bias == "quality":
            size = [224, 224] if min_side >= 224 else [128, 128]; method = "bilinear"; suggested_model = "ResNet50 / EfficientNet-Lite (quality)"
        else:
            if min_side >= 224:
                size = [224, 224]; method = "bilinear"; suggested_model = "ResNet18 / MobileNetV2"
            elif min_side >= 96:
                size = [96, 96]; method = "bilinear"; suggested_model = "EfficientNet-B0 (small)"
            elif min_side <= 32:
                size = [32, 32]; method = "nearest"; suggested_model = "TinyNet / CIFAR style"
            else:
                size = [128, 128]; method = "bilinear"; suggested_model = "General-purpose (mid)"

        # tweak augmentation scores a bit by bias
        if bias == "quality":
            for a in plan["augmentations"]:
                a["score"] = round(min(1.0, a.get("score", 0.0) + 0.05), 3)
        elif bias in ("fast", "small"):
            for a in plan["augmentations"]:
                if a["name"] == "RandomResizedCrop":
                    a["score"] = round(max(0.0, a.get("score", 0.0) - 0.15), 3)

        channels = int(stats.get("channels", 3))
        mean, std = _defaults_for_channels(channels)

        return {
            "user_prefs": user_prefs or [],
            "bias": bias,
            "augmentation_plan": plan,
            "resize": {"size": size, "method": method, "suggested_model": suggested_model},
            "normalize": {"mean": mean, "std": std},
            "mode": "RGB" if channels == 3 else "Grayscale",
            "entropy": stats.get("entropy_mean"),
            "channels": channels,
            "image_count": stats.get("image_count", 1),
        }
    except Exception as exc:
        logger.exception("recommend_preprocessing failed: %s", exc)
        return _deterministic_fallback_preprocessing(user_prefs)


def recommend_dataset(dataset_input: Any, sample_limit: int = 200, user_prefs: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Dataset-level recommender that ALWAYS returns:
    { "user_prefs": [...], "dataset_summary": {...}, "representative_preprocessing": {...} }
    """
    try:
        # build list of candidate image paths
        images: List[Any] = []
        if isinstance(dataset_input, (str, Path)):
            p = Path(dataset_input).expanduser().resolve()
            logger.info("recommend_dataset: resolved dataset path -> %s", p)
            if not p.exists() or not p.is_dir():
                logger.warning("recommend_dataset: invalid dataset path: %s", dataset_input)
                return _deterministic_fallback_dataset(user_prefs)
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix and f.suffix.lower() in exts:
                    images.append(f)
        elif isinstance(dataset_input, Iterable):
            images = list(dataset_input)[:sample_limit]
        else:
            logger.warning("recommend_dataset: unsupported input type %r", type(dataset_input))
            return _deterministic_fallback_dataset(user_prefs)

        sampled_paths_count = len(images)
        if sampled_paths_count == 0:
            logger.warning("recommend_dataset: no images found in %s", dataset_input)
            return _deterministic_fallback_dataset(user_prefs)

        total = 0
        shape_counter = Counter()
        entropy_vals: List[float] = []
        channels_set = set()
        unreadable = 0
        rep = None

        for item in images:
            if total >= sample_limit:
                break
            pil = _open_image_from_input(item)
            if pil is None:
                unreadable += 1
                continue
            total += 1
            shp = _shape_from_image(pil)
            if shp:
                h, w, c = shp
                shape_counter[f"{w}x{h}"] += 1
                channels_set.add(c)
            ent = _entropy_from_image(pil)
            if ent is not None:
                entropy_vals.append(ent)
            if rep is None:
                rep = pil

        if total == 0:
            logger.warning("recommend_dataset: no readable images after scanning")
            return _deterministic_fallback_dataset(user_prefs)

        avg_entropy = round(float(sum(entropy_vals) / len(entropy_vals)), 3) if entropy_vals else None
        most_common_shape, most_common_count = shape_counter.most_common(1)[0] if shape_counter else (None, 0)
        channels = sorted(list(channels_set)) if channels_set else [3]

        stats = {
            "image_count": total,
            "unique_shapes": dict(shape_counter),
            "most_common_shape": most_common_shape,
            "most_common_shape_count": most_common_count,
            "avg_entropy": avg_entropy,
            "channels": channels,
            "unreadable_count": unreadable,
            "sampled_paths_count": sampled_paths_count,
        }

        if rep is None:
            logger.warning("recommend_dataset: could not open representative image")
            return _deterministic_fallback_dataset(user_prefs)

        pre = recommend_preprocessing(rep, user_prefs=user_prefs)

        return {
            "user_prefs": user_prefs or [],
            "dataset_summary": stats,
            "representative_preprocessing": pre,
        }
    except Exception as exc:
        logger.exception("recommend_dataset failed: %s", exc)
        return _deterministic_fallback_dataset(user_prefs)


# -----------------------
# RecommendEngine class (v3 API)
# -----------------------
class RecommendEngine:
    """
    Thin engine wrapper that provides methods the v3 GUI/server expect.

    Example:
        engine = RecommendEngine(profile="imagenet-small")
        rec = engine.recommend_from_image(pil_img)
        rec2 = engine.recommend_from_dataset("assets/sample_images")
        rec3 = engine.recommend_from_bytes(open("img.jpg","rb").read())
        rec4 = engine.recommend_from_analysis({"avg_width":32, "avg_height":32, "channels":3, "entropy_mean":4.0})
    """

    def __init__(self, profile: Optional[Union[str, Dict[str, Any]]] = None):
        """
        profile: either a dict (already parsed profile) or a filename (in src/imgshape/profiles/) or None.
        If a string and matches a YAML in src/imgshape/profiles, it will be loaded.
        """
        self.profile = None
        if isinstance(profile, dict):
            self.profile = profile
        elif isinstance(profile, str):
            # try to load from package profiles directory
            # profiles are expected at src/imgshape/profiles/<name>.yaml or provided full path
            p = Path(profile)
            if p.exists():
                try:
                    self.profile = yaml.safe_load(p.read_text(encoding="utf-8"))
                except Exception:
                    logger.debug("Failed to parse provided profile path %s", p, exc_info=True)
            else:
                # try package-relative
                pkg_profiles = Path(__file__).resolve().parent.joinpath("profiles").joinpath(profile)
                if pkg_profiles.exists():
                    try:
                        self.profile = yaml.safe_load(pkg_profiles.read_text(encoding="utf-8"))
                    except Exception:
                        logger.debug("Failed to parse profile %s", pkg_profiles, exc_info=True)
        # else None -> no profile

    # ---- single-image entrypoints ----
    def recommend_from_image(self, pil: Image.Image, user_prefs: Optional[List[str]] = None) -> Dict[str, Any]:
        try:
            return recommend_preprocessing(pil, user_prefs=user_prefs)
        except Exception:
            logger.exception("RecommendEngine.recommend_from_image failed")
            return _deterministic_fallback_preprocessing(user_prefs)

    def recommend_from_bytes(self, b: Union[bytes, BytesIO], user_prefs: Optional[List[str]] = None) -> Dict[str, Any]:
        try:
            if isinstance(b, BytesIO):
                b.seek(0)
                data = b.read()
            else:
                data = b
            pil = _open_image_from_input(data)
            if pil is None:
                return _deterministic_fallback_preprocessing(user_prefs)
            return self.recommend_from_image(pil, user_prefs=user_prefs)
        except Exception:
            logger.exception("RecommendEngine.recommend_from_bytes failed")
            return _deterministic_fallback_preprocessing(user_prefs)

    def recommend_from_analysis(self, analysis: Dict[str, Any], user_prefs: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Accept an analysis dict (e.g. produced by analyze_type / analyze_dataset)
        and convert to a preprocessing recommendation deterministically.
        """
        try:
            # Build a minimal stats dict compatible with _generate_augmentation_plan_from_stats
            stats: Dict[str, Any] = {}
            # prefer explicit keys if present
            if "avg_width" in analysis or "avg_height" in analysis or "channels" in analysis:
                stats["avg_width"] = analysis.get("avg_width", analysis.get("resolution_mean", [None, None])[0] if isinstance(analysis.get("resolution_mean"), (list, tuple)) else None)
                stats["avg_height"] = analysis.get("avg_height", analysis.get("resolution_mean", [None, None])[1] if isinstance(analysis.get("resolution_mean"), (list, tuple)) else None)
                stats["channels"] = analysis.get("channels", 3)
                stats["entropy_mean"] = analysis.get("entropy_mean", analysis.get("entropy", None))
                stats["image_count"] = analysis.get("image_count", analysis.get("count", 1))
            elif "resolution_mean" in analysis and isinstance(analysis.get("resolution_mean"), (list, tuple)):
                res = analysis["resolution_mean"]
                stats["avg_width"] = res[0]
                stats["avg_height"] = res[1] if len(res) > 1 else res[0]
                stats["channels"] = analysis.get("channels", 3)
                stats["entropy_mean"] = analysis.get("entropy", None)
                stats["image_count"] = analysis.get("image_count", 1)
            else:
                # last resort: take some keys present in older formats
                stats["avg_width"] = analysis.get("width") or analysis.get("w") or 224
                stats["avg_height"] = analysis.get("height") or analysis.get("h") or 224
                stats["channels"] = analysis.get("channels", 3)
                stats["entropy_mean"] = analysis.get("entropy", analysis.get("entropy_mean", None))
                stats["image_count"] = analysis.get("image_count", analysis.get("count", 1))

            # generate augmentation plan from stats
            plan = _generate_augmentation_plan_from_stats(stats)
            pref_meta = interpret_prefs(user_prefs or [])
            bias = pref_meta.get("bias", "neutral")

            # decide resize based on stats
            try:
                min_side = min(int(stats.get("avg_width", 224)), int(stats.get("avg_height", 224)))
            except Exception:
                min_side = 224

            if bias == "fast":
                if min_side >= 96:
                    size = [96, 96]; method = "bilinear"; suggested_model = "EfficientNet-B0 (fast)"
                else:
                    size = [64, 64]; method = "nearest"; suggested_model = "TinyNet (very fast)"
            elif bias == "small":
                size = [96, 96] if min_side >= 96 else [64, 64]; method = "bilinear"; suggested_model = "MobileNetV2 (edge)"
            elif bias == "quality":
                size = [224, 224] if min_side >= 224 else [128, 128]; method = "bilinear"; suggested_model = "ResNet50 / EfficientNet-Lite (quality)"
            else:
                if min_side >= 224:
                    size = [224, 224]; method = "bilinear"; suggested_model = "ResNet18 / MobileNetV2"
                elif min_side >= 96:
                    size = [96, 96]; method = "bilinear"; suggested_model = "EfficientNet-B0 (small)"
                elif min_side <= 32:
                    size = [32, 32]; method = "nearest"; suggested_model = "TinyNet / CIFAR style"
                else:
                    size = [128, 128]; method = "bilinear"; suggested_model = "General-purpose (mid)"

            channels = int(stats.get("channels", 3))
            mean, std = _defaults_for_channels(channels)

            return {
                "user_prefs": user_prefs or [],
                "bias": bias,
                "augmentation_plan": plan,
                "resize": {"size": size, "method": method, "suggested_model": suggested_model},
                "normalize": {"mean": mean, "std": std},
                "mode": "RGB" if channels == 3 else "Grayscale",
                "entropy": stats.get("entropy_mean"),
                "channels": channels,
                "image_count": stats.get("image_count", 1),
                "meta_from_analysis": True,
            }
        except Exception as e:
            logger.exception("recommend_from_analysis failed: %s", e)
            return _deterministic_fallback_preprocessing(user_prefs)

    # ---- dataset entrypoints ----
    def recommend_from_dataset(self, dataset_input: Any, sample_limit: int = 200, user_prefs: Optional[List[str]] = None) -> Dict[str, Any]:
        try:
            return recommend_dataset(dataset_input, sample_limit=sample_limit, user_prefs=user_prefs)
        except Exception:
            logger.exception("recommend_from_dataset failed")
            return _deterministic_fallback_dataset(user_prefs)


# -----------------------
# simple smoke test
# -----------------------
if __name__ == "__main__":
    import json
    print("smoke test recommend_dataset on assets/sample_images")
    try:
        print(json.dumps(recommend_dataset("assets/sample_images"), indent=2))
    except Exception as e:
        print("smoke failed:", e)
