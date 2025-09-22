"""
compatibility.py — model/dataset compatibility checks for imgshape v2.2.0

This module exposes:
- check_compatibility(model, dataset_path, **kwargs)  (preferred)
- check_model_compatibility(dataset_path, model=..., **kwargs)  (legacy alias accepted)
"""

from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("imgshape.compatibility")
if not logger.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# Attempt imports; fall back gracefully.
try:
    from imgshape.analyze import analyze_dataset
except Exception:
    analyze_dataset = None
    logger.warning("analyze_dataset not available; compatibility checks will be limited.")

try:
    from imgshape.recommender import recommend_preprocessing
except Exception:
    recommend_preprocessing = None
    logger.warning("recommend_preprocessing not available; recommendations will be conservative.")


def _infer_expected_shape_from_model(model_name: str) -> Optional[Tuple[int, int, int]]:
    if not model_name:
        return None
    name = model_name.lower()
    if "resnet" in name or "mobilenet" in name or "efficientnet" in name:
        return (224, 224, 3)
    if "inception" in name:
        return (299, 299, 3)
    if "vit" in name or "visiontransformer" in name:
        return (224, 224, 3)
    if "gray" in name or name.endswith("_gray") or name.endswith("_grey"):
        return (224, 224, 1)
    return None


def _shape_matches(observed: Tuple[int, int], expected: Tuple[int, int, int]) -> bool:
    if not observed or not expected:
        return False
    oh, ow = observed
    eh, ew, _ = expected
    return (oh == eh and ow == ew) or (oh == ew and ow == eh)


def _safe_analyze(dataset_path: Path, **kwargs) -> Dict[str, Any]:
    """
    Safe wrapper around analyze_dataset. Returns a dict with analysis or an error dict.
    """
    if analyze_dataset is None:
        return {"error": "analyze_unavailable", "message": "analyze_dataset not available."}
    try:
        return analyze_dataset(str(dataset_path), **kwargs)
    except Exception as e:
        logger.warning("analyze_dataset failed: %s", e)
        return {"error": "analysis_failed", "message": str(e)}


def _safe_recommend(analysis: Dict[str, Any], model: str, **kwargs) -> Dict[str, Any]:
    """
    Safe wrapper around recommend_preprocessing. Returns recommendation dict or error dict.
    """
    if recommend_preprocessing is None:
        return {"error": "recommender_unavailable", "message": "recommend_preprocessing not available."}
    try:
        # Some recommenders accept model_name keyword, others accept only analysis
        try:
            return recommend_preprocessing(analysis or {}, model_name=model, **kwargs)
        except TypeError:
            return recommend_preprocessing(analysis or {}, **kwargs)
    except Exception as e:
        logger.warning("recommend_preprocessing failed: %s", e)
        return {"error": "recommender_failed", "message": str(e)}


def _fallback_recommendations(report: Dict[str, Any], expected_shape: Optional[Tuple[int, int, int]] = None) -> Dict[str, Any]:
    """
    Conservative fallback recommendations when recommender is unavailable.
    """
    rec = {"actions": [], "augmentations": {"training": ["random_crop", "horizontal_flip"], "validation": ["center_crop"]}}

    if expected_shape:
        h, w, _ = expected_shape
        rec["actions"].append({"type": "resize", "height": h, "width": w, "mode": "center_crop_then_resize"})
    else:
        rec["actions"].append({"type": "resize_suggestion", "message": "Use 224x224 or 256x256 depending on model."})

    # Normalization
    rec["actions"].append({"type": "normalize", "message": "Use ImageNet mean/std for ImageNet models."})
    return rec


def check_compatibility(model: str, dataset_path: str, allow_partial: bool = True, verbose: bool = False, **kwargs) -> Dict[str, Any]:
    """
    Assess compatibility between `model` (name/config string) and a local dataset path.

    Returns a dict:
    {
      "model": <model>,
      "dataset_summary": {...} or None,
      "compatibility_report": {"status": "ok|warning|error", "checks": [...]},
      "recommendations": {...},
      "total": <int>  # number of images examined or 0
    }
    """
    if verbose:
        logger.setLevel(logging.INFO)

    result: Dict[str, Any] = {"model": model}

    p = Path(dataset_path)
    if not p.exists():
        err = {"status": "error", "error": "dataset_not_found", "message": f"Path not found: {dataset_path}"}
        logger.warning(err["message"])
        return {"model": model, "dataset_summary": None, "compatibility_report": err, "recommendations": None, "total": 0}

    # Analysis (best-effort)
    analysis = _safe_analyze(p, **kwargs)
    dataset_summary = None
    if analysis and not analysis.get("error"):
        # Map analysis fields into a compact summary
        dataset_summary = {
            "image_count": int(analysis.get("image_count", 0)),
            "unique_shapes": analysis.get("unique_shapes") or analysis.get("shapes") or {},
            "channels_distribution": analysis.get("channels_distribution") or analysis.get("channels") or {},
            "avg_entropy": analysis.get("avg_entropy"),
            "min_entropy": analysis.get("min_entropy"),
            "max_entropy": analysis.get("max_entropy"),
            "unreadable_files": analysis.get("unreadable_count") or analysis.get("unreadable_files") or [],
        }
    else:
        dataset_summary = {"error": analysis.get("error") if isinstance(analysis, dict) else "analysis_missing", "detail": analysis}

    result["dataset_summary"] = dataset_summary

    # Build compatibility report
    report: Dict[str, Any] = {"status": "unknown", "checks": []}
    if isinstance(dataset_summary, dict) and dataset_summary.get("error"):
        report["status"] = "partial"
        report["checks"].append({"name": "analysis", "result": "missing", "detail": dataset_summary.get("detail")})
    else:
        img_count = dataset_summary.get("image_count", 0)
        if img_count == 0:
            report["checks"].append({"name": "image_count", "result": "error", "message": "No images found."})
        else:
            report["checks"].append({"name": "image_count", "result": "ok", "value": img_count})

        expected_shape = _infer_expected_shape_from_model(model)
        observed_shapes = list(dataset_summary.get("unique_shapes", {}).keys()) if isinstance(dataset_summary.get("unique_shapes"), dict) else dataset_summary.get("unique_shapes") or []

        if expected_shape:
            # observed_shapes are likely strings "WxH" or tuples; normalize to tuples when possible
            matches = []
            for s in observed_shapes:
                try:
                    if isinstance(s, str) and "x" in s:
                        w, h = s.split("x")
                        if _shape_matches((int(h), int(w)), expected_shape):
                            matches.append(s)
                    elif isinstance(s, (list, tuple)) and len(s) >= 2:
                        if _shape_matches((int(s[0]), int(s[1])), expected_shape):
                            matches.append(s)
                except Exception:
                    continue
            if matches:
                report["checks"].append({"name": "shape", "result": "ok", "expected": expected_shape, "examples": matches[:5]})
            else:
                report["checks"].append({"name": "shape", "result": "warning", "expected": expected_shape, "observed_sample": observed_shapes[:5]})
        else:
            report["checks"].append({"name": "shape_expectation", "result": "unknown"})

        ch_keys = list(dataset_summary.get("channels_distribution", {}).keys()) if isinstance(dataset_summary.get("channels_distribution"), dict) else dataset_summary.get("channels_distribution") or []
        if not ch_keys:
            report["checks"].append({"name": "channels", "result": "warning", "message": "No channel info available."})
        else:
            if set(map(int, ch_keys)) == {3}:
                report["checks"].append({"name": "channels", "result": "ok", "message": "RGB detected."})
            elif set(map(int, ch_keys)) == {1}:
                report["checks"].append({"name": "channels", "result": "warning", "message": "Grayscale images detected."})
            elif 4 in map(int, ch_keys):
                report["checks"].append({"name": "channels", "result": "warning", "message": "Alpha channel present in some images."})
            else:
                report["checks"].append({"name": "channels", "result": "warning", "message": f"Channels observed: {ch_keys}"})

        # Entropy checks
        avg_ent = dataset_summary.get("avg_entropy")
        if avg_ent is not None:
            report["checks"].append({"name": "entropy", "result": "ok", "avg_entropy": avg_ent})

        unread = dataset_summary.get("unreadable_files", []) or []
        if unread:
            report["checks"].append({"name": "unreadable_files", "result": "warning", "examples": unread[:5]})

        # derive overall
        if any(c.get("result") == "error" for c in report["checks"]):
            report["status"] = "error"
        elif any(c.get("result") == "warning" for c in report["checks"]):
            report["status"] = "warning"
        else:
            report["status"] = "ok"

    result["compatibility_report"] = report

    # Recommendations (prefer recommender, fallback to conservative rules)
    recommendations = None
    if not isinstance(dataset_summary, dict) or dataset_summary.get("error"):
        # if analysis missing, try calling recommender with minimal input
        if recommend_preprocessing:
            try:
                recommendations = _safe_recommend({}, model, **kwargs)
            except Exception:
                recommendations = _fallback_recommendations(report, expected_shape=_infer_expected_shape_from_model(model))
        else:
            recommendations = _fallback_recommendations(report, expected_shape=_infer_expected_shape_from_model(model))
    else:
        recommendations = _safe_recommend(analysis or {}, model, **kwargs)
        if not recommendations or isinstance(recommendations, dict) and recommendations.get("error"):
            recommendations = _fallback_recommendations(report, expected_shape=_infer_expected_shape_from_model(model))

    result["recommendations"] = recommendations

    # Ensure "total" exists (number of images considered)
    # Ensure "total" exists (number of images considered)
    total_count = 0
    try:
        total_count = int(dataset_summary.get("image_count", 0)) if isinstance(dataset_summary, dict) else 0
    except Exception:
        total_count = 0
    result["total"] = total_count

    # Count passed checks (result == "ok")
    try:
        checks = result.get("compatibility_report", {}).get("checks", [])
        passed_count = sum(1 for c in checks if c.get("result") == "ok")
    except Exception:
        passed_count = 0
    result["passed"] = passed_count

    return result


def check_model_compatibility(*args, **kwargs):
    """
    Backwards-compatible alias. Accepts both:
      - check_model_compatibility(dataset_path, model="name")
      - check_model_compatibility(model, dataset_path)
    and delegates to check_compatibility without causing multiple-value errors.
    """
    logger.warning("check_model_compatibility is deprecated; use check_compatibility instead.")

    # Common legacy pattern in tests: check_model_compatibility("dataset_path", model="mobilenet_v2")
    if len(args) == 1 and "model" in kwargs:
        dataset_path = args[0]
        model = kwargs.pop("model")
        return check_compatibility(model=model, dataset_path=dataset_path, **kwargs)

    # If called as check_model_compatibility(model, dataset_path)
    if len(args) >= 2:
        # assume first is model, second is dataset_path
        model = args[0]
        dataset_path = args[1]
        return check_compatibility(model=model, dataset_path=dataset_path, **kwargs)

    # Fallback to passing through kwargs (may raise if signature mismatched)
    return check_compatibility(*args, **kwargs)


__all__ = ["check_compatibility", "check_model_compatibility"]
