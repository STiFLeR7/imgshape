# src/imgshape/compatibility.py
"""
compatibility.py for imgshape v2.2.0

Purpose
-------
Provide a single well-documented function that returns a structured compatibility
report for a model and an image dataset.

Compatibility note
------------------
Older versions of the CLI (and third-party code) import `check_model_compatibility`.
To remain backwards-compatible we expose both:
  - check_compatibility(...)  (preferred new name)
  - check_model_compatibility(...)  (thin alias for backwards compatibility)

Design
------
- Uses safe fallbacks for analyze_dataset and recommend_preprocessing.
- Never raises on bad inputs; instead returns structured error dicts in the result.
- Logging is INFO-level by default and emits warnings when falling back.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Module logger. The package should configure handlers; we add a fallback stream handler if none present.
logger = logging.getLogger("imgshape.compatibility")
if not logger.handlers:
    import sys

    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

# Robust imports: try refactor paths first, then legacy paths, log warnings on fallbacks.
try:
    from imgshape.analyze import analyze_dataset  # preferred new location
except Exception:
    try:
        from imgshape.analyze import analyze_dataset  # legacy
        logger.warning("Imported analyze_dataset from imgshape.dataset.analysis (fallback).")
    except Exception:
        analyze_dataset = None
        logger.warning("analyze_dataset not found. Analysis-based checks will be limited.")

try:
    from imgshape.recommender import recommend_preprocessing
except Exception:
    try:
        from imgshape.recommender import recommend_preprocessing
        logger.warning("Imported recommend_preprocessing from imgshape.preprocessing.recommender (fallback).")
    except Exception:
        recommend_preprocessing = None
        logger.warning("recommend_preprocessing not found. Recommendations will be conservative defaults.")


# ---------------------- Internal safe-call helpers ----------------------


def _safe_analyze(dataset_path: Path, **kwargs) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Safely call analyze_dataset. Returns (analysis_result, error_dict) where exactly one is None.
    """
    if analyze_dataset is None:
        err = {"error": "analyze_dataset_unavailable", "message": "Dataset analysis function not available."}
        return None, err
    try:
        logger.info("Running dataset analysis for: %s", str(dataset_path))
        analysis = analyze_dataset(str(dataset_path), **kwargs)
        return analysis, None
    except Exception as e:
        logger.warning("analyze_dataset failed: %s", e)
        return None, {"error": "analysis_failed", "message": str(e)}


def _safe_recommend(analysis: Dict[str, Any], model: str, **kwargs) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Safely call recommend_preprocessing. Returns (recommendation, error_dict).
    """
    if recommend_preprocessing is None:
        return None, {"error": "recommender_unavailable", "message": "Preprocessing recommender not available."}
    try:
        logger.info("Generating recommendations for model: %s", model)
        rec = recommend_preprocessing(analysis, model_name=model, **kwargs)
        return rec, None
    except Exception as e:
        logger.warning("recommend_preprocessing failed: %s", e)
        return None, {"error": "recommender_failed", "message": str(e)}


# ---------------------- Small helpers ----------------------


def _infer_expected_shape_from_model(model_name: str) -> Optional[Tuple[int, int, int]]:
    """
    Heuristic inference of expected input shape (H, W, C) from model name.
    Conservative: use only common conventions.
    """
    if not model_name:
        return None
    name = model_name.lower()
    if "resnet" in name or "efficientnet" in name or "mobilenet" in name:
        return (224, 224, 3)
    if "inception" in name:
        return (299, 299, 3)
    if "vit" in name or "visiontransformer" in name:
        return (224, 224, 3)
    if "grayscale" in name or name.endswith("_gray") or name.endswith("_grey"):
        return (224, 224, 1)
    return None


def _shape_matches(observed: Tuple[int, int], expected: Tuple[int, int, int]) -> bool:
    """
    Return True if observed (h,w) matches expected (h,w,c) either directly or rotated.
    """
    if observed is None or expected is None:
        return False
    oh, ow = observed
    eh, ew, _ = expected
    return (oh == eh and ow == ew) or (oh == ew and ow == eh)


def _fallback_recommendations_from_report(report: Dict[str, Any], expected_shape: Optional[Tuple[int, int, int]] = None) -> Dict[str, Any]:
    """
    Produce conservative recommendations based on compatibility report when recommender is unavailable.
    """
    rec: Dict[str, Any] = {"actions": []}
    checks_by_name = {c.get("name"): c for c in report.get("checks", [])}

    # Resize suggestion
    shape_check = checks_by_name.get("shape") or {}
    if shape_check.get("result") == "ok":
        rec["actions"].append({"type": "noop", "message": "Images already match expected shape."})
    else:
        if expected_shape:
            h, w, c = expected_shape
            rec["actions"].append({
                "type": "resize",
                "message": f"Resize images to ({h}, {w}) using center-crop then resize to preserve aspect ratio.",
                "params": {"height": h, "width": w, "mode": "center_crop_then_resize"},
            })
        else:
            rec["actions"].append({"type": "resize_suggestion", "message": "Model input size unknown; use a square size like 224 or 256."})

    # Channels
    chan_check = checks_by_name.get("channels") or {}
    msg = (chan_check.get("message") or "").lower()
    if "grayscale" in msg or "grayscale" in str(chan_check.get("message", "")).lower() or chan_check.get("result") == "warning":
        rec["actions"].append({"type": "to_rgb", "message": "Convert grayscale images to RGB (repeat channels or convert appropriately)."})
    if "alpha" in msg or "4" in msg:
        rec["actions"].append({"type": "remove_alpha", "message": "Drop alpha channel or composite onto background before model input."})

    # Entropy
    entropy = checks_by_name.get("entropy")
    if entropy and entropy.get("min") is not None and entropy.get("min") < 1e-3:
        rec["actions"].append({"type": "inspect_low_entropy", "message": "Some images appear blank or low-information; inspect and remove/replace."})

    # Unreadable files
    unread = checks_by_name.get("unreadable_files")
    if unread:
        rec["actions"].append({"type": "fix_unreadable", "message": "Remove or re-encode unreadable files listed in report."})

    # Normalization suggestion (safe default)
    rec["actions"].append({
        "type": "normalize",
        "message": "Normalize using ImageNet mean/std if using pretrained ImageNet models; otherwise compute dataset mean/std.",
        "params": {"imagenet_default": True},
    })

    rec["augmentations"] = {
        "training": ["random_crop", "horizontal_flip", "color_jitter (small)"],
        "validation": ["center_crop"],
        "notes": "Adjust augmentations for domain-specific data (e.g., avoid color jitter for medical imagery)."
    }

    return rec


# ---------------------- Public API ----------------------


def check_compatibility(
    model: str,
    dataset_path: str,
    allow_partial: bool = True,
    verbose: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Assess compatibility between a model config/name and an image dataset.

    Returns a JSON-serializable dict with keys:
      - model: the provided model string
      - dataset_summary: aggregated dataset info or an error object
      - compatibility_report: structured assessment (status, checks...)
      - recommendations: concrete suggestions for resizing/normalizing/augmentations

    This function never raises; errors are returned inside the dict.
    """
    if verbose:
        logger.setLevel(logging.INFO)

    result: Dict[str, Any] = {"model": model}

    p = Path(dataset_path)
    if not p.exists():
        err = {"error": "dataset_not_found", "message": f"Path not found: {dataset_path}"}
        logger.warning(err["message"])
        return {"model": model, "dataset_summary": None, "compatibility_report": {"status": "error", **err}, "recommendations": None}

    # Step 1: analyze dataset (safe)
    analysis, analysis_err = _safe_analyze(p, **kwargs)
    if analysis is None:
        if not allow_partial:
            return {"model": model, "dataset_summary": None, "compatibility_report": {"status": "error", **(analysis_err or {})}, "recommendations": None}
        dataset_summary = {"error": "analysis_missing", "detail": analysis_err}
    else:
        dataset_summary = {
            "image_count": int(analysis.get("image_count", 0)),
            "shapes": analysis.get("shapes", []),
            "unique_shapes_count": int(len(set(map(tuple, analysis.get("shapes", []))))),
            "channels": sorted(list(set(analysis.get("channels", [])))) if analysis.get("channels") else analysis.get("channels"),
            "avg_entropy": float(analysis.get("avg_entropy")) if analysis.get("avg_entropy") is not None else None,
            "min_entropy": float(analysis.get("min_entropy")) if analysis.get("min_entropy") is not None else None,
            "max_entropy": float(analysis.get("max_entropy")) if analysis.get("max_entropy") is not None else None,
            "unreadable_files": analysis.get("unreadable_files", []),
        }

    result["dataset_summary"] = dataset_summary

    # Step 2: compatibility checks
    report: Dict[str, Any] = {"status": "unknown", "checks": []}

    if analysis is None:
        report["status"] = "partial"
        report["checks"].append({"name": "analysis", "result": "missing", "reason": analysis_err})
    else:
        img_count = dataset_summary.get("image_count", 0)
        if img_count == 0:
            report["checks"].append({"name": "image_count", "result": "error", "message": "No images found in dataset."})
        else:
            report["checks"].append({"name": "image_count", "result": "ok", "value": img_count})

        expected_shape = _infer_expected_shape_from_model(model)
        observed_shapes = [tuple(s) for s in (analysis.get("shapes") or [])]

        if expected_shape is None:
            report["checks"].append({"name": "shape_expectation", "result": "unknown", "message": "Could not infer expected input shape from model name."})
        else:
            matches = [s for s in observed_shapes if _shape_matches(s, expected_shape)]
            if matches:
                report["checks"].append({"name": "shape", "result": "ok", "message": f"{len(matches)} images match expected shape {expected_shape}", "expected": expected_shape, "examples": matches[:5]})
            else:
                report["checks"].append({"name": "shape", "result": "warning", "message": f"No images match expected shape {expected_shape}. Resizing required.", "expected": expected_shape, "observed_shapes_sample": observed_shapes[:5]})

        observed_channels = set(analysis.get("channels") or [])
        if not observed_channels:
            report["checks"].append({"name": "channels", "result": "warning", "message": "No channel information available."})
        else:
            if observed_channels == {3}:
                report["checks"].append({"name": "channels", "result": "ok", "message": "RGB (3) observed."})
            elif observed_channels == {1}:
                report["checks"].append({"name": "channels", "result": "warning", "message": "Grayscale images observed. Model may expect 3-channel RGB."})
            elif 4 in observed_channels:
                report["checks"].append({"name": "channels", "result": "warning", "message": "Some images have an alpha channel (4). Consider removing alpha or converting to RGB."})
            else:
                report["checks"].append({"name": "channels", "result": "warning", "message": f"Observed channels: {sorted(list(observed_channels))}. Verify model expectations."})

        avg_ent = analysis.get("avg_entropy")
        min_ent = analysis.get("min_entropy")
        max_ent = analysis.get("max_entropy")
        if avg_ent is not None:
            report["checks"].append({"name": "entropy", "result": "ok", "avg_entropy": avg_ent, "min": min_ent, "max": max_ent})
            if min_ent is not None and min_ent < 1e-3:
                report["checks"].append({"name": "entropy_low", "result": "warning", "message": "Some images have extremely low entropy (nearly constant)."})
            if max_ent is not None and max_ent > 7.5:
                report["checks"].append({"name": "entropy_high", "result": "warning", "message": "Some images have very high entropy. Check noise/compression artifacts."})
        else:
            report["checks"].append({"name": "entropy", "result": "unknown", "message": "Entropy stats not available from analysis."})

        unreadable = analysis.get("unreadable_files") or []
        if unreadable:
            report["checks"].append({"name": "unreadable_files", "result": "warning", "message": f"{len(unreadable)} unreadable files found.", "examples": unreadable[:5]})

        # Determine overall status
        if any(c.get("result") == "error" for c in report["checks"]):
            report["status"] = "error"
        elif any(c.get("result") == "warning" for c in report["checks"]):
            report["status"] = "warning"
        else:
            report["status"] = "ok"

    result["compatibility_report"] = report

    # Step 3: recommendations
    recommendations, rec_err = None, None
    if analysis is not None:
        recommendations, rec_err = _safe_recommend(analysis, model, **kwargs)
    else:
        # best-effort: try to call recommender with minimal input if available
        if recommend_preprocessing is not None:
            try:
                recommendations, rec_err = recommend_preprocessing({}, model_name=model, **kwargs), None
            except Exception as e:
                recommendations, rec_err = None, {"error": "recommender_failed_on_minimal", "message": str(e)}

    if recommendations is None:
        recommendations = _fallback_recommendations_from_report(report, expected_shape=_infer_expected_shape_from_model(model))
        if rec_err:
            recommendations = {"note": "partial_recommendations", "detail": rec_err, "recommendations": recommendations}

    result["recommendations"] = recommendations
    return result


# Backwards-compatible alias used by older CLI and imports.
def check_model_compatibility(*args, **kwargs):
    """
    Backwards-compatible wrapper for older code that imports `check_model_compatibility`.
    Delegates to check_compatibility.
    """
    logger.warning("check_model_compatibility is deprecated; use check_compatibility instead.")
    return check_compatibility(*args, **kwargs)


# Public API
__all__ = ["check_compatibility", "check_model_compatibility"]
