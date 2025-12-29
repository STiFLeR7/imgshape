from __future__ import annotations

# src/imgshape/__init__.py
"""
imgshape package public API (v2.2.0 -> v3.0.0 -> v4.0.0) — lazy exports + safe optional telemetry.

Behavior:
- Avoid importing heavy submodules at import-time.
- Provide the same top-level names as before via lazy import on attribute access.
- Telemetry (klyne) is initialized only when env var KLYNE_API_KEY is present.
- Safe: import-time failures are swallowed so users don't get ImportError when optional deps are missing.
- v3 additions: pipeline, recommender & plugin helpers exposed lazily.
- v4 additions: Atlas, fingerprinting, and decision engine exposed directly.
"""

import importlib
from importlib import metadata
import os
from typing import Any

# Public names we want to expose lazily
__all__ = [
    # shape / analyze
    "get_shape",
    "get_shape_batch",
    "analyze_type",
    "analyze_dataset",
    "plot_shape_distribution",
    # augmentations (v2)
    "AugmentationRecommender",
    "AugmentationPlan",
    # reports (v2)
    "generate_markdown_report",
    "generate_html_report",
    "generate_pdf_report",
    # torchloader (v2)
    "to_torch_transform",
    "to_dataloader",
    # recommender (v3)
    "RecommendEngine",
    "recommend_preprocessing",
    "recommend_dataset",
    # pipeline & plugins (v3)
    "RecommendationPipeline",
    "PipelineStep",
    "load_plugins_from_dir",
    "PluginBase",
    "AnalyzerPlugin",
    "RecommenderPlugin",
    "ExporterPlugin",
    # v4 - Atlas (eager import for better UX)
    "Atlas",
    "analyze_dataset_v4",
    "fingerprint_dataset",
    "FingerprintExtractor",
    "DecisionEngine",
    "UserIntent",
    "TaskType",
    "DeploymentTarget",
    "Priority",
    "__version__",
]

# --- version resolution (best-effort, non-fatal) ---
try:
    __version__ = metadata.version("imgshape")
except Exception:
    try:
        # fallback if you have a version.py with __version__
        from .version import __version__  # type: ignore
    except Exception:
        __version__ = "0.0.0"


# --- mapping attribute name -> (submodule, attr_name) for lazy import ---
_lazy_map = {
    # shape & analysis
    "get_shape": ("imgshape.shape", "get_shape"),
    "get_shape_batch": ("imgshape.shape", "get_shape_batch"),
    "analyze_type": ("imgshape.analyze", "analyze_type"),
    "analyze_dataset": ("imgshape.analyze", "analyze_dataset"),
    "plot_shape_distribution": ("imgshape.viz", "plot_shape_distribution"),
    # augmentations (v2)
    "AugmentationRecommender": ("imgshape.augmentations", "AugmentationRecommender"),
    "AugmentationPlan": ("imgshape.augmentations", "AugmentationPlan"),
    # reports (v2)
    "generate_markdown_report": ("imgshape.report", "generate_markdown_report"),
    "generate_html_report": ("imgshape.report", "generate_html_report"),
    "generate_pdf_report": ("imgshape.report", "generate_pdf_report"),
    # torchloader (v2)
    "to_torch_transform": ("imgshape.torchloader", "to_torch_transform"),
    "to_dataloader": ("imgshape.torchloader", "to_dataloader"),
    # recommender (v3)
    "RecommendEngine": ("imgshape.recommender", "RecommendEngine"),
    "recommend_preprocessing": ("imgshape.recommender", "recommend_preprocessing"),
    "recommend_dataset": ("imgshape.recommender", "recommend_dataset"),
    # pipeline & plugins (v3)
    "RecommendationPipeline": ("imgshape.pipeline", "RecommendationPipeline"),
    "PipelineStep": ("imgshape.pipeline", "PipelineStep"),
    "load_plugins_from_dir": ("imgshape.plugins", "load_plugins_from_dir"),
    "PluginBase": ("imgshape.plugins", "PluginBase"),
    "AnalyzerPlugin": ("imgshape.plugins", "AnalyzerPlugin"),
    "RecommenderPlugin": ("imgshape.plugins", "RecommenderPlugin"),
    "ExporterPlugin": ("imgshape.plugins", "ExporterPlugin"),
    # v4 - Atlas integration (updated to direct imports, no v4 subfolder)
    "Atlas": ("imgshape.atlas", "Atlas"),
    "analyze_dataset_v4": ("imgshape.atlas", "analyze_dataset"),
    "fingerprint_dataset": ("imgshape.atlas", "fingerprint_only"),
    "FingerprintExtractor": ("imgshape.fingerprint_v4", "FingerprintExtractor"),
    "DecisionEngine": ("imgshape.decision_v4", "DecisionEngine"),
    "UserIntent": ("imgshape.decision_v4", "UserIntent"),
    "TaskType": ("imgshape.decision_v4", "TaskType"),
    "DeploymentTarget": ("imgshape.decision_v4", "DeploymentTarget"),
    "Priority": ("imgshape.decision_v4", "Priority"),
}


# --- helpers for lazy import ---
def _lazy_import(name: str):
    """
    Import the module + attribute for a public name.
    Raises AttributeError if not found so __getattr__ can behave correctly.
    """
    if name not in _lazy_map:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    mod_name, attr = _lazy_map[name]
    try:
        module = importlib.import_module(mod_name)
        return getattr(module, attr)
    except Exception as exc:
        # Wrap and raise AttributeError for nicer UX when attribute is missing
        raise AttributeError(
            f"Could not import {attr!r} from {mod_name!r}. "
            "Optional dependency may be missing or import failed. "
            f"Original error: {exc}"
        ) from exc


def __getattr__(name: str) -> Any:
    """
    Module-level getattr that lazily imports attributes in _lazy_map.
    This is PEP 562 — supported on Python 3.7+.
    """
    return _lazy_import(name)


def __dir__() -> list[str]:
    # Provide nice tab-completion: show lazy names plus regular module attributes
    names = list(globals().keys()) + list(_lazy_map.keys())
    return sorted(set(names))


# --- Safe Klyne analytics init (non-fatal) ---
def _init_klyne() -> None:
    """
    Initialize klyne if enabled and key present. Absolutely non-fatal.
    """
    try:
        # opt-out toggle
        if os.getenv("ENABLE_ANALYTICS", "1").strip().lower() in ("0", "false", "no"):
            return

        api_key = os.getenv("KLYNE_API_KEY")
        if not api_key:
            # No key -> no telemetry.
            return

        # Import lazily and guard all errors
        try:
            import atexit
            import klyne  # optional dependency; may not be installed
        except Exception:
            return

        if hasattr(klyne, "init"):
            try:
                klyne.init(
                    api_key=api_key,
                    project="imgshape",
                    package_version=__version__,
                )
            except Exception:
                return

            if hasattr(klyne, "flush"):
                try:
                    atexit.register(lambda: klyne.flush(timeout=5.0))
                except Exception:
                    pass
    except Exception:
        # do not propagate any telemetry-related issues
        return


# Run telemetry init safely (non-fatal)
_init_klyne()
