# src/imgshape/__init__.py
"""
imgshape package public API + optional (safe) Klyne telemetry bootstrap.

Behavior:
- Telemetry is initialized only when the env var KLYNE_API_KEY is set.
- Set ENABLE_ANALYTICS=0 or ENABLE_ANALYTICS=false to disable telemetry.
- The init is wrapped in try/except so imports never fail for users.
"""

from importlib import metadata
import os

# --- Public API exports (keep as you had them) ---
from .augmentations import AugmentationRecommender, AugmentationPlan
from .report import generate_markdown_report, generate_html_report, generate_pdf_report
from .torchloader import to_torch_transform, to_dataloader

__all__ = [
    "AugmentationRecommender",
    "AugmentationPlan",
    "generate_markdown_report",
    "generate_html_report",
    "generate_pdf_report",
    "to_torch_transform",
    "to_dataloader",
]

# --- package version detection (preferred: metadata from installed package) ---
try:
    __version__ = metadata.version("imgshape")
except Exception:
    # fallback for dev/editable installs: optional single-source version file
    try:
        from .version import __version__  # create if you don't have it
    except Exception:
        __version__ = "0.0.0"

# --- Safe Klyne analytics init ---
def _init_klyne():
    """Initialize klyne if enabled and key present. Absolutely non-fatal."""
    try:
        # opt-out toggle
        if os.getenv("ENABLE_ANALYTICS", "1").strip().lower() in ("0", "false", "no"):
            return

        api_key = os.getenv("KLYNE_API_KEY")
        if not api_key:
            # No key -> no telemetry. Keeps package safe and private by default.
            return

        import atexit
        import klyne

        if hasattr(klyne, "init"):
            try:
                klyne.init(
                    api_key=api_key,
                    project="imgshape",            # must exactly match PyPI name
                    package_version=__version__,
                )
            except Exception:
                # don't crash on any SDK internals
                return

            # register flush for short-lived scripts
            if hasattr(klyne, "flush"):
                try:
                    atexit.register(lambda: klyne.flush(timeout=5.0))
                except Exception:
                    # ignore atexit failures
                    pass
    except Exception:
        # never bubble any analytics error to importers
        return

# run it (safe)
_init_klyne()
