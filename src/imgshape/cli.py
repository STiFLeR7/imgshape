# src/imgshape/cli.py
"""
imgshape CLI (v3.0.0 compat)
- Preserves v2 CLI commands
- Adds `--web` to directly launch Streamlit UI (root-level app.py)
- Removes deprecated gui.py dependency
"""

from __future__ import annotations
import argparse
import json
import sys
import shutil
import datetime
from pathlib import Path
from typing import Any, Dict
import subprocess
import os

# Core imports (safe)
from imgshape.shape import get_shape, get_shape_batch
from imgshape.analyze import analyze_type, analyze_dataset
from imgshape.recommender import recommend_preprocessing, recommend_dataset
from imgshape.compatibility import check_model_compatibility
from imgshape.viz import plot_shape_distribution

# Optional v3 modules
try:
    from imgshape.pipeline import RecommendationPipeline, PipelineStep
except Exception:
    RecommendationPipeline = None
    PipelineStep = None

try:
    from imgshape.plugins import load_plugins_from_dir
except Exception:
    load_plugins_from_dir = None


# ----------------------------
# CLI argument parser
# ----------------------------
def cli_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="imgshape", description="imgshape CLI (v3)")

    # Core arguments
    p.add_argument("--path", type=str, help="Path to image or dataset folder")
    p.add_argument("--url", type=str, help="Image URL to analyze")
    p.add_argument("--batch", action="store_true", help="Batch mode (operate on folder)")
    p.add_argument("--verbose", action="store_true", help="Verbose output")

    # Core actions
    p.add_argument("--analyze", action="store_true", help="Analyze image/dataset")
    p.add_argument("--shape", action="store_true", help="Get shape for image")
    p.add_argument("--shape-batch", action="store_true", help="Get shapes for a directory")
    p.add_argument("--recommend", action="store_true", help="Recommend preprocessing pipeline")
    p.add_argument("--augment", action="store_true", help="Include augmentation suggestions")

    # Visualization and report
    p.add_argument("--viz", type=str, help="Plot dataset shape distribution")
    p.add_argument("--report", action="store_true", help="Generate Markdown/HTML report")
    p.add_argument("--out", type=str, help="Output file for JSON/report")

    # New: direct web launch (Streamlit)
    p.add_argument("--web", action="store_true", help="Launch Streamlit web app (root app.py)")

    # v3 additions (pipeline, plugins, etc.)
    p.add_argument("--pipeline-export", action="store_true", help="Export recommended pipeline as code/json/yaml")
    p.add_argument("--pipeline-format", type=str, default="torchvision", help="Export format")
    p.add_argument("--plugin-list", action="store_true", help="List detected plugins")

    return p.parse_args()


# ----------------------------
# CLI Main
# ----------------------------
def main() -> None:
    args = cli_args()

    if args.verbose:
        print("🔍 Running imgshape CLI in verbose mode")

    # Single image shape
    if args.shape and args.path:
        print(f"\n📐 Shape for: {args.path}")
        try:
            print(get_shape(args.path))
        except Exception as e:
            print(f"❌ Error: {e}")

    # Batch shapes
    if args.shape_batch and args.path:
        print(f"\n📦 Batch shapes for: {args.path}")
        try:
            result = get_shape_batch(args.path)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"❌ Error: {e}")

    # Analyze (auto detect single vs dataset)
    if args.analyze and (args.path or args.url):
        target = args.path or args.url
        print(f"\n🔍 Analyzing: {target}")
        try:
            if args.batch or Path(target).is_dir():
                stats = analyze_dataset(target)
            else:
                stats = analyze_type(target)
            print(json.dumps(stats, indent=2))
        except Exception as e:
            print(f"❌ Analysis failed: {e}")

    # Recommend preprocessing
    if args.recommend and args.path:
        print(f"\n🧠 Recommending preprocessing for: {args.path}")
        try:
            if args.batch or Path(args.path).is_dir():
                rec = recommend_dataset(args.path)
            else:
                rec = recommend_preprocessing(args.path)
            print(json.dumps(rec, indent=2))
        except Exception as e:
            print(f"❌ Recommendation failed: {e}")

    # Visualization
    if args.viz:
        print(f"\n📊 Generating visualization for: {args.viz}")
        try:
            fig = plot_shape_distribution(args.viz, save=True)
            if hasattr(fig, "write_html"):
                out_html = Path(args.viz) / "shape_distribution.html"
                fig.write_html(str(out_html))
                print(f"✅ Saved interactive plot to {out_html}")
        except Exception as e:
            print(f"❌ Visualization failed: {e}")

    # --- NEW: Streamlit Web UI launch ---
    if args.web:
        app_path = Path(__file__).resolve().parents[2] / "app.py"
        if not app_path.exists():
            print(f"❌ Could not find app.py at expected location: {app_path}")
            print("Place your Streamlit entrypoint at the repository root as app.py, or run manually:")
            print("   streamlit run app.py")
            sys.exit(1)

        print(f"🚀 Launching Streamlit app at: {app_path}")
        try:
            subprocess.run(
                [sys.executable, "-m", "streamlit", "run", str(app_path)],
                check=True,
            )
        except FileNotFoundError:
            print("❌ Streamlit not installed. Install it via:")
            print("   pip install streamlit")
        except subprocess.CalledProcessError as e:
            print(f"❌ Streamlit process failed: {e}")
        except KeyboardInterrupt:
            print("\n🛑 Web UI stopped by user.")

    # Plugins
    if args.plugin_list:
        plugins_dir = Path(__file__).parent / "plugins"
        print(f"\n🔌 Plugins at {plugins_dir}:")
        if load_plugins_from_dir:
            try:
                plugins = load_plugins_from_dir(str(plugins_dir))
                for p in plugins:
                    name = getattr(p, "NAME", p.__class__.__name__)
                    print(f"- {name}")
            except Exception as e:
                print(f"❌ Failed to load plugins: {e}")
        else:
            print("⚠️ Plugin system not available.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — exiting.")
        sys.exit(1)
