# src/imgshape/cli.py
"""
imgshape CLI (v4.0.0 Atlas)
- Preserves v2/v3 CLI commands for backward compatibility
- Adds v4 Atlas commands: --atlas, --fingerprint, --decisions
- Adds `--web` to directly launch web UI (service/app.py)
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

# v4 Atlas modules
try:
    from imgshape.atlas import Atlas, analyze_dataset as analyze_dataset_v4
    from imgshape.atlas import fingerprint_only as fingerprint_dataset
    V4_AVAILABLE = True
except Exception:
    Atlas = None
    analyze_dataset_v4 = None
    fingerprint_dataset = None
    V4_AVAILABLE = False


# ----------------------------
# CLI argument parser
# ----------------------------
def cli_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="imgshape", description="imgshape CLI (v4.0.0 Atlas)")

    # Core arguments
    p.add_argument("--path", type=str, help="Path to image or dataset folder")
    p.add_argument("--url", type=str, help="Image URL to analyze")
    p.add_argument("--batch", action="store_true", help="Batch mode (operate on folder)")
    p.add_argument("--verbose", action="store_true", help="Verbose output")

    # Core actions (v2/v3 compatibility)
    p.add_argument("--analyze", action="store_true", help="Analyze image/dataset (v3)")
    p.add_argument("--shape", action="store_true", help="Get shape for image")
    p.add_argument("--shape-batch", action="store_true", help="Get shapes for a directory")
    p.add_argument("--recommend", action="store_true", help="Recommend preprocessing pipeline (v3)")
    p.add_argument("--augment", action="store_true", help="Include augmentation suggestions")

    # v4 Atlas actions
    p.add_argument("--atlas", action="store_true", help="Run full Atlas analysis (v4 fingerprint + decisions + artifacts)")
    p.add_argument("--fingerprint", action="store_true", help="Extract v4 dataset fingerprint only")
    p.add_argument("--decisions", action="store_true", help="Make decisions from fingerprint (requires --fingerprint-file)")
    p.add_argument("--fingerprint-file", type=str, help="Path to existing fingerprint.json (for --decisions)")
    
    # v4 intent parameters
    p.add_argument("--task", type=str, choices=["classification", "detection", "segmentation", "generation", "other"], 
                   default="classification", help="Task type: classification, detection, segmentation, generation, other")
    p.add_argument("--deployment", type=str, choices=["cloud", "edge", "mobile", "embedded", "other"], 
                   default="cloud", help="Deployment target: cloud, edge, mobile, embedded, other")
    p.add_argument("--priority", type=str, choices=["accuracy", "speed", "size", "balanced"], 
                   default="balanced", help="Optimization priority: accuracy, speed, size, balanced")
    p.add_argument("--max-model-size", type=int, help="Max model size in MB")
    
    # Deprecated aliases (for backward compatibility)
    p.add_argument("--intent", type=str, dest="_deprecated_intent", help=argparse.SUPPRESS)

    # Visualization and report
    p.add_argument("--viz", type=str, help="Plot dataset shape distribution")
    p.add_argument("--report", action="store_true", help="Generate Markdown/HTML report")
    p.add_argument("--out", type=str, help="Output file for JSON/report")

    # Direct web launch
    p.add_argument("--web", action="store_true", help="Launch FastAPI web service")

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
    
    # Handle deprecated --intent parameter
    if hasattr(args, '_deprecated_intent') and args._deprecated_intent:
        print("Warning: --intent is deprecated. Use --task instead.")
        print(f"    Example: --task {args._deprecated_intent}")
        print(f"    Available tasks: classification, detection, segmentation, generation, other")
        sys.exit(2)

    if args.verbose:
        print("Running imgshape CLI in verbose mode")

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

    # --- v4 Atlas commands ---
    if args.atlas and args.path and V4_AVAILABLE:
        print(f"\nRunning Atlas v4 analysis: {args.path}")
        try:
            from imgshape.decision_v4 import UserIntent, TaskType, DeploymentTarget, Priority
            
            intent = UserIntent(
                task=TaskType(args.task),
                deployment_target=DeploymentTarget(args.deployment),
                priority=Priority(args.priority),
            )
            
            result = analyze_dataset_v4(args.path, user_intent=intent)
            
            # Convert to dict for JSON serialization
            if isinstance(result, dict):
                artifacts_dict = {}
                if "artifacts" in result and isinstance(result["artifacts"], dict):
                    # Convert Path objects to strings
                    for key, value in result["artifacts"].items():
                        artifacts_dict[key] = str(value) if hasattr(value, '__fspath__') else value
                
                result_dict = {
                    "fingerprint": result["fingerprint"].to_dict() if hasattr(result["fingerprint"], "to_dict") else result["fingerprint"],
                    "decisions": result["decisions"].to_dict() if hasattr(result["decisions"], "to_dict") else result["decisions"],
                    "artifacts": artifacts_dict
                }
            else:
                result_dict = result.to_dict() if hasattr(result, 'to_dict') else result
            
            if args.out:
                out_path = Path(args.out)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(result_dict, indent=2), encoding="utf-8")
                print(f"Saved Atlas result to: {out_path}")
            else:
                print(json.dumps(result_dict, indent=2))
                
        except Exception as e:
            print(f"Atlas analysis failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
                
    elif args.atlas and not V4_AVAILABLE:
        print("❌ v4 Atlas not available. Check installation.")
        
    # v4 Fingerprint only
    if args.fingerprint and args.path and V4_AVAILABLE:
        print(f"\n👆 Extracting v4 fingerprint: {args.path}")
        try:
            result = fingerprint_dataset(args.path)
            result_dict = result.to_dict() if hasattr(result, 'to_dict') else result
            
            if args.out:
                out_path = Path(args.out)
                out_path.write_text(json.dumps(result_dict, indent=2), encoding="utf-8")
                print(f"✅ Saved fingerprint to: {out_path}")
            else:
                print(json.dumps(result_dict, indent=2))
                
        except Exception as e:
            print(f"❌ Fingerprint extraction failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
                
    elif args.fingerprint and not V4_AVAILABLE:
        print("❌ v4 fingerprint not available. Check installation.")

    # --- NEW: FastAPI Web UI launch ---
    if args.web:
        service_path = Path(__file__).resolve().parents[2] / "service" / "app.py"
        if not service_path.exists():
            print(f"❌ Could not find service/app.py at expected location: {service_path}")
            print("Run the web service manually with:")
            print("   uvicorn service.app:app --reload")
            sys.exit(1)

        print(f"Launching FastAPI service at: {service_path}")
        print("   Open browser at: http://127.0.0.1:8080")
        try:
            subprocess.run(
                ["uvicorn", "service.app:app", "--reload", "--host", "127.0.0.1", "--port", "8080"],
                check=True,
                cwd=str(service_path.parent.parent)
            )
        except FileNotFoundError:
            print("❌ uvicorn not installed. Install it via:")
            print("   pip install uvicorn[standard]")
        except subprocess.CalledProcessError as e:
            print(f"❌ Web service process failed: {e}")
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
