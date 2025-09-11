# src/imgshape/cli.py
"""
imgshape CLI v2.1.x — CLI that uses recommend_dataset for directories and recommend_preprocessing for single images.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from imgshape.shape import get_shape, get_shape_batch
from imgshape.analyze import analyze_type
from imgshape.recommender import recommend_preprocessing, recommend_dataset
from imgshape.compatibility import check_model_compatibility
from imgshape.viz import plot_shape_distribution
from imgshape.gui import launch_gui

# optional imports
try:
    from imgshape.augmentations import AugmentationRecommender
except Exception:
    AugmentationRecommender = None

try:
    from imgshape.report import (
        generate_markdown_report,
        generate_html_report,
        generate_pdf_report,
    )
except Exception:
    generate_markdown_report = None
    generate_html_report = None
    generate_pdf_report = None

try:
    from imgshape.torchloader import to_torch_transform, to_dataloader
except Exception:
    to_torch_transform = None
    to_dataloader = None


def _read_jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _read_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_read_jsonable(x) for x in obj]
    try:
        json.dumps(obj)
        return obj
    except Exception:
        if hasattr(obj, "__dict__"):
            try:
                return _read_jsonable(vars(obj))
            except Exception:
                pass
        return str(obj)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_read_jsonable(payload), indent=2))


def cli_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="📦 imgshape — Image Shape, Analysis & Preprocessing Toolkit (v2.1.x)"
    )

    p.add_argument("--path", type=str, help="Path to a single image or a directory")
    p.add_argument("--url", type=str, help="Image URL to analyze (single image)")
    p.add_argument("--batch", action="store_true", help="Operate on a directory / multiple images")
    p.add_argument("--seed", type=int, default=None, help="Seed for deterministic recommendations")

    p.add_argument("--analyze", action="store_true", help="Analyze image/dataset (stats)")
    p.add_argument("--shape", action="store_true", help="Get shape for a single image")
    p.add_argument("--shape-batch", action="store_true", help="Get shapes for multiple images in a directory")
    p.add_argument("--recommend", action="store_true", help="Recommend preprocessing for image/dataset")
    p.add_argument("--augment", action="store_true", help="Include augmentation recommendations with --recommend")

    p.add_argument("--check", type=str, help="Check compatibility with a model (model name or config)")
    p.add_argument("--dir", type=str, help="Directory of images for compatibility check")

    p.add_argument("--viz", type=str, help="Plot dataset shape/size distribution (path)")
    p.add_argument("--web", action="store_true", help="Launch web GUI (Gradio)")
    p.add_argument("--report", action="store_true", help="Generate dataset report")
    p.add_argument("--report-format", type=str, default="md", help="Comma-separated report format(s): md,html,pdf")
    p.add_argument("--out", type=str, default=None, help="Output path for JSON/report/script (depends on action)")

    p.add_argument("--torchloader", action="store_true", help="Generate torchvision transforms / DataLoader stub")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size for generated DataLoader stub")
    p.add_argument("--num-workers", type=int, default=4, help="num_workers for generated DataLoader stub")

    return p.parse_args()


def main() -> None:
    args = cli_args()

    # single image shape
    if args.shape and args.path:
        print(f"\n📐 Shape for: {args.path}")
        try:
            print(get_shape(args.path))
        except Exception as e:
            print(f"❌ Error getting shape: {e}")

    # shape batch
    if args.shape_batch and args.path:
        print(f"\n📐 Shapes for directory: {args.path}")
        try:
            results = get_shape_batch(args.path)
            print(json.dumps(results, indent=2))
        except Exception as e:
            print(f"❌ Error getting batch shapes: {e}")

    # analyze
    if args.analyze and (args.path or args.url):
        target = args.path if args.path else args.url
        print(f"\n🔍 Analysis for: {target}")
        try:
            # prefer dataset analyzer if present
            from imgshape import analyze as _anmod  # dynamic
            if hasattr(_anmod, "analyze_dataset") and args.batch:
                stats = _anmod.analyze_dataset(target)
            else:
                stats = _anmod.analyze_dataset(target) if hasattr(_anmod, "analyze_dataset") else analyze_type(target)
            print(json.dumps(_read_jsonable(stats), indent=2))
        except Exception as e:
            print(f"❌ Error analyzing: {e}")

    # recommend preprocessing
    if args.recommend and args.path:
        print(f"\n🧠 Recommendation for: {args.path}")
        try:
            # if directory / batch -> use recommend_dataset
            if args.batch or Path(args.path).is_dir():
                result = recommend_dataset(args.path)
                out_payload = {"dataset_recommendation": result}
            else:
                # single image: try to open and pass PIL.Image or path
                out_payload = {"preprocessing": recommend_preprocessing(args.path)}
                if args.augment and AugmentationRecommender is not None:
                    ar = AugmentationRecommender(seed=args.seed)
                    plan = ar.recommend_for_dataset({"entropy_mean": out_payload["preprocessing"].get("entropy"), "image_count":1})
                    out_payload["augmentation_plan"] = {
                        "order": plan.recommended_order,
                        "augmentations": [a.__dict__ for a in plan.augmentations],
                        "seed": plan.seed,
                    }

            if args.out:
                _write_json(Path(args.out), out_payload)
                print(f"📁 Wrote recommendations to {args.out}")
            else:
                print(json.dumps(_read_jsonable(out_payload), indent=2))

        except Exception as e:
            print(f"❌ Error generating recommendation: {e}")

    # compatibility
    if args.dir and args.check:
        print(f"\n✅ Model Compatibility Check — {args.check}")
        try:
            result = check_model_compatibility(args.dir, args.check)
            if isinstance(result, dict):
                total = result.get("total", 0)
                passed = result.get("passed", 0)
                issues = result.get("issues", [])
            else:
                try:
                    passed, failed = result
                    total = passed + failed
                    issues = []
                except Exception:
                    total = passed = 0
                    issues = []
            print(f"🖼️ Total Images: {total}")
            print(f"✔️ Passed: {passed}")
            if issues:
                print(f"❌ Issues: {len(issues)}")
            else:
                print("🎉 All images are compatible!")
        except Exception as e:
            print(f"❌ Error checking model compatibility: {e}")

    # viz
    if args.viz:
        print(f"\n📊 Plotting shape distribution for: {args.viz}")
        try:
            plot_shape_distribution(args.viz)
        except Exception as e:
            print(f"❌ Error plotting: {e}")

    # torchloader
    if args.torchloader and args.path:
        print(f"\n🔗 Generating Torch DataLoader/Transform helper for: {args.path}")
        try:
            preprocessing = recommend_preprocessing(args.path) if not (args.batch or Path(args.path).is_dir()) else recommend_dataset(args.path)
        except Exception:
            preprocessing = {}
        try:
            snippet_or_transform = to_torch_transform({}, preprocessing or {})
            if isinstance(snippet_or_transform, str):
                if args.out:
                    Path(args.out).write_text(snippet_or_transform)
                    print(f"🧾 Wrote transform snippet to {args.out}")
                else:
                    print("\n=== Transform snippet ===\n")
                    print(snippet_or_transform)
            else:
                print("✅ Transform object created (use programmatically).")
                if args.out:
                    Path(args.out).write_text("# Use the to_torch_transform in code to create transforms\n")
                    print(f"📁 Wrote stub to {args.out}")
        except ImportError:
            print("⚠️ torch is not available in this environment.")
        except Exception as e:
            print(f"❌ Error building transform snippet: {e}")

    # web gui
    if args.web:
        print("\n🚀 Launching imgshape Web GUI...")
        try:
            launch_gui()
        except Exception as e:
            print(f"❌ Error launching GUI: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — exiting.")
        sys.exit(1)
