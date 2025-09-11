# src/imgshape/cli.py
"""
imgshape CLI v2.1.0 (updated)

- Unified dataset recommendations
- Report generation with embedded plots (when possible)
- Torchloader / DataLoader helper with safe fallbacks
- Backwards-compatible with previous flags
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, List

# core modules
from imgshape.shape import get_shape, get_shape_batch
from imgshape.analyze import analyze_type
from imgshape.recommender import recommend_preprocessing
from imgshape.compatibility import check_model_compatibility

# viz: prefer the new convenience function
try:
    from imgshape.viz import plot_dataset_distribution, plot_shape_distribution
except Exception:
    plot_dataset_distribution = None
    plot_shape_distribution = None

from imgshape.gui import launch_gui

# optional/new modules (import guarded)
try:
    from imgshape.recommender import recommend_dataset
except Exception:
    recommend_dataset = None

try:
    from imgshape.augmentations import AugmentationRecommender, AugmentationPlan
except Exception:
    AugmentationRecommender = None
    AugmentationPlan = None

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
    """Convert dataclasses/objects to JSON-serializable structures where possible."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _read_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_read_jsonable(x) for x in obj]
    if hasattr(obj, "__dict__"):
        try:
            return _read_jsonable(vars(obj))
        except Exception:
            pass
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def _safe_analyze_dataset(path: str) -> Dict[str, Any]:
    """
    Attempt to call analyze_dataset (preferred). If unavailable, fall back to analyze_type for single-image stats.
    """
    try:
        from imgshape import analyze as _anmod

        if hasattr(_anmod, "analyze_dataset"):
            return _anmod.analyze_dataset(path)
    except Exception:
        pass

    try:
        info = analyze_type(path)
        return {
            "entropy_mean": info.get("entropy") if isinstance(info, dict) else None,
            "entropy_std": None,
            "colorfulness_mean": info.get("colorfulness") if isinstance(info, dict) else None,
            "shape_distribution": {},
            "edge_density": None,
            "class_balance": {},
            "channels": info.get("channels") if isinstance(info, dict) else None,
            "image_count": 1,
            "raw": info,
            "source_dir": path if Path(path).is_dir() else None,
        }
    except Exception:
        return {"image_count": 0, "source_dir": path if Path(path).is_dir() else None}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_read_jsonable(payload), indent=2), encoding="utf-8")


def cli_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="📦 imgshape — Image Shape, Analysis & Preprocessing Toolkit (v2.1.0)")
    p.add_argument("--path", type=str, help="Path to a single image or a directory")
    p.add_argument("--url", type=str, help="Image URL to analyze (single image)")
    p.add_argument("--batch", action="store_true", help="Operate on a directory / multiple images")
    p.add_argument("--seed", type=int, default=None, help="Seed for deterministic recommendations")

    p.add_argument("--analyze", action="store_true", help="Analyze image/dataset (stats)")
    p.add_argument("--shape", action="store_true", help="Get shape for a single image")
    p.add_argument("--shape-batch", action="store_true", help="Get shapes for multiple images in a directory")
    p.add_argument("--recommend", action="store_true", help="Recommend preprocessing for image/dataset")
    p.add_argument("--augment", action="store_true", help="Include augmentation recommendations with --recommend / --report / --torchloader")
    p.add_argument("--check", type=str, help="Check compatibility with a model (model name or config)")
    p.add_argument("--dir", type=str, help="Directory of images for compatibility check")

    p.add_argument("--viz", type=str, help="Plot dataset shape/size distribution (path)")
    p.add_argument("--web", action="store_true", help="Launch web GUI (Gradio)")
    p.add_argument("--report", action="store_true", help="Generate dataset report")
    p.add_argument(
        "--report-format",
        type=str,
        default="md",
        help="Comma-separated report format(s): md,html,pdf (pdf requires optional deps). Default: md",
    )
    p.add_argument("--out", type=str, default=None, help="Output path for JSON/report/script (depends on action)")

    p.add_argument("--torchloader", action="store_true", help="Generate torchvision transforms / DataLoader stub (requires optional torch)")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size for generated DataLoader stub")
    p.add_argument("--num-workers", type=int, default=4, help="num_workers for generated DataLoader stub")
    return p.parse_args()


def _is_dir_like(path: Optional[str]) -> bool:
    if not path:
        return False
    p = Path(path)
    return p.exists() and p.is_dir()


def main() -> None:
    args = cli_args()

    # ---------- shape (single) ----------
    if args.shape and args.path:
        print(f"\n📐 Shape for: {args.path}")
        try:
            print(get_shape(args.path))
        except Exception as e:
            print(f"❌ Error getting shape: {e}")

    # ---------- shape batch ----------
    if args.shape_batch and args.path:
        print(f"\n📐 Shapes for directory: {args.path}")
        try:
            # get_shape_batch accepts a directory or list; support both
            if _is_dir_like(args.path):
                results = get_shape_batch(args.path)
            else:
                results = get_shape_batch([args.path])
            print(json.dumps(_read_jsonable(results), indent=2))
        except Exception as e:
            print(f"❌ Error getting batch shapes: {e}")

    # ---------- analyze ----------
    if args.analyze and (args.path or args.url):
        target = args.path if args.path else args.url
        print(f"\n🔍 Analysis for: {target}")
        try:
            stats = _safe_analyze_dataset(target)
            print(json.dumps(_read_jsonable(stats), indent=2))
        except Exception as e:
            print(f"❌ Error analyzing: {e}")

    # ---------- recommend preprocessing (+ augment) ----------
    if args.recommend and args.path:
        print(f"\n🧠 Recommendation for: {args.path}")
        try:
            stats = _safe_analyze_dataset(args.path)
            # prefer unified recommend_dataset if available
            if recommend_dataset is not None:
                rec = recommend_dataset(stats, seed=args.seed)
            else:
                try:
                    rec = {"preprocessing": recommend_preprocessing(stats or args.path)}
                except Exception:
                    rec = {"preprocessing": recommend_preprocessing(args.path)}
            # optionally include augmentations if recommend_dataset is not available
            if args.augment and "augmentation_plan" not in rec:
                if AugmentationRecommender is None:
                    print("⚠️ AugmentationRecommender not available; skipping augmentation plan.")
                    rec["augmentation_plan"] = {}
                else:
                    ar = AugmentationRecommender(seed=args.seed)
                    plan = ar.recommend_for_dataset(stats or {})
                    rec["augmentation_plan"] = {
                        "order": plan.recommended_order,
                        "augmentations": [a.__dict__ for a in plan.augmentations],
                        "seed": plan.seed,
                    }

            if args.out:
                _write_json(Path(args.out), rec)
                print(f"📁 Wrote recommendations to {args.out}")
            else:
                print(json.dumps(_read_jsonable(rec), indent=2))

        except Exception as e:
            print(f"❌ Error generating recommendation: {e}")

    # ---------- model compatibility ----------
    if args.dir and args.check:
        print(f"\n✅ Model Compatibility Check — {args.check}")
        try:
            result = check_model_compatibility(args.dir, args.check)
            total = result.get("total", 0) if isinstance(result, dict) else 0
            passed = result.get("passed", 0) if isinstance(result, dict) else 0
            failed = result.get("failed", total - passed) if isinstance(result, dict) else 0
            print(f"🖼️ Total Images: {total}")
            print(f"✔️ Passed: {passed}")
            if failed:
                print(f"❌ Failed: {failed}")
            else:
                print("🎉 All images are compatible!")
            if result.get("issues"):
                print("\nIssues (first 10):")
                for fname, reason in result.get("issues", [])[:10]:
                    print(f"- {fname}: {reason}")
        except Exception as e:
            print(f"❌ Error checking model compatibility: {e}")

    # ---------- visualization ----------
    if args.viz:
        print(f"\n📊 Plotting dataset distribution for: {args.viz}")
        try:
            if plot_dataset_distribution is not None:
                # prefer new all-in-one function which saves plots
                plot_dataset_distribution(args.viz, save=False)
            elif plot_shape_distribution is not None:
                plot_shape_distribution(args.viz, save=False)
            else:
                print("⚠️ No plotting function available in imgshape.viz.")
        except Exception as e:
            print(f"❌ Error plotting: {e}")

    # ---------- report generation ----------
    if args.report and args.path:
        print(f"\n📄 Generating report for: {args.path}")
        fmt_list = [f.strip().lower() for f in args.report_format.split(",")] if args.report_format else ["md"]
        out_base = Path(args.out) if args.out else Path("imgshape_report")
        try:
            stats = _safe_analyze_dataset(args.path)
        except Exception:
            stats = {"image_count": 0, "source_dir": args.path if _is_dir_like(args.path) else None}

        try:
            # try unified recommendation
            if recommend_dataset is not None:
                rec = recommend_dataset(stats, seed=args.seed)
                preprocessing = rec.get("preprocessing", {})
                augmentation_plan = rec.get("augmentation_plan", {})
            else:
                preprocessing = recommend_preprocessing(stats or args.path)
                augmentation_plan = {}
                if args.augment and AugmentationRecommender is not None:
                    ar = AugmentationRecommender(seed=args.seed)
                    plan = ar.recommend_for_dataset(stats or {})
                    augmentation_plan = {
                        "order": plan.recommended_order,
                        "augmentations": [a.__dict__ for a in plan.augmentations],
                        "seed": plan.seed,
                    }
        except Exception:
            preprocessing = {}
            augmentation_plan = {}

        # choose plots_from if path is directory
        plots_from = args.path if _is_dir_like(args.path) else None

        md_path = None
        if "md" in fmt_list:
            md_out = out_base if out_base.suffix == ".md" else out_base.with_suffix(".md")
            if generate_markdown_report is None:
                print("⚠️ report.generate_markdown_report not available. Install/implement imgshape.report to enable reports.")
            else:
                try:
                    md_path = generate_markdown_report(
                        md_out,
                        stats,
                        compatibility={},
                        preprocessing=preprocessing,
                        augmentation_plan=augmentation_plan or {},
                        plots_from=plots_from,
                    )
                    print(f"📝 Markdown report written to {md_path}")
                except Exception as e:
                    print(f"❌ Error generating markdown report: {e}")
                    md_path = None

        if "html" in fmt_list:
            if generate_html_report is None:
                print("⚠️ HTML report converter not available.")
            else:
                try:
                    if md_path is None:
                        tmp_md = out_base.with_suffix(".md")
                        generate_markdown_report(tmp_md, stats, compatibility={}, preprocessing=preprocessing, augmentation_plan=augmentation_plan or {}, plots_from=plots_from)
                        md_path = tmp_md
                    html_path = out_base.with_suffix(".html")
                    generate_html_report(md_path, html_path)
                    print(f"🌐 HTML report written to {html_path}")
                except Exception as e:
                    print(f"❌ Error generating HTML report: {e}")

        if "pdf" in fmt_list:
            if generate_pdf_report is None:
                print("⚠️ PDF report generator not available. Install 'weasyprint' to enable PDF output.")
            else:
                try:
                    if md_path is None:
                        tmp_md = out_base.with_suffix(".md")
                        generate_markdown_report(tmp_md, stats, compatibility={}, preprocessing=preprocessing, augmentation_plan=augmentation_plan or {}, plots_from=plots_from)
                        md_path = tmp_md
                    html_path = out_base.with_suffix(".html")
                    if not html_path.exists() and generate_html_report is not None:
                        generate_html_report(md_path, html_path)
                    pdf_out = out_base.with_suffix(".pdf")
                    generate_pdf_report(html_path, pdf_out)
                    print(f"📕 PDF report written to {pdf_out}")
                except Exception as e:
                    print(f"❌ Error generating PDF report: {e}")

    # ---------- torchloader / dataloader helper ----------
    if args.torchloader and args.path:
        print(f"\n🔗 Generating Torch DataLoader/Transform helper for: {args.path}")
        try:
            stats = _safe_analyze_dataset(args.path)
        except Exception:
            stats = {"image_count": 0}
        try:
            if recommend_dataset is not None:
                rec = recommend_dataset(stats, seed=args.seed)
                preprocessing = rec.get("preprocessing", {})
                augmentation_plan = rec.get("augmentation_plan", {})
            else:
                preprocessing = recommend_preprocessing(stats or args.path)
                augmentation_plan = {}
                if args.augment and AugmentationRecommender is not None:
                    ar = AugmentationRecommender(seed=args.seed)
                    plan = ar.recommend_for_dataset(stats or {})
                    augmentation_plan = {
                        "order": plan.recommended_order,
                        "augmentations": [a.__dict__ for a in plan.augmentations],
                        "seed": plan.seed,
                    }
        except Exception:
            preprocessing = {}
            augmentation_plan = {}

        # Try to create an actual DataLoader if available
        created_dl = False
        if to_dataloader is not None:
            try:
                dl = to_dataloader(
                    dataset_paths=[args.path],
                    labels=None,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    augmentation_plan=augmentation_plan,
                    preprocessing=preprocessing,
                    pin_memory=False,
                )
                created_dl = True
                print("✅ DataLoader object created (returned). Use programmatically in your training script.")
                if args.out:
                    Path(args.out).write_text("# DataLoader created programmatically. Use to_dataloader(...) in your code.\n")
                    print(f"📁 Wrote stub/notes to {args.out}")
            except ImportError:
                print("⚠️ torch not installed. to_dataloader requires torch. Falling back to transform snippet.")
            except Exception as e:
                print(f"❌ Error creating DataLoader: {e}")

        # Fallback to transform snippet or Compose
        if not created_dl:
            if to_torch_transform is None:
                print("⚠️ to_torch_transform not available. Install imgshape.torchloader or the torch optional extra.")
            else:
                try:
                    transforms_obj_or_snippet = to_torch_transform(augmentation_plan or {}, preprocessing or {})
                    if isinstance(transforms_obj_or_snippet, str):
                        if args.out:
                            Path(args.out).write_text(transforms_obj_or_snippet, encoding="utf-8")
                            print(f"🧾 Wrote transform snippet to {args.out}")
                        else:
                            print("\n=== Transform snippet ===\n")
                            print(transforms_obj_or_snippet)
                    else:
                        # Compose object created
                        print("✅ torchvision.transforms.Compose returned. Use it programmatically.")
                        if args.out:
                            Path(args.out).write_text("# Compose object created programmatically; import and use in your code.", encoding="utf-8")
                            print(f"📁 Wrote stub to {args.out}")
                except ImportError:
                    print("⚠️ torch not installed; cannot build real torchvision transforms.")
                except Exception as e:
                    print(f"❌ Error building transform snippet: {e}")

    # ---------- web GUI ----------
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
