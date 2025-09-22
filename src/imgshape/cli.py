# src/imgshape/cli.py
"""
imgshape CLI v2.2.0 — command-line interface.
Robust: builds reports in-CLI to avoid brittle report.py signatures.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import datetime

from imgshape.shape import get_shape, get_shape_batch
from imgshape.analyze import analyze_type
from imgshape.recommender import recommend_preprocessing, recommend_dataset
from imgshape.compatibility import check_model_compatibility
from imgshape.viz import plot_shape_distribution
from imgshape.gui import launch_gui

# optional imports
try:
    from imgshape.augmentations import AugmentationRecommender, AugmentationPlan
except Exception:
    AugmentationRecommender = None
    AugmentationPlan = None

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
        # dataclass-like / as_dict support
        if hasattr(obj, "as_dict") and callable(getattr(obj, "as_dict")):
            try:
                return _read_jsonable(obj.as_dict())
            except Exception:
                pass
        if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
            try:
                return _read_jsonable(obj.to_dict())
            except Exception:
                pass
        if hasattr(obj, "__dict__"):
            try:
                return _read_jsonable(vars(obj))
            except Exception:
                pass
        # fallback to str
        return str(obj)


def _write_text_file(path: Path, content: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return str(path)


def _normalize_augmentation_plan(ap: Any) -> Dict[str, Any]:
    """Turn several augmentation-plan shapes into a plain serializable dict."""
    if ap is None:
        return {}
    # If it's already a dict
    if isinstance(ap, dict):
        return _read_jsonable(ap)
    # dataclass-like with as_dict
    if hasattr(ap, "as_dict") and callable(getattr(ap, "as_dict")):
        try:
            return _read_jsonable(ap.as_dict())
        except Exception:
            pass
    # dataclass-like standard __dict__ / list of objects
    try:
        if hasattr(ap, "__dict__"):
            return _read_jsonable(vars(ap))
    except Exception:
        pass
    # list of augmentations?
    if isinstance(ap, (list, tuple)):
        return {"augmentations": _read_jsonable(ap)}
    # fallback to string
    return {"note": str(ap)}


def _emit_markdown_report(dataset_path: str, ds_rec: Dict[str, Any], compatibility: Dict[str, Any], augmentation_plan: Any) -> str:
    out_lines = []
    out_lines.append(f"# 📊 imgshape Report")
    out_lines.append(f"- Generated: {datetime.datetime.utcnow().isoformat()}Z")
    out_lines.append(f"- Dataset: `{dataset_path}`")
    out_lines.append("")
    out_lines.append("## Dataset Summary")
    ds_summary = ds_rec.get("dataset_summary", {})
    out_lines.append("```json")
    out_lines.append(json.dumps(_read_jsonable(ds_summary), indent=2))
    out_lines.append("```")
    out_lines.append("")
    if compatibility:
        out_lines.append("## Compatibility")
        out_lines.append("```json")
        out_lines.append(json.dumps(_read_jsonable(compatibility), indent=2))
        out_lines.append("```")
        out_lines.append("")
    out_lines.append("## Representative Preprocessing")
    out_lines.append("```json")
    out_lines.append(json.dumps(_read_jsonable(ds_rec.get("representative_preprocessing", {})), indent=2))
    out_lines.append("```")
    out_lines.append("")
    if augmentation_plan:
        out_lines.append("## Augmentation Plan")
        ap_serial = _normalize_augmentation_plan(augmentation_plan)
        out_lines.append("```json")
        out_lines.append(json.dumps(ap_serial, indent=2))
        out_lines.append("```")
    return "\n".join(out_lines)


def _emit_html_report(dataset_path: str, ds_rec: Dict[str, Any], compatibility: Dict[str, Any], augmentation_plan: Any) -> str:
    parts = []
    parts.append(f"<h1>📊 imgshape Report</h1>")
    parts.append(f"<p><b>Generated:</b> {datetime.datetime.utcnow().isoformat()}Z</p>")
    parts.append(f"<p><b>Dataset:</b> <code>{dataset_path}</code></p>")
    parts.append("<h2>Dataset Summary</h2>")
    parts.append(f"<pre>{json.dumps(_read_jsonable(ds_rec.get('dataset_summary', {})), indent=2)}</pre>")
    if compatibility:
        parts.append("<h2>Compatibility</h2>")
        parts.append(f"<pre>{json.dumps(_read_jsonable(compatibility), indent=2)}</pre>")
    parts.append("<h2>Representative Preprocessing</h2>")
    parts.append(f"<pre>{json.dumps(_read_jsonable(ds_rec.get('representative_preprocessing', {})), indent=2)}</pre>")
    if augmentation_plan:
        try:
            ap_serial = _normalize_augmentation_plan(augmentation_plan)
            parts.append("<h2>Augmentation Plan</h2>")
            parts.append(f"<pre>{json.dumps(ap_serial, indent=2)}</pre>")
        except Exception:
            parts.append("<p><code>&lt;could not serialize augmentation plan&gt;</code></p>")
    return "<html><body>" + "\n".join(parts) + "</body></html>"


def cli_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="📦 imgshape CLI v2.2.0")

    p.add_argument("--path", type=str, help="Path to a single image or a directory")
    p.add_argument("--url", type=str, help="Image URL to analyze (single image)")
    p.add_argument("--batch", action="store_true", help="Operate on a directory / multiple images")
    p.add_argument("--seed", type=int, default=None, help="Seed for deterministic recommendations")
    p.add_argument("--verbose", action="store_true", help="Enable verbose/logging output for CLI actions")

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
    p.add_argument("--report-format", type=str, default="md", help="Comma-separated formats: md,html,pdf")
    p.add_argument("--out", type=str, default=None, help="Output path for JSON/report/script")

    p.add_argument("--torchloader", action="store_true", help="Generate torchvision transforms / DataLoader stub")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size for DataLoader stub")
    p.add_argument("--num-workers", type=int, default=4, help="num_workers for DataLoader stub")

    return p.parse_args()


def main() -> None:
    args = cli_args()

    if args.verbose:
        print("🔎 CLI running in verbose mode")

    # shape (single)
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
            from imgshape import analyze as _anmod
            if hasattr(_anmod, "analyze_dataset") and args.batch:
                stats = _anmod.analyze_dataset(target)
            else:
                stats = _anmod.analyze_dataset(target) if hasattr(_anmod, "analyze_dataset") and args.batch else analyze_type(target)
            print(json.dumps(_read_jsonable(stats), indent=2))
        except Exception as e:
            print(f"❌ Error analyzing: {e}")

    # recommend
    if args.recommend and args.path:
        print(f"\n🧠 Recommendation for: {args.path}")
        try:
            if args.batch or Path(args.path).is_dir():
                result = recommend_dataset(args.path)
                out_payload = {"dataset_recommendation": result}
            else:
                out_payload = {"preprocessing": recommend_preprocessing(args.path)}
                if args.augment and AugmentationRecommender is not None:
                    ar = AugmentationRecommender(seed=args.seed)
                    plan = ar.recommend_for_dataset(
                        {"entropy_mean": out_payload["preprocessing"].get("entropy"), "image_count": 1}
                    )
                    out_payload["augmentation_plan"] = _normalize_augmentation_plan(plan)
            if args.out:
                _write_text_file(Path(args.out), json.dumps(_read_jsonable(out_payload), indent=2))
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
            print(json.dumps(_read_jsonable(result), indent=2))
        except Exception as e:
            print(f"❌ Error checking compatibility: {e}")

    # viz
    if args.viz:
        print(f"\n📊 Plotting shape distribution for: {args.viz}")
        try:
            out_file = plot_shape_distribution(args.viz)
            print(f"✅ Saved plot to {out_file}")
        except Exception as e:
            print(f"❌ Error plotting: {e}")

    # report (now implemented directly to avoid report.py signature issues)
    if args.report:
        dataset_path = args.dir or args.path
        if not dataset_path:
            print("❌ --report requires --dir <dataset_dir> or --path <dataset_path>")
        else:
            try:
                # build dataset summary and preprocessing
                ds_rec = recommend_dataset(dataset_path)
                preprocessing = ds_rec.get("representative_preprocessing", ds_rec)
                compatibility = {}
                if args.check:
                    try:
                        compatibility = check_model_compatibility(dataset_path, args.check)
                    except Exception as e:
                        print(f"⚠️ Compatibility check failed: {e}")
                        compatibility = {}

                augmentation_plan = {}
                if AugmentationRecommender is not None:
                    try:
                        ar = AugmentationRecommender(seed=args.seed)
                        augmentation_plan = ar.recommend_for_dataset(
                            {"entropy_mean": preprocessing.get("entropy"), "image_count": preprocessing.get("image_count", 1)}
                        )
                    except Exception as e:
                        print(f"⚠️ Augmentation generation failed: {e}")
                        augmentation_plan = {}

                formats = [f.strip().lower() for f in (args.report_format or "md").split(",") if f.strip()]
                out_base = args.out or "imgshape_report"

                for fmt in formats:
                    base = Path(out_base)
                    # produce proper filename with extension (don't treat directory as file)
                    if base.exists() and base.is_dir():
                        base = base / "imgshape_report"
                    if base.suffix.lower() == f".{fmt}":
                        out_path = str(base)
                    else:
                        out_path = str(base.with_suffix(f".{fmt}"))

                    if fmt == "md":
                        md = _emit_markdown_report(dataset_path, ds_rec, compatibility, augmentation_plan)
                        _write_text_file(Path(out_path), md)
                        print(f"✅ Markdown report written: {out_path}")
                    elif fmt == "html":
                        html = _emit_html_report(dataset_path, ds_rec, compatibility, augmentation_plan)
                        _write_text_file(Path(out_path), html)
                        print(f"✅ HTML report written: {out_path}")
                    elif fmt == "pdf":
                        # try to generate a PDF (requires reportlab)
                        try:
                            from reportlab.lib.pagesizes import letter
                            from reportlab.pdfgen import canvas
                        except Exception:
                            # fallback to markdown
                            md = _emit_markdown_report(dataset_path, ds_rec, compatibility, augmentation_plan)
                            fallback_path = str(Path(out_path).with_suffix(".md"))
                            _write_text_file(Path(fallback_path), md)
                            print(f"⚠️ reportlab not available — wrote markdown fallback: {fallback_path}")
                        else:
                            # produce a simple PDF using reportlab
                            c = canvas.Canvas(out_path, pagesize=letter)
                            c.setFont("Helvetica", 12)
                            y = 750
                            title = f"imgshape Report — {dataset_path}"
                            c.drawString(50, y, title)
                            y -= 24
                            # write sections
                            sections = [
                                ("Dataset Summary", ds_rec.get("dataset_summary", {})),
                                ("Compatibility", compatibility),
                                ("Representative Preprocessing", ds_rec.get("representative_preprocessing", {})),
                                ("Augmentation Plan", _normalize_augmentation_plan(augmentation_plan)),
                            ]
                            for section_name, data in sections:
                                if not data:
                                    continue
                                c.setFont("Helvetica-Bold", 10)
                                c.drawString(50, y, section_name)
                                y -= 16
                                c.setFont("Courier", 8)
                                for line in json.dumps(_read_jsonable(data), indent=2).splitlines():
                                    c.drawString(50, y, line)
                                    y -= 10
                                    if y < 50:
                                        c.showPage()
                                        y = 750
                                y -= 10
                            c.save()
                            print(f"✅ PDF report written: {out_path}")
                    else:
                        print(f"⚠️ Unsupported report format: {fmt}")
            except Exception as e:
                print(f"❌ Error generating report: {e}")

    # torchloader
    if args.torchloader and args.path:
        print(f"\n🔗 Generating Torch DataLoader/Transform helper for: {args.path}")
        try:
            preprocessing = (
                recommend_dataset(args.path) if args.batch or Path(args.path).is_dir() else recommend_preprocessing(args.path)
            )
            snippet_or_transform = to_torch_transform({}, preprocessing or {})
            if isinstance(snippet_or_transform, str):
                if args.out:
                    Path(args.out).write_text(snippet_or_transform)
                    print(f"🧾 Wrote transform snippet to {args.out}")
                else:
                    print("\n=== Transform snippet ===\n")
                    print(snippet_or_transform)
            else:
                print("✅ Transform object created.")
        except Exception as e:
            print(f"❌ Error building transform snippet: {e}")

    # web
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
