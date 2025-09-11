# src/imgshape/gui.py
"""
Gradio GUI for imgshape v2.1.0+

Tabs:
 - Analyze: single-image shape + stats
 - Recommend: preprocessing + augmentation plan (dataset-aware)
 - Report: generate Markdown (and optionally HTML/PDF if converters exist)
 - TorchLoader: generate torchvision transform snippet or Compose object preview

This file is defensive: optional modules (report generation, torch) are handled gracefully.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import gradio as gr

from imgshape.shape import get_shape
from imgshape.analyze import analyze_type, get_entropy
from imgshape.recommender import recommend_preprocessing, recommend_dataset

# optional imports guarded
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


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, default=str)
    except Exception:
        return str(obj)


# ----- Handlers used by the UI -----
def analyze_handler(image_path: str) -> Dict[str, Any]:
    """Return a JSON-like dict with shape and analysis for a single image."""
    out = {}
    try:
        out["shape"] = get_shape(image_path)
    except Exception as e:
        out["shape_error"] = str(e)

    try:
        analysis = analyze_type(image_path)
        # include entropy explicitly (helpful for augmentation heuristics)
        try:
            analysis["entropy"] = round(get_entropy(image_path), 3)
        except Exception:
            pass
        out["analysis"] = analysis
    except Exception as e:
        out["analysis_error"] = str(e)

    return out


def recommend_handler(image_path: str, include_augment: bool, seed: Optional[int]) -> Dict[str, Any]:
    """
    Recommend preprocessing + (optionally) augmentation plan.
    For a single image we treat it as a dataset of 1 (use analyze_type).
    """
    try:
        stats = analyze_type(image_path)
    except Exception:
        stats = {"image_count": 1}
    # prefer dataset-level recommend if available
    try:
        rec = recommend_dataset(stats, seed=seed)  # new unified API
    except Exception:
        # fallback to legacy recommend_preprocessing (path-based)
        try:
            prep = recommend_preprocessing(image_path)
        except Exception as e:
            prep = {"error": str(e)}
        rec = {"preprocessing": prep, "augmentation_plan": {}}

    if not include_augment:
        # hide augmentation_plan if user didn't request it
        rec["augmentation_plan"] = {}

    return rec


def report_handler(image_path: str, include_augment: bool, seed: Optional[int]) -> Dict[str, Any]:
    """
    Generate a Markdown report and return metadata + download link if created.
    Returns dict with keys: status, md_path (if produced), html_path (if produced), pdf_path (if produced)
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="imgshape_gui_"))
    out_base = tmpdir / "imgshape_report"

    # gather stats & recommendations
    try:
        stats = analyze_type(image_path)
    except Exception:
        stats = {"image_count": 1}

    try:
        unified = recommend_dataset(stats, seed=seed)
    except Exception:
        unified = {"preprocessing": recommend_preprocessing(image_path), "augmentation_plan": {}}

    if not include_augment:
        unified["augmentation_plan"] = {}

    result = {"status": "ok", "paths": {}}

    # markdown
    if generate_markdown_report is None:
        result["status"] = "md_missing"
        result["message"] = "Markdown report generator not available."
        return result

    try:
        md_path = generate_markdown_report(
            out_base.with_suffix(".md"),
            stats,
            compatibility={},
            preprocessing=unified.get("preprocessing", {}),
            augmentation_plan=unified.get("augmentation_plan", {}),
        )
        result["paths"]["markdown"] = str(md_path)
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"Error generating markdown: {e}"
        return result

    # html (optional)
    if generate_html_report is not None:
        try:
            html_path = out_base.with_suffix(".html")
            generate_html_report(md_path, html_path)
            result["paths"]["html"] = str(html_path)
        except Exception:
            # non-fatal
            pass

    # pdf (optional)
    if generate_pdf_report is not None and "html" in result["paths"]:
        try:
            pdf_path = out_base.with_suffix(".pdf")
            generate_pdf_report(Path(result["paths"]["html"]), pdf_path)
            result["paths"]["pdf"] = str(pdf_path)
        except Exception:
            pass

    return result


def torchloader_handler(image_path: str, include_augment: bool, seed: Optional[int]) -> Dict[str, Any]:
    """
    Produce a torchvision transform snippet or Compose object preview.
    Returns dict with keys: snippet (str) or info.
    """
    try:
        stats = analyze_type(image_path)
    except Exception:
        stats = {"image_count": 1}

    try:
        unified = recommend_dataset(stats, seed=seed)
    except Exception:
        unified = {"preprocessing": recommend_preprocessing(image_path), "augmentation_plan": {}}

    if not include_augment:
        unified["augmentation_plan"] = {}

    if to_torch_transform is None:
        return {"error": "torchloader not available (imgshape.torchloader missing). Install optional torch extra."}

    try:
        transforms_obj_or_snippet = to_torch_transform(unified.get("augmentation_plan", {}), unified.get("preprocessing", {}))
        if isinstance(transforms_obj_or_snippet, str):
            return {"snippet": transforms_obj_or_snippet}
        else:
            # Not easily serializable — show repr and a short help text
            return {"info": "torch transforms.Compose created (use programmatically).", "repr": repr(transforms_obj_or_snippet)}
    except Exception as e:
        return {"error": f"Error building transforms: {e}"}


# ----- Build Gradio Blocks UI -----
def launch_gui(server_port: int = 7860, share: bool = False):
    title = "📦 imgshape GUI (v2.1.0)"
    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}\nMulti-tab dataset assistant: Analyze → Recommend → Report → TorchLoader")

        with gr.Tab("Analyze"):
            with gr.Row():
                inp_img = gr.Image(type="filepath", label="Upload Image")
                out_json = gr.JSON(label="Analysis Output")
            analyze_btn = gr.Button("Analyze")
            analyze_btn.click(fn=analyze_handler, inputs=inp_img, outputs=out_json)

        with gr.Tab("Recommend"):
            with gr.Row():
                rec_img = gr.Image(type="filepath", label="Upload Image / Sample")
                include_aug = gr.Checkbox(label="Include augmentation recommendations", value=True)
                seed_in = gr.Number(label="Seed (optional)", value=None, precision=0)
            rec_out = gr.JSON(label="Recommendations (preprocessing + augmentations)")
            rec_btn = gr.Button("Recommend")
            rec_btn.click(fn=recommend_handler, inputs=[rec_img, include_aug, seed_in], outputs=rec_out)

        with gr.Tab("Report"):
            with gr.Row():
                rep_img = gr.Image(type="filepath", label="Upload Image / Sample")
                rep_include_aug = gr.Checkbox(label="Include augmentations in report", value=True)
                rep_seed = gr.Number(label="Seed (optional)", value=None, precision=0)
            rep_status = gr.JSON(label="Report generation result")
            rep_btn = gr.Button("Generate Report (Markdown)")
            rep_btn.click(fn=report_handler, inputs=[rep_img, rep_include_aug, rep_seed], outputs=rep_status)

        with gr.Tab("TorchLoader"):
            with gr.Row():
                t_img = gr.Image(type="filepath", label="Upload Image / Sample")
                t_include_aug = gr.Checkbox(label="Include augmentations", value=True)
                t_seed = gr.Number(label="Seed (optional)", value=None, precision=0)
            t_out = gr.JSON(label="Torchloader output (snippet or info)")
            t_btn = gr.Button("Build Transform Snippet")
            t_btn.click(fn=torchloader_handler, inputs=[t_img, t_include_aug, t_seed], outputs=t_out)

        gr.Markdown(
            "Notes:\n"
            "- Report generation requires `imgshape.report`. HTML/PDF outputs require `markdown` and `weasyprint` respectively.\n"
            "- TorchLoader requires `imgshape.torchloader` and, for real Compose objects, `torch`/`torchvision`.\n"
        )

    demo.launch(server_port=server_port, share=share)


if __name__ == "__main__":
    # default run for local dev
    launch_gui()
