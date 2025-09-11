# src/imgshape/gui.py
from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, Optional
from PIL import Image, UnidentifiedImageError
import gradio as gr

from imgshape.shape import get_shape
from imgshape.analyze import analyze_type
from imgshape.recommender import recommend_preprocessing, recommend_dataset

# Optional: AugmentationRecommender for dataset-level augment suggestions
try:
    from imgshape.augmentations import AugmentationRecommender
except Exception:
    AugmentationRecommender = None


def _open_bytes_to_pil(b: Any) -> Optional[Image.Image]:
    # Accept filepath string, BytesIO, or file-like object
    if b is None:
        return None
    if isinstance(b, Image.Image):
        return b.convert("RGB")
    try:
        if hasattr(b, "read"):
            b.seek(0)
            data = b.read()
            return Image.open(BytesIO(data)).convert("RGB")
    except UnidentifiedImageError:
        return None
    except Exception:
        return None
    try:
        # if it's a path
        return Image.open(str(b)).convert("RGB")
    except Exception:
        return None


def analyze_handler(inp: Any) -> Dict[str, Any]:
    """Return shape + analysis for a given input (file, path, URL)."""
    try:
        # if input is a directory path -> not supported here (use dataset analyze)
        if isinstance(inp, str):
            # if path looks like a directory, let GUI call dataset functions separately
            pass
        pil = _open_bytes_to_pil(inp)
        shape = get_shape(pil) if pil is not None else None
        analysis = analyze_type(pil if pil is not None else inp)
        return {"shape": shape, "analysis": analysis}
    except Exception as e:
        return {"error": str(e)}


def recommend_handler(inp: Any, include_augment: bool = False, seed: Optional[int] = None) -> Dict[str, Any]:
    """
    For a single image: call recommend_preprocessing(pil_image).
    For a directory path: call recommend_dataset(path).
    """
    try:
        # if input is a string path and is a directory, call dataset-level recommender
        if isinstance(inp, str):
            from pathlib import Path
            p = Path(inp)
            if p.exists() and p.is_dir():
                ds_rec = recommend_dataset(str(p))
                return {"dataset_recommendation": ds_rec}

        pil = _open_bytes_to_pil(inp)
        if pil is None:
            return {"error": "Could not open image input"}

        pre = recommend_preprocessing(pil)
        out = {"preprocessing": pre}
        if include_augment and AugmentationRecommender is not None:
            ar = AugmentationRecommender(seed=seed)
            plan = ar.recommend_for_dataset({"entropy_mean": pre.get("entropy"), "image_count": 1})
            out["augmentation_plan"] = {"order": plan.recommended_order, "augmentations": [a.__dict__ for a in plan.augmentations]}
        return out
    except Exception as e:
        return {"error": str(e)}


def report_handler(dataset_path: str, include_augment: bool = False, seed: Optional[int] = None) -> Dict[str, Any]:
    """Wrapper for generating a report. Returns paths/status dict (non-blocking minimal)."""
    try:
        # Prefer dataset-level stats if possible; we call recommend_dataset to get preprocessing summary
        ds_rec = recommend_dataset(dataset_path)
        # minimal report generation could be handled here; for GUI we return the recommended summary
        return {"status": "ok", "dataset_recommendation": ds_rec}
    except Exception as e:
        return {"error": str(e)}


def torchloader_handler(inp: Any, include_augment: bool = False, seed: Optional[int] = None) -> Dict[str, Any]:
    """Return a transform snippet or an object (string or repr) for the UI to display."""
    try:
        # prefer preprocessing dict from single-image or dataset
        if isinstance(inp, str):
            from pathlib import Path
            p = Path(inp)
            if p.exists() and p.is_dir():
                pre = recommend_dataset(str(p))
            else:
                pre = recommend_preprocessing(inp)
        else:
            pil = _open_bytes_to_pil(inp)
            if pil is None:
                return {"error": "Could not open input"}
            pre = recommend_preprocessing(pil)

        # import lazy to avoid heavy deps at module import
        from imgshape.torchloader import to_torch_transform

        snippet_or_transform = to_torch_transform({}, pre or {})
        if isinstance(snippet_or_transform, str):
            return {"snippet": snippet_or_transform}
        else:
            return {"transform_repr": repr(snippet_or_transform)}
    except Exception as e:
        return {"error": str(e)}


def launch_gui(server_port: int = 7860, share: bool = False):
    """Small Gradio interface that binds the handlers above."""
    import gradio as gr

    with gr.Blocks(title="imgshape") as demo:
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="filepath", label="Upload Image or enter path")
                analyze_btn = gr.Button("Analyze")
                recommend_btn = gr.Button("Recommend")
                torch_btn = gr.Button("TorchLoader")
            with gr.Column():
                out = gr.JSON(label="Output")

        analyze_btn.click(fn=analyze_handler, inputs=inp, outputs=out)
        recommend_btn.click(fn=recommend_handler, inputs=inp, outputs=out)
        torch_btn.click(fn=torchloader_handler, inputs=inp, outputs=out)

    demo.launch(server_port=server_port, share=share)


# allow direct run for quick dev
if __name__ == "__main__":
    launch_gui()
