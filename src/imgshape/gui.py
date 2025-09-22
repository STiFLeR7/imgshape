# src/imgshape/gui.py
from __future__ import annotations

from typing import Any, Dict, Optional, List
from pathlib import Path
import logging
from io import BytesIO

import numpy as np
from PIL import Image

import gradio as gr

from imgshape.analyze import analyze_type
from imgshape.recommender import recommend_preprocessing, recommend_dataset
from imgshape.shape import get_shape

# Optional augmentation recommender
try:
    from imgshape.augmentations import AugmentationRecommender
except Exception:
    AugmentationRecommender = None

logger = logging.getLogger("imgshape.gui")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


def _normalize_input(inp: Any) -> Any:
    """Normalize Gradio input to PIL.Image, path string, or bytes."""
    if inp is None:
        return None

    try:
        if isinstance(inp, Image.Image):
            return inp.convert("RGB")
    except Exception:
        logger.debug("Not a PIL.Image", exc_info=True)

    try:
        import numpy as _np
        if isinstance(inp, _np.ndarray):
            arr = inp
            if arr.dtype != _np.uint8:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype("uint8")
            return Image.fromarray(arr).convert("RGB")
    except Exception:
        logger.debug("Not a numpy array or conversion failed", exc_info=True)

    try:
        if isinstance(inp, str):
            p = Path(inp)
            if p.exists():
                return str(p)
            return inp
    except Exception:
        logger.debug("String->Path handling failed", exc_info=True)

    try:
        if hasattr(inp, "read"):
            try:
                pos = None
                try:
                    pos = inp.tell()
                except Exception:
                    pos = None
                data = inp.read()
                if isinstance(data, (bytes, bytearray)):
                    return Image.open(BytesIO(data)).convert("RGB")
                if isinstance(data, str):
                    return data.encode("utf8")
                if isinstance(data, np.ndarray):
                    return Image.fromarray(data).convert("RGB")
                if pos is not None:
                    try:
                        inp.seek(pos)
                    except Exception:
                        pass
            except Exception:
                logger.debug("file-like read failed", exc_info=True)
    except Exception:
        logger.debug("No read attribute", exc_info=True)

    return inp


def analyze_handler(inp: Any) -> Dict[str, Any]:
    try:
        norm = _normalize_input(inp)
        analysis = analyze_type(norm)
        shape = None
        if isinstance(norm, Image.Image):
            w, h = norm.size
            c = len(norm.getbands()) if norm.getbands() else None
            shape = (h, w, c)
        elif isinstance(norm, str):
            try:
                shp = get_shape(norm)
                shape = tuple(shp) if shp else None
            except Exception:
                shape = None
        return {"shape": shape, "analysis": analysis}
    except Exception as exc:
        logger.exception("analyze_handler failed: %s", exc)
        return {"error": "analyze failed", "detail": str(exc)}


def recommend_handler(inp: Any, prefs: Optional[str] = None, include_augment: bool = False, seed: Optional[int] = None) -> Dict[str, Any]:
    try:
        norm = _normalize_input(inp)
        user_prefs: Optional[List[str]] = None
        if prefs:
            user_prefs = [p.strip() for p in prefs.split(",") if p.strip()]

        if isinstance(norm, str):
            p = Path(norm)
            if p.exists() and p.is_dir():
                ds_rec = recommend_dataset(str(p), user_prefs=user_prefs)
                return {"dataset_recommendation": ds_rec}

        pre = recommend_preprocessing(norm, user_prefs=user_prefs)
        out = {"preprocessing": pre}
        if include_augment and AugmentationRecommender is not None:
            ar = AugmentationRecommender(seed=seed)
            plan = ar.recommend_for_dataset({"entropy_mean": pre.get("entropy"), "image_count": 1})
            out["augmentation_plan"] = {"order": plan.recommended_order, "augmentations": [a.__dict__ for a in plan.augmentations]}
        return out
    except Exception as exc:
        logger.exception("recommend_handler failed: %s", exc)
        return {"error": "recommendation failed", "detail": str(exc)}


def torchloader_handler(inp: Any, prefs: Optional[str] = None, include_augment: bool = False, seed: Optional[int] = None) -> Dict[str, Any]:
    try:
        norm = _normalize_input(inp)
        user_prefs: Optional[List[str]] = None
        if prefs:
            user_prefs = [p.strip() for p in prefs.split(",") if p.strip()]

        if isinstance(norm, str):
            p = Path(norm)
            if p.exists() and p.is_dir():
                pre = recommend_dataset(str(p), user_prefs=user_prefs)
            else:
                pre = recommend_preprocessing(norm, user_prefs=user_prefs)
        else:
            pre = recommend_preprocessing(norm, user_prefs=user_prefs)

        from imgshape.torchloader import to_torch_transform

        snippet_or_transform = to_torch_transform({}, pre or {})
        if isinstance(snippet_or_transform, str):
            return {"snippet": snippet_or_transform}
        else:
            return {"transform_repr": repr(snippet_or_transform)}
    except Exception as exc:
        logger.exception("torchloader_handler failed: %s", exc)
        return {"error": "torchloader failed", "detail": str(exc)}


def launch_gui(server_port: int = 7860, share: bool = False):
    with gr.Blocks(title="imgshape") as demo:
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Upload Image or enter path")
                prefs = gr.Textbox(label="Preference chips (comma-separated, e.g. fast,small,quality)", value="")
                analyze_btn = gr.Button("Analyze")
                recommend_btn = gr.Button("Recommend")
                torch_btn = gr.Button("TorchLoader")
            with gr.Column():
                out = gr.JSON(label="Output", value={})

        analyze_btn.click(fn=analyze_handler, inputs=inp, outputs=out)
        recommend_btn.click(fn=recommend_handler, inputs=[inp, prefs], outputs=out)
        torch_btn.click(fn=torchloader_handler, inputs=[inp, prefs], outputs=out)

    demo.launch(server_port=server_port, share=share)


if __name__ == "__main__":
    launch_gui()
