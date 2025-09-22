# src/imgshape/gui.py
from __future__ import annotations

"""
Lightweight, robust Gradio GUI for imgshape v2.2.0

Changes in this version:
- gr.JSON replaced by a plain JSON text box to always show full expanded JSON.
- Handlers already sanitize outputs to JSON-friendly types via _make_serializable.
- _wrap_handler now returns pretty-printed JSON text for the output textbox.
- Minor UI polish: expose augmentation checkbox and seed as explicit controls (reused across buttons).
"""

from typing import Any, Dict, Optional, List, Union
from pathlib import Path
import logging
from io import BytesIO
import base64
import textwrap
import json

import numpy as np
from PIL import Image, ImageOps
import gradio as gr

try:
    from imgshape.analyze import analyze_type
except Exception:
    analyze_type = None

try:
    from imgshape.recommender import recommend_preprocessing, recommend_dataset
except Exception:
    recommend_preprocessing = None
    recommend_dataset = None

try:
    from imgshape.shape import get_shape
except Exception:
    get_shape = None

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


# ---------------------- Serialization helpers ----------------------


def _to_serializable_scalar(x: Any) -> Any:
    """Convert numpy scalar types to native python scalars."""
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _to_serializable(obj: Any) -> Any:
    """Recursively convert numpy arrays/scalars and bytes to JSON-friendly types."""
    # None, str, int, float, bool pass through
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # Numpy scalar
    if isinstance(obj, (np.generic,)):
        return _to_serializable_scalar(obj)

    # Bytes -> base64 string (rare in analysis results)
    if isinstance(obj, (bytes, bytearray)):
        try:
            return base64.b64encode(obj).decode("ascii")
        except Exception:
            return str(obj)

    # Numpy arrays -> lists
    if isinstance(obj, np.ndarray):
        try:
            return obj.tolist()
        except Exception:
            # fallback to list conversion elementwise
            return [_to_serializable_scalar(x) for x in obj.flatten().tolist()]

    # Lists/tuples -> convert elements
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]

    # Dict -> convert values recursively
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}

    # PIL Image -> not serializable; return basic metadata
    try:
        if isinstance(obj, Image.Image):
            w, h = obj.size
            return {"_pil_image": True, "width": w, "height": h, "mode": obj.mode}
    except Exception:
        pass

    # Other objects -> fallback to str()
    try:
        return str(obj)
    except Exception:
        return None


def _make_serializable(obj: Any) -> Any:
    """Top-level wrapper: attempts to make a complex nested structure JSON-serializable."""
    try:
        return _to_serializable(obj)
    except Exception as e:
        logger.warning("Serialization failed: %s", e)
        # Last resort: return a string description
        try:
            return str(obj)
        except Exception:
            return {"error": "unserializable_object", "detail": repr(obj)}


# ---------------------- Image / input helpers ----------------------


def _pil_to_base64(img: Image.Image, format: str = "PNG") -> str:
    buf = BytesIO()
    img.save(buf, format=format)
    b = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{format.lower()};base64,{b}"


def _safe_open_image_from_path(path: str) -> Optional[Image.Image]:
    try:
        p = Path(path)
        if not p.exists():
            return None
        img = Image.open(p).convert("RGB")
        return img
    except Exception:
        return None


def _normalize_input(inp: Any) -> Any:
    """Normalize Gradio input to one of: PIL.Image, path string, bytes, or raw value."""
    if inp is None:
        return None
    try:
        if isinstance(inp, Image.Image):
            return inp.convert("RGB")
    except Exception:
        pass
    try:
        if isinstance(inp, np.ndarray):
            arr = inp
            if np.issubdtype(arr.dtype, np.floating):
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype("uint8")
            return Image.fromarray(arr).convert("RGB")
    except Exception:
        pass
    try:
        if isinstance(inp, (bytes, bytearray)):
            return Image.open(BytesIO(inp)).convert("RGB")
    except Exception:
        pass
    try:
        if isinstance(inp, str):
            p = Path(inp)
            if p.exists():
                return str(p)
            return inp
    except Exception:
        pass
    return inp


# ---------------------- 3D helper ----------------------


def _threejs_plane_html(image_data_url: str) -> str:
    return f"""
    <div style="width:100%;height:420px; background:#111; color:#eee;">
      <div id="three-root" style="width:100%;height:100%"></div>
    </div>
    <script crossorigin src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r152/three.min.js"></script>
    <script>
    (function(){{
      const root=document.getElementById('three-root');
      try {{
        const scene=new THREE.Scene();
        const camera=new THREE.PerspectiveCamera(45, root.clientWidth/root.clientHeight,0.1,1000);
        camera.position.set(0,0,3);
        const renderer=new THREE.WebGLRenderer({{antialias:true}});
        renderer.setSize(root.clientWidth,root.clientHeight);
        root.appendChild(renderer.domElement);
        const geometry=new THREE.PlaneGeometry(1.6,1.0);
        const texture=new THREE.TextureLoader().load('{image_data_url}');
        const material=new THREE.MeshBasicMaterial({{map:texture}});
        const plane=new THREE.Mesh(geometry,material);
        scene.add(plane);
        function animate(){{requestAnimationFrame(animate); renderer.render(scene,camera);}}
        animate();
      }} catch (err) {{
        root.innerHTML = '<div style="padding:12px;color:#ddd">3D preview unavailable: ' + err.toString() + '</div>';
        console.warn('threejs embed failed', err);
      }}
    }})();
    </script>
    """


# ---------------------- Handlers (sanitized outputs) ----------------------


def analyze_handler(inp: Any) -> Dict[str, Any]:
    try:
        norm = _normalize_input(inp)
        if analyze_type is None:
            return {"error": "analyze_unavailable"}
        analysis = analyze_type(norm)
        # Ensure serializable
        analysis = _make_serializable(analysis)
        shape = None
        if isinstance(norm, Image.Image):
            w, h = norm.size
            c = len(norm.getbands())
            shape = (h, w, c)
        elif isinstance(norm, str):
            img = _safe_open_image_from_path(norm)
            if img:
                w, h = img.size
                shape = (h, w, len(img.getbands()))
        return {"shape": shape, "analysis": analysis}
    except Exception as e:
        logger.exception("analyze_handler failed: %s", e)
        return {"error": "analyze_failed", "detail": str(e)}


def recommend_handler(inp: Any, prefs: Optional[str] = None, include_augment: bool = False, seed: Optional[int] = None) -> Dict[str, Any]:
    try:
        norm = _normalize_input(inp)
        user_prefs = [p.strip() for p in prefs.split(",") if p.strip()] if prefs else None
        if isinstance(norm, str) and Path(norm).is_dir():
            if recommend_dataset is None:
                return {"error": "recommend_dataset_unavailable"}
            ds = recommend_dataset(norm, user_prefs=user_prefs)
            ds = _make_serializable(ds)
            return {"dataset_recommendation": ds}

        if recommend_preprocessing is None:
            return {"error": "recommender_unavailable"}
        rec = recommend_preprocessing(norm, user_prefs=user_prefs)
        rec = _make_serializable(rec)
        out: Dict[str, Any] = {"preprocessing": rec}

        if include_augment and AugmentationRecommender is not None:
            try:
                ar = AugmentationRecommender(seed=seed)
                plan = ar.recommend_for_dataset({"entropy_mean": rec.get("entropy", 0.0), "image_count": rec.get("image_count", 1)})
                out["augmentation_plan"] = _make_serializable({"order": getattr(plan, "recommended_order", None), "augmentations": getattr(plan, "augmentations", None)})
            except Exception as e:
                logger.warning("AugmentationRecommender failed: %s", e)
                out["augmentation_plan_error"] = str(e)

        return out
    except Exception as e:
        logger.exception("recommend_handler failed: %s", e)
        return {"error": "recommendation_failed", "detail": str(e)}


def torchloader_handler(inp: Any, prefs: Optional[str] = None) -> Dict[str, Any]:
    try:
        norm = _normalize_input(inp)
        pre = None
        if isinstance(norm, str) and Path(norm).is_dir():
            if recommend_dataset:
                pre = recommend_dataset(norm)
        elif recommend_preprocessing:
            pre = recommend_preprocessing(norm)
        pre = _make_serializable(pre or {})

        try:
            from imgshape.torchloader import to_torch_transform
        except Exception:
            to_torch_transform = None

        if to_torch_transform is None:
            return {"error": "torchloader_unavailable", "message": "to_torch_transform not found"}

        snippet_or_transform = to_torch_transform({}, pre or {})
        # If snippet_or_transform is not JSON-serializable, represent via repr()
        if isinstance(snippet_or_transform, str):
            out = {"snippet": snippet_or_transform}
        else:
            try:
                out = {"transform_repr": repr(snippet_or_transform)}
            except Exception:
                out = {"transform_repr": str(type(snippet_or_transform))}
        return _make_serializable(out)
    except Exception as e:
        logger.exception("torchloader_handler failed: %s", e)
        return {"error": "torchloader_failed", "detail": str(e)}


# ---------------------- UI wiring helpers ----------------------


def _wrap_handler(fn, *args):
    """
    Generic wrapper used by Gradio buttons to call handlers and produce
    outputs: (json_text, pil_image or None, html or fallback)
    """
    try:
        res = fn(*args)
    except Exception as e:
        logger.exception("Handler raised: %s", e)
        res = {"error": "handler_exception", "detail": str(e)}

    # Ensure serializable object
    json_obj = _make_serializable(res)
    try:
        json_text = json.dumps(json_obj, indent=2, ensure_ascii=False)
    except Exception:
        # Last-resort: string representation
        json_text = str(json_obj)

    # Build image preview: prefer a 'preview_image' key or derive from first arg
    pil_img = None
    html = "<div style='padding:12px;color:#ddd'>No image available for 3D preview.</div>"
    try:
        # if handler returned a preview image under known keys, use it
        if isinstance(res, dict):
            preview = res.get("preview_image") or res.get("image_preview") or None
            if isinstance(preview, Image.Image):
                pil_img = preview
        # fallback: if first arg is a PIL image, use that
        if pil_img is None and args:
            maybe_img = args[0]
            if isinstance(maybe_img, Image.Image):
                pil_img = maybe_img
            elif isinstance(maybe_img, str):
                # maybe a path
                pil_img = _safe_open_image_from_path(maybe_img)
        if pil_img is not None:
            thumb = ImageOps.contain(pil_img, (800, 500))
            data_url = _pil_to_base64(thumb, format="PNG")
            html = _threejs_plane_html(data_url)
    except Exception as e:
        logger.warning("Failed to build image/3D preview: %s", e)
        html = f"<div style='padding:12px;color:#ddd'>3D preview not available: {e}</div>"

    return json_text, pil_img, html


def _CSS():
    return textwrap.dedent(
        """
    .gradio-container { font-family: Inter, sans-serif; }
    .gradio-row .gradio-column:first-child { position: sticky; top: 12px; align-self:flex-start; }
    .gr-json { font-size: 13px; }
    """
    )


# ---------------------- Launch GUI ----------------------


def launch_gui(server_port: int = 7860, share: bool = False):
    with gr.Blocks(title="imgshape", css=_CSS()) as demo:
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                inp = gr.Image(type="pil", label="Upload Image")
                path_text = gr.Textbox(label="Or enter path")
                prefs = gr.Textbox(label="Prefs")
                include_aug = gr.Checkbox(label="Include augmentation plan", value=False)
                seed = gr.Number(label="Augmentation seed", value=0)
                analyze_btn = gr.Button("Analyze")
                recommend_btn = gr.Button("Recommend")
                torch_btn = gr.Button("TorchLoader")
            with gr.Column(scale=2, min_width=520):
                with gr.Tabs():
                    with gr.TabItem("JSON"):
                        out = gr.Textbox(label="Output (JSON)", value="", lines=30)
                    with gr.TabItem("Image"):
                        img_preview = gr.Image(type="pil", label="Processed Preview")
                    with gr.TabItem("3D"):
                        html_preview = gr.HTML(
                            value="<div style='padding:12px'>Upload an image and click Analyze/Recommend to see 3D preview.</div>"
                        )

        analyze_btn.click(
            fn=lambda image_obj, path_text_value: _wrap_handler(analyze_handler, image_obj or path_text_value),
            inputs=[inp, path_text],
            outputs=[out, img_preview, html_preview],
        )
        recommend_btn.click(
            fn=lambda image_obj, path_text_value, prefs_v, include_aug_v, seed_v: _wrap_handler(
                recommend_handler,
                image_obj or path_text_value,
                prefs_v,
                include_aug_v,
                int(seed_v) if seed_v is not None else None,
            ),
            inputs=[inp, path_text, prefs, include_aug, seed],
            outputs=[out, img_preview, html_preview],
        )
        torch_btn.click(
            fn=lambda image_obj, path_text_value, prefs_v: _wrap_handler(torchloader_handler, image_obj or path_text_value, prefs_v),
            inputs=[inp, path_text, prefs],
            outputs=[out, img_preview, html_preview],
        )

    demo.launch(server_port=server_port, share=share)


if __name__ == "__main__":
    launch_gui()
