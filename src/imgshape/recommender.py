# src/imgshape/recommender.py
import os
from pathlib import Path
from typing import Any, Dict, Tuple, Union, Optional
from PIL import Image, UnidentifiedImageError
from io import BytesIO

# import your existing helpers
from imgshape.shape import get_shape
from imgshape.analyze import get_entropy


def _open_image_from_input(inp: Any) -> Optional[Image.Image]:
    """
    Accepts: PIL.Image.Image, str/path, bytes/BytesIO, UploadedFile-like
    Returns a PIL.Image.Image or None
    """
    # already a PIL image
    try:
        from PIL import Image as PILImage
        if isinstance(inp, PILImage.Image):
            return inp
    except Exception:
        pass

    # path-like (str or Path)
    if isinstance(inp, (str, Path)):
        try:
            return Image.open(str(inp)).convert("RGB")
        except Exception:
            return None

    # bytes or BytesIO or file-like
    if hasattr(inp, "read"):
        try:
            # ensure we are at start
            try:
                inp.seek(0)
            except Exception:
                pass
            data = inp.read()
            if not data:
                return None
            buf = BytesIO(data)
            return Image.open(buf).convert("RGB")
        except Exception:
            return None

    # raw bytes
    if isinstance(inp, (bytes, bytearray)):
        try:
            return Image.open(BytesIO(inp)).convert("RGB")
        except Exception:
            return None

    return None


def _normalize_mean_std_for_channels(channels: int) -> Tuple[list, list]:
    """
    Return sensible default ImageNet-like mean/std for channels=1 or 3.
    """
    if channels == 1:
        mean = [0.5]
        std = [0.5]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    return mean, std


def recommend_preprocessing(input_obj: Any) -> Dict[str, Any]:
    """
    Suggest preprocessing steps for ML models.

    Accepts:
      - stats dict (already-computed dataset stats) OR
      - single image input (PIL.Image, path, BytesIO, uploaded-file)
    Returns a dict with keys:
      - resize: (w, h) or dict with 'size' and 'method'
      - normalize: {'mean': [...], 'std': [...]}
      - mode: "RGB" or "Grayscale"
      - entropy: float (if available)
      - suggested_model: str (hint)
    """
    # If the user already passed a dict-like stats object, use it
    if isinstance(input_obj, dict):
        stats = input_obj
        # try to extract minimal values
        entropy = stats.get("entropy_mean") or stats.get("entropy") or stats.get("entropy_mean")
        channels = stats.get("channels")
        # shape: try to find representative image shape (use most_common)
        shape_dist = stats.get("shape_distribution", {})
        rep_shape = None
        if isinstance(shape_dist, dict):
            unique_shapes = shape_dist.get("unique_shapes", {})
            if unique_shapes:
                # take first key like "400x300"
                ks = list(unique_shapes.keys())
                if ks:
                    try:
                        w, h = ks[0].split("x")
                        rep_shape = (int(h), int(w))  # function uses (h,w)
                    except Exception:
                        rep_shape = None
        # fallthrough if rep_shape None
    else:
        # Try to interpret input_obj as an image and compute stats
        pil = _open_image_from_input(input_obj)
        if pil is None:
            return {"error": "Unsupported input for recommend_preprocessing. Provide path, PIL.Image, BytesIO, or stats dict."}

        # compute shape using existing helper (get_shape supports path or image depending on your impl)
        try:
            shape = get_shape(pil)
        except Exception:
            # fallback to pillow size
            try:
                w, h = pil.size
                shape = (h, w, len(pil.getbands()))
            except Exception:
                shape = None

        try:
            entropy = round(get_entropy(pil), 3)
        except Exception:
            entropy = None

        channels = shape[2] if shape and len(shape) >= 3 else (len(pil.getbands()) if pil else 3)
        rep_shape = (shape[0], shape[1]) if shape and len(shape) >= 2 else None

    # Prepare recommendation
    rec: Dict[str, Any] = {}
    # Decide resize based on min dim (use representative shape if available)
    min_side = None
    if rep_shape:
        # rep_shape is (h,w) — be defensive
        try:
            h0, w0 = int(rep_shape[0]), int(rep_shape[1])
            min_side = min(h0, w0)
        except Exception:
            min_side = None

    # fallback to sensible defaults
    if min_side is None:
        min_side = 224

    # choose sizes
    if min_side >= 224:
        rec["resize"] = {"size": [224, 224], "method": "bilinear"}
        rec["suggested_model"] = "MobileNet/ResNet"
    elif min_side >= 96:
        rec["resize"] = {"size": [96, 96], "method": "bilinear"}
        rec["suggested_model"] = "EfficientNet-B0 (small)"
    elif min_side <= 32:
        rec["resize"] = {"size": [32, 32], "method": "nearest"}
        rec["suggested_model"] = "TinyNet/MNIST/CIFAR"
    else:
        rec["resize"] = {"size": [128, 128], "method": "bilinear"}
        rec["suggested_model"] = "General Use"

    rec["mode"] = "RGB" if channels == 3 else "Grayscale"
    mean, std = _normalize_mean_std_for_channels(channels if channels else 3)
    rec["normalize"] = {"mean": mean, "std": std}
    rec["entropy"] = entropy

    return rec


# Backwards compatibility alias
recommend = recommend_preprocessing
