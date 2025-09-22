# src/imgshape/torch_loader.py
"""
torch_loader.py — utilities to produce PyTorch `transforms` (or snippet) from imgshape recommendations.

Goals
-----
- Provide `to_torch_transform(config, recommendation)` as the primary integration point.
- Prefer returning an actual `torchvision.transforms.Compose` object when torchvision is available.
- If torchvision is not installed (or environment prefers snippet), return a ready-to-paste Python snippet (str).
- Be defensive: never crash on unexpected recommendation contents. Return informative errors/snippets.
- Keep the transform pipeline small and efficient by default (suitable for edge / research workflows).

Expected `recommendation` examples (supported keys)
- {"resize": {"height": 224, "width": 224, "mode": "center_crop_then_resize"}}
- {"normalize": {"imagenet_default": True}}  # or {"mean": [...], "std": [...]}
- {"augmentations": ["random_crop", "horizontal_flip"], "augmentation_params": {...}}
- {"image_count": 100, "entropy": 4.2}
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger("imgshape.torch_loader")
if not logger.handlers:
    import sys, logging as _logging

    h = _logging.StreamHandler(sys.stderr)
    h.setFormatter(_logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# Try to import torchvision transforms; if not available, we'll fall back to snippet generation.
try:
    from torchvision import transforms  # type: ignore
    _TORCHVISION_AVAILABLE = True
except Exception:
    transforms = None  # type: ignore
    _TORCHVISION_AVAILABLE = False
    logger.warning("torchvision.transforms not available; to_torch_transform will return a code snippet.")


# Default ImageNet normalization
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------- Helpers ----------------------


def _parse_resize(rec: Dict[str, Any]) -> Optional[Tuple[int, int, str]]:
    """
    Parse resize recommendation into (width, height, mode)
    Accepts different shapes for backward compatibility.
    """
    if not rec:
        return None
    # Common places the size might live
    size = rec.get("size") or rec.get("target_size") or rec.get("resize") or {}
    if isinstance(size, dict):
        h = size.get("height") or size.get("h") or size.get("H")
        w = size.get("width") or size.get("w") or size.get("W")
        mode = size.get("mode") or rec.get("mode") or "center_crop_then_resize"
        if h and w:
            return int(w), int(h), str(mode)
    # Accept strings like "224x224" or int
    if isinstance(size, str) and "x" in size:
        w, h = size.lower().split("x")
        return int(w), int(h), rec.get("mode", "center_crop_then_resize")
    if isinstance(size, int):
        return int(size), int(size), rec.get("mode", "center_crop_then_resize")
    return None


def _parse_normalize(rec: Dict[str, Any]) -> Optional[Tuple[List[float], List[float]]]:
    """
    Parse normalization recommendation into (mean, std).
    """
    if not rec:
        return None
    norm = rec.get("normalize") or rec.get("normalization")
    if isinstance(norm, dict):
        if norm.get("imagenet_default") or norm.get("imagenet"):
            return _IMAGENET_MEAN, _IMAGENET_STD
        mean = norm.get("mean")
        std = norm.get("std")
        if mean and std:
            return list(map(float, mean)), list(map(float, std))
    # legacy keys
    if rec.get("imagenet_default") or rec.get("use_imagenet"):
        return _IMAGENET_MEAN, _IMAGENET_STD
    return None


def _build_augmentations(rec: Dict[str, Any]) -> List[Any]:
    """
    Build a list of augmentation transform factories from recommendation.
    Only supports a conservative subset for safety.
    """
    aug_list: List[Any] = []
    augs = rec.get("augmentations") or rec.get("training_aug") or []
    params = rec.get("augmentation_params") or {}
    for a in augs:
        a_low = str(a).lower()
        if "random_crop" in a_low:
            # params: size or padding; fallback to center crop later if missing
            size = params.get("crop_size") or params.get("size")
            if isinstance(size, (list, tuple)):
                aug_list.append(("RandomResizedCrop", tuple(size)))
            elif isinstance(size, int):
                aug_list.append(("RandomResizedCrop", (size, size)))
            else:
                aug_list.append(("RandomResizedCrop", (224, 224)))
        elif "horizontal" in a_low and "flip" in a_low:
            aug_list.append(("RandomHorizontalFlip", params.get("p", 0.5)))
        elif "color_jitter" in a_low:
            cj_params = params.get("color_jitter") or {"brightness": 0.1, "contrast": 0.1, "saturation": 0.05, "hue": 0.02}
            aug_list.append(("ColorJitter", cj_params))
        elif "randaugment" in a_low:
            # conservative default
            aug_list.append(("RandAugment", {"n": 2, "m": 9}))
        # Skip heavy or domain-changing augmentations (e.g., aggressive color transforms) by default.
    return aug_list


def _augmentations_to_torch(aug_list: List[Any]) -> List[Any]:
    """
    Convert our internal augmentation descriptors into torchvision transforms instances.
    Only called when torchvision is available.
    """
    out: List[Any] = []
    for item in aug_list:
        if not item:
            continue
        if item[0] == "RandomResizedCrop":
            size = item[1]
            # torchvision expects int or tuple (h,w)
            out.append(transforms.RandomResizedCrop(size if isinstance(size, int) else size))
        elif item[0] == "RandomHorizontalFlip":
            p = item[1] if isinstance(item[1], (int, float)) else 0.5
            out.append(transforms.RandomHorizontalFlip(p=p))
        elif item[0] == "ColorJitter":
            params = item[1]
            out.append(
                transforms.ColorJitter(
                    brightness=params.get("brightness", 0.1),
                    contrast=params.get("contrast", 0.1),
                    saturation=params.get("saturation", 0.05),
                    hue=params.get("hue", 0.02),
                )
            )
        elif item[0] == "RandAugment":
            # torchvision >=0.9 includes RandAugment; we try to import lazily
            try:
                from torchvision.transforms import RandAugment  # type: ignore

                out.append(RandAugment(num_ops=item[1].get("n", 2), magnitude=item[1].get("m", 9)))
            except Exception:
                logger.warning("RandAugment not available in torchvision installed; skipping.")
        else:
            logger.debug("Skipping unsupported augmentation: %s", item[0])
    return out


# ---------------------- Public API ----------------------


def to_torch_transform(config: Dict[str, Any], recommendation: Dict[str, Any], prefer_snippet: bool = False) -> Union[str, Any]:
    """
    Convert a recommendation dict into a torchvision.transforms pipeline or a Python snippet (string).

    Parameters
    ----------
    config : dict
        Optional runtime config (kept for future extension; currently unused).
    recommendation : dict
        The recommendation produced by imgshape.recommender or derived helpers.
    prefer_snippet : bool
        If True, return a string snippet even if torchvision is available.

    Returns
    -------
    torchvision.transforms.Compose instance OR str snippet of Python code that constructs equivalent transforms.

    Notes
    -----
    - The caller should handle serialization of the transform object (e.g., repr) if needed.
    - When returning a snippet, it's minimal and uses torchvision; it assumes the user will paste it in an environment with torchvision installed.
    """
    try:
        rec = recommendation or {}
        # Parse resize, normalize, and augmentation info
        resize_parsed = _parse_resize(rec)
        normalize_parsed = _parse_normalize(rec)
        aug_list = _build_augmentations(rec)

        # If torchvision is available and snippet not preferred, return actual transform
        if _TORCHVISION_AVAILABLE and not prefer_snippet:
            tlist = []

            # Augmentations (training) go first
            if aug_list:
                tlist.extend(_augmentations_to_torch(aug_list))

            # Resize / crop behavior
            if resize_parsed:
                w, h, mode = resize_parsed
                # Interpret common modes
                if mode in ("center_crop_then_resize", "center_crop"):
                    # center crop to min side then resize
                    tlist.append(transforms.CenterCrop(min(w, h)))
                    tlist.append(transforms.Resize((h, w)))
                elif mode in ("resize", "force_resize", "stretch"):
                    tlist.append(transforms.Resize((h, w)))
                elif mode in ("random_resized_crop", "random_crop"):
                    tlist.append(transforms.RandomResizedCrop((h, w) if isinstance(h, int) else h))
                else:
                    # conservative fallback
                    tlist.append(transforms.Resize((h, w)))

            # ToTensor is necessary to convert PIL -> tensor and scale to [0,1]
            tlist.append(transforms.ToTensor())

            # Normalization
            if normalize_parsed:
                mean, std = normalize_parsed
                tlist.append(transforms.Normalize(mean=mean, std=std))

            # Compose
            pipeline = transforms.Compose(tlist)
            return pipeline

        # Else create a Python snippet
        snippet_lines: List[str] = [
            "from torchvision import transforms",
            "",
            "transform = transforms.Compose([",
        ]

        # Augmentations
        for item in _build_augmentations(rec):
            if item[0] == "RandomResizedCrop":
                size = item[1]
                if isinstance(size, (list, tuple)):
                    snippet_lines.append(f"    transforms.RandomResizedCrop({tuple(size)}),")
                else:
                    snippet_lines.append(f"    transforms.RandomResizedCrop({int(size)}),")
            elif item[0] == "RandomHorizontalFlip":
                p = item[1] if isinstance(item[1], (int, float)) else 0.5
                snippet_lines.append(f"    transforms.RandomHorizontalFlip(p={float(p)}),")
            elif item[0] == "ColorJitter":
                params = item[1]
                snippet_lines.append(
                    "    transforms.ColorJitter("
                    f"brightness={params.get('brightness',0.1)},"
                    f"contrast={params.get('contrast',0.1)},"
                    f"saturation={params.get('saturation',0.05)},"
                    f"hue={params.get('hue',0.02)}),"
                )
            elif item[0] == "RandAugment":
                snippet_lines.append("    # RandAugment may require torchvision >= 0.9")
                snippet_lines.append(f"    transforms.RandAugment(num_ops={item[1].get('n',2)}, magnitude={item[1].get('m',9)}),")
            else:
                snippet_lines.append(f"    # Unsupported augmentation: {item[0]} (skipped)")

        # Resize
        if resize_parsed:
            w, h, mode = resize_parsed
            if mode in ("center_crop_then_resize", "center_crop"):
                snippet_lines.append(f"    transforms.CenterCrop({min(w, h)}),")
                snippet_lines.append(f"    transforms.Resize(({h}, {w})),")
            elif mode in ("resize", "force_resize", "stretch"):
                snippet_lines.append(f"    transforms.Resize(({h}, {w})),")
            elif mode in ("random_resized_crop", "random_crop"):
                snippet_lines.append(f"    transforms.RandomResizedCrop(({h}, {w})),")
            else:
                snippet_lines.append(f"    transforms.Resize(({h}, {w})),")
        else:
            # default safe resize to 224
            snippet_lines.append("    transforms.Resize((224, 224)),")
        snippet_lines.append("    transforms.ToTensor(),")

        # Normalize
        if normalize_parsed:
            mean, std = normalize_parsed
            snippet_lines.append(f"    transforms.Normalize(mean={mean}, std={std}),")
        else:
            snippet_lines.append(f"    transforms.Normalize(mean={_IMAGENET_MEAN}, std={_IMAGENET_STD}),")

        snippet_lines.append("])")
        snippet = "\n".join(snippet_lines)
        return snippet

    except Exception as e:
        logger.exception("to_torch_transform failed: %s", e)
        # Return a defensive snippet that the user can paste
        fallback = textwrap = None
        try:
            import textwrap as _tw

            textwrap = _tw
        except Exception:
            textwrap = None
        fallback_snippet = (
            "from torchvision import transforms\n\n"
            "transform = transforms.Compose([\n"
            "    transforms.Resize((224,224)),\n"
            "    transforms.ToTensor(),\n"
            f"    transforms.Normalize(mean={_IMAGENET_MEAN}, std={_IMAGENET_STD}),\n"
            "])\n"
        )
        return {"error": "transform_generation_failed", "detail": str(e), "fallback_snippet": fallback_snippet}
