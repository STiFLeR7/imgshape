# service/app.py
"""
FastAPI wrapper for imgshape v4.1.0 — robust archive support + UI integration

Notes:
 - Injects runtime endpoints into the index template as `window.IMGSHAPE` (via Jinja).
 - Keeps archive extraction safety, remote download limits, and temp cleanup.
 - Adds debug endpoints to inspect static files served by FastAPI.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import shutil
import tempfile
import uuid
import zipfile
import tarfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

# imgshape internals (assumed present in your PYTHONPATH)
from imgshape.analyze import analyze_type
from imgshape.compatibility import check_compatibility
from imgshape.recommender import recommend_preprocessing, recommend_dataset
from imgshape.shape import get_shape

# v4 imports for Atlas functionality
try:
    from imgshape.atlas import Atlas, analyze_dataset as analyze_dataset_v4
    from imgshape.decision_v4 import UserIntent, UserConstraints, TaskType, DeploymentTarget, Priority
    from imgshape.fingerprint_v4 import FingerprintExtractor, GPUHandler
    from imgshape.compare_v4 import DatasetComparator
    V4_AVAILABLE = True
except ImportError:
    # logger may not be initialized yet; set flag only
    V4_AVAILABLE = False

__version__ = "4.1.0"

# -------------------------
# Configuration (env-driven)
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = Path(os.environ.get("IMGSHAPE_DATASET_ROOT", str(BASE_DIR / "datasets"))).resolve()
DATASET_CACHE = Path(os.environ.get("IMGSHAPE_DATASET_CACHE", str(BASE_DIR / ".dataset_cache"))).resolve()
MAX_REMOTE_BYTES = int(os.environ.get("IMGSHAPE_MAX_REMOTE_BYTES", str(2 * 1024 * 1024 * 1024)))  # 2GB default
MAX_EXTRACT_FILES = int(os.environ.get("IMGSHAPE_MAX_EXTRACT_FILES", str(100_000)))
CORS_ORIGINS_RAW = os.environ.get("IMGSHAPE_CORS_ORIGINS", '["*"]')
try:
    CORS_ORIGINS = json.loads(CORS_ORIGINS_RAW)
except Exception:
    CORS_ORIGINS = ["*"]

try:
    _registry_raw = os.environ.get("IMGSHAPE_DATASET_REGISTRY", "{}")
    DATASET_REGISTRY: Dict[str, str] = json.loads(_registry_raw)
except Exception:
    DATASET_REGISTRY = {}

# ensure roots exist
DATASET_ROOT.mkdir(parents=True, exist_ok=True)
DATASET_CACHE.mkdir(parents=True, exist_ok=True)

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("imgshape.api")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("streamlit").setLevel(logging.ERROR)

# -------------------------
# FastAPI app & middleware
# -------------------------
app = FastAPI(
    title="imgshape API",
    description="FastAPI wrapper for imgshape — analyze, recommend, and compatibility (archive-safe)",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------- serialization helpers ---------------------


def _to_serializable_scalar(x: Any) -> Any:
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None

    if np is not None:
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        if isinstance(x, (np.generic,)):
            return _to_serializable_scalar(x.item())
    return x


def _to_serializable(obj: Any) -> Any:
    # primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None

    # numpy arrays
    if np is not None and isinstance(obj, np.ndarray):
        try:
            return obj.tolist()
        except Exception:
            return [_to_serializable_scalar(x) for x in obj.flatten().tolist()]

    if np is not None and isinstance(obj, (np.integer, np.floating, np.bool_, np.generic)):
        return _to_serializable_scalar(obj)

    # bytes
    if isinstance(obj, (bytes, bytearray)):
        try:
            return base64.b64encode(obj).decode("ascii")
        except Exception:
            return str(obj)

    # pillow image -> small preview
    if isinstance(obj, Image.Image):
        try:
            w, h = obj.size
            buf = io.BytesIO()
            obj.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return {"_pil_image": True, "width": w, "height": h, "mode": obj.mode, "preview_base64": f"data:image/png;base64,{b64}"}
        except Exception:
            return {"_pil_image": True, "info": str(obj)}

    # mapping
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            try:
                out[str(k)] = _to_serializable(v)
            except Exception:
                out[str(k)] = str(v)
        return out

    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]

    try:
        if hasattr(obj, "tolist"):
            tl = obj.tolist()
            return _to_serializable(tl)
    except Exception:
        pass

    try:
        return str(obj)
    except Exception:
        return {"error": "unserializable_object", "repr": repr(obj)}


def make_serializable_response(obj: Any) -> Any:
    try:
        return _to_serializable(obj)
    except Exception as e:
        logger.warning("Serialization fallback: %s", e)
        return {"error": "serialization_failed", "detail": str(e), "repr": str(obj)}


def format_decision_pretty(decision: Dict[str, Any]) -> str:
    """Format a single decision into human-readable text"""
    lines = []
    lines.append(f"✓ {decision.get('selected', 'N/A')}")
    lines.append(f"  Confidence: {decision.get('confidence', 0):.2%}")
    
    why = decision.get('why', [])
    if why:
        lines.append("  Reasoning:")
        for i, reason in enumerate(why, 1):
            lines.append(f"    {i}. {reason}")
    
    alternatives = decision.get('alternatives', [])
    if alternatives:
        lines.append(f"  Alternatives: {', '.join(alternatives)}")
    
    return "\n".join(lines)


def format_fingerprint_pretty(fingerprint: Dict[str, Any]) -> str:
    """Format fingerprint into human-readable text"""
    lines = []
    lines.append("=== Dataset Fingerprint ===\n")
    lines.append(f"URI: {fingerprint.get('dataset_uri', 'N/A')}")
    lines.append(f"ID: {fingerprint.get('dataset_id', 'N/A')[:16]}...")
    lines.append(f"Samples: {fingerprint.get('sample_count', 0):,}\n")
    
    profiles = fingerprint.get('profiles', {})
    
    if 'spatial' in profiles:
        spatial = profiles['spatial']
        lines.append("📐 Spatial Profile:")
        lines.append(f"  Resolution: {spatial.get('resolution_range', {}).get('median', 'N/A')}")
        lines.append(f"  Aspect Ratio Variance: {spatial.get('aspect_ratio_variance', 0):.3f}")
        lines.append("")
    
    if 'signal' in profiles:
        signal = profiles['signal']
        lines.append("🔌 Signal Profile:")
        lines.append(f"  Channels: {signal.get('channel_count', 'N/A')}")
        lines.append(f"  Bit Depth: {signal.get('bit_depth', 'N/A')}")
        lines.append("")
    
    if 'distribution' in profiles:
        dist = profiles['distribution']
        lines.append("📊 Distribution Profile:")
        lines.append(f"  Entropy: {dist.get('entropy', 0):.2f}")
        lines.append(f"  Color Uniformity: {dist.get('color_uniformity', 0):.3f}")
        lines.append("")
    
    if 'quality' in profiles:
        quality = profiles['quality']
        lines.append("✓ Quality Profile:")
        lines.append(f"  Corruption Rate: {quality.get('corruption_rate', 0):.1%}")
        lines.append(f"  Blur: {quality.get('blur_percentage', 0):.1%}")
        lines.append("")
    
    if 'semantic' in profiles:
        semantic = profiles['semantic']
        lines.append("🧠 Semantic Profile:")
        lines.append(f"  Type: {semantic.get('inferred_type', 'N/A')}")
        lines.append(f"  Confidence: {semantic.get('confidence', 0):.2%}")
    
    return "\n".join(lines)


def format_analysis_pretty(data: Dict[str, Any]) -> str:
    """Format complete analysis into human-readable text"""
    lines = []
    
    if 'fingerprint' in data:
        lines.append(format_fingerprint_pretty(data['fingerprint']))
        lines.append("\n" + "="*50 + "\n")
    
    if 'decisions' in data:
        decisions = data['decisions']
        lines.append("=== Decisions ===\n")
        
        for domain, decision in decisions.items():
            if isinstance(decision, dict):
                lines.append(f"\n{domain.replace('_', ' ').title()}:")
                lines.append(format_decision_pretty(decision))
    
    if 'artifacts' in data:
        lines.append("\n\n=== Generated Artifacts ===")
        for name, path in data['artifacts'].items():
            lines.append(f"  • {name}: {path}")
    
    return "\n".join(lines)


# --------------------- optional UI mounting ---------------------
# Try to serve built React UI from ui/dist first, fallback to templates/static
ui_dist_dir = BASE_DIR.parent / "ui" / "dist"
tpl_dir = BASE_DIR / "templates"
static_dir = BASE_DIR / "static"

templates: Optional[Jinja2Templates] = None

# Priority 1: Serve built React UI from ui/dist
if ui_dist_dir.exists() and (ui_dist_dir / "index.html").exists():
    logger.info(f"Found React UI at {ui_dist_dir}, serving built app...")
    # Serve assets under /assets to avoid catching API endpoints
    assets_dir = ui_dist_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/", response_class=HTMLResponse)
    def ui_index_react(request: Request):
        """Serve built React UI index only"""
        index_html = (ui_dist_dir / "index.html").read_text(encoding="utf-8")
        return index_html

# Priority 2: Fallback to Jinja2 templates
elif tpl_dir.exists() and static_dir.exists():
    logger.info(f"React UI not found, falling back to Jinja2 templates at {tpl_dir}...")
    templates = Jinja2Templates(directory=str(tpl_dir))
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def ui_index(request: Request):
        """
        Render index.html and inject API endpoints & config for client-side convenience.
        The HTML may reference window.IMGSHAPE (injected into the template).
        """
        endpoints = {
            "analyze": app.url_path_for("analyze_image"),
            "recommend": app.url_path_for("recommend_image"),
            "compatibility": app.url_path_for("check_dataset"),
            "datasets": app.url_path_for("list_datasets"),
            "health": app.url_path_for("health"),
            "download_report": app.url_path_for("download_report"),
            # v4 endpoints
            "v4_fingerprint": app.url_path_for("v4_fingerprint") if V4_AVAILABLE else None,
            "v4_analyze": app.url_path_for("v4_analyze") if V4_AVAILABLE else None,
            "v4_info": app.url_path_for("v4_info") if V4_AVAILABLE else None,
        }
        context = {
            "request": request,
            "endpoints": endpoints,
            "version": app.version,
            "dataset_registry": DATASET_REGISTRY,
            "v4_available": V4_AVAILABLE,
        }
        return templates.TemplateResponse("index.html", context)
else:
    @app.get("/", response_class=JSONResponse)
    def root():
        return JSONResponse({"service": "imgshape", "version": app.version})


# --------------------- archive safety helpers ---------------------


def _is_within_directory(directory: Path, target: Path) -> bool:
    try:
        directory = directory.resolve()
        target = target.resolve()
        return str(target).startswith(str(directory))
    except Exception:
        return False


def _safe_extract_zip(zip_path: Path, dest_dir: Path, max_files: int = MAX_EXTRACT_FILES) -> int:
    with zipfile.ZipFile(str(zip_path)) as zf:
        members = zf.namelist()
        if len(members) > max_files:
            raise HTTPException(status_code=400, detail=f"Zip contains too many files ({len(members)} > {max_files})")
        extracted = 0
        for name in members:
            if name.endswith("/") or name.startswith("__MACOSX"):
                continue
            dest_path = dest_dir / Path(name)
            if not _is_within_directory(dest_dir, dest_path):
                raise HTTPException(status_code=400, detail="Zip contains unsafe paths")
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(dest_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1
        return extracted


def _collect_image_paths(root: Path, limit: int = 16) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    if root.is_file() and root.suffix.lower() in exts:
        return [root]
    found: list[Path] = []
    for candidate in root.rglob("*"):
        if candidate.suffix.lower() in exts:
            found.append(candidate)
            if len(found) >= limit:
                break
    return found


def _coerce_bool(val: Optional[Any], default: bool) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes", "on"}
    return bool(val)


def _safe_extract_tar(tar_path: Path, dest_dir: Path, max_files: int = MAX_EXTRACT_FILES) -> int:
    with tarfile.open(str(tar_path)) as tf:
        members = tf.getmembers()
        if len(members) > max_files:
            raise HTTPException(status_code=400, detail=f"Archive contains too many members ({len(members)} > {max_files})")
        extracted = 0
        for member in members:
            if member.isdir():
                continue
            name = member.name
            # skip suspicious names
            if name.startswith("__MACOSX") or name.startswith("."):
                continue
            dest_path = dest_dir / Path(name)
            if not _is_within_directory(dest_dir, dest_path):
                raise HTTPException(status_code=400, detail="Archive contains unsafe paths")
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            f = tf.extractfile(member)
            if f is None:
                continue
            with f, open(dest_path, "wb") as out:
                shutil.copyfileobj(f, out)
            extracted += 1
        return extracted


def _detect_archive_type_from_path(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".zip"):
        return "zip"
    if name.endswith((".tar.gz", ".tgz", ".tar")):
        return "tar"
    if name.endswith((".tar.bz2", ".tbz", ".tbz2")):
        return "tar"
    # fallback: try magic by opening
    try:
        if zipfile.is_zipfile(str(path)):
            return "zip"
    except Exception:
        pass
    try:
        with tarfile.open(str(path)):
            return "tar"
    except Exception:
        pass
    return "unknown"


# --------------------- remote downloader (stream to temp file) ---------------------


def _download_remote_to_file(url: str, max_bytes: int = MAX_REMOTE_BYTES, timeout: int = 60) -> Path:
    logger.info("Downloading remote dataset from %s", url)
    tmpf = Path(tempfile.mkstemp(prefix="imgshape_dl_", suffix="")[1])
    total = 0
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(tmpf, "wb") as out:
                for chunk in r.iter_content(chunk_size=1024 * 64):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        raise HTTPException(status_code=400, detail=f"Remote file exceeds size limit ({max_bytes} bytes)")
                    out.write(chunk)
        return tmpf
    except HTTPException:
        try:
            tmpf.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    except Exception as e:
        try:
            tmpf.unlink(missing_ok=True)
        except Exception:
            pass
        logger.exception("Remote download failed: %s", e)
        raise HTTPException(status_code=502, detail=f"Failed to download remote dataset: {e}")


# --------------------- ensure dataset exists (with registry + extraction) ---------------------


def ensure_local_dataset(dataset_name: str) -> Path:
    dataset_dir = (DATASET_ROOT / dataset_name).resolve()
    if dataset_dir.exists():
        return dataset_dir

    url = DATASET_REGISTRY.get(dataset_name)
    if not url:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found locally and no registry URL configured.")

    tmp_archive = _download_remote_to_file(url)
    try:
        tmp_extract = Path(tempfile.mkdtemp(prefix="imgshape_extract_"))
        try:
            arch_type = _detect_archive_type_from_path(tmp_archive)
            if arch_type == "zip":
                _safe_extract_zip(tmp_archive, tmp_extract)
            elif arch_type == "tar":
                _safe_extract_tar(tmp_archive, tmp_extract)
            else:
                # try both
                try:
                    _safe_extract_zip(tmp_archive, tmp_extract)
                except Exception:
                    _safe_extract_tar(tmp_archive, tmp_extract)

            dataset_dir.mkdir(parents=True, exist_ok=True)
            for child in tmp_extract.iterdir():
                target = dataset_dir / child.name
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                shutil.move(str(child), str(target))
            logger.info("Downloaded & prepared dataset '%s' -> %s", dataset_name, dataset_dir)
            return dataset_dir
        finally:
            try:
                if tmp_extract.exists():
                    shutil.rmtree(tmp_extract)
            except Exception:
                pass
    finally:
        try:
            if tmp_archive.exists():
                tmp_archive.unlink(missing_ok=True)
        except Exception:
            pass


# --------------------- API endpoints ---------------------


@app.post("/analyze", response_class=JSONResponse)
async def analyze_image(
    file: Optional[UploadFile] = File(None), dataset_path: Optional[str] = Form(None)
):
    tmp_file_path: Optional[Path] = None
    try:
        if file is not None:
            contents = await file.read()
            img = Image.open(io.BytesIO(contents)).convert("RGB")
        elif dataset_path:
            candidate = Path(dataset_path)
            if not candidate.is_absolute():
                candidate = DATASET_ROOT / candidate
            candidate = candidate.resolve()
            if not candidate.exists():
                raise HTTPException(status_code=404, detail=f"File not found: {dataset_path}")
            if not _is_within_directory(DATASET_ROOT, candidate) and not str(candidate).startswith(str(DATASET_CACHE)):
                logger.warning("analyze: dataset_path outside DATASET_ROOT: %s", candidate)
                raise HTTPException(status_code=403, detail="Access to provided dataset_path is forbidden")
            img = Image.open(str(candidate)).convert("RGB")
        else:
            raise HTTPException(status_code=400, detail="Provide an uploaded file or dataset_path")
        shape = get_shape(img)
        analysis = analyze_type(img) if analyze_type is not None else {"error": "analyze_unavailable"}
        payload = {"status": "ok", "shape": shape, "analysis": analysis}
        return JSONResponse(make_serializable_response(payload))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("analyze failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if tmp_file_path is not None and tmp_file_path.exists():
            try:
                tmp_file_path.unlink(missing_ok=True)
            except Exception:
                pass


@app.post("/recommend", response_class=JSONResponse)
async def recommend_image(
    file: Optional[UploadFile] = File(None), prefs: Optional[str] = Form(None), dataset_path: Optional[str] = Form(None)
):
    try:
        if file is not None:
            contents = await file.read()
            img = Image.open(io.BytesIO(contents)).convert("RGB")
        elif dataset_path:
            candidate = Path(dataset_path)
            if not candidate.is_absolute():
                candidate = DATASET_ROOT / candidate
            candidate = candidate.resolve()
            if not candidate.exists():
                raise HTTPException(status_code=404, detail=f"File not found: {dataset_path}")
            if not _is_within_directory(DATASET_ROOT, candidate) and not str(candidate).startswith(str(DATASET_CACHE)):
                raise HTTPException(status_code=403, detail="Access to provided dataset_path is forbidden")
            img = Image.open(str(candidate)).convert("RGB")
        else:
            raise HTTPException(status_code=400, detail="Provide an uploaded file or dataset_path")
        user_prefs = [p.strip() for p in prefs.split(",") if p.strip()] if prefs else None
        rec = recommend_preprocessing(img, user_prefs=user_prefs) if recommend_preprocessing is not None else {"error": "recommender_unavailable"}
        payload = {"status": "ok", "recommendations": rec}
        return JSONResponse(make_serializable_response(payload))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("recommend failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/augment", response_class=JSONResponse)
async def get_augmentation_recommendations(
    dataset_path: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    dataset: Optional[UploadFile] = File(None),
    num_to_generate: int = Form(4),
    brightness: float = Form(0.5),
    contrast: float = Form(0.5),
    saturation: float = Form(0.5),
    rotation: float = Form(15),
    # UI flags (new) — optional to keep backward compatibility
    color_jitter: Optional[bool] = Form(None),
    rotate: Optional[bool] = Form(None),
    blur: Optional[bool] = Form(None),
    crop: Optional[bool] = Form(None),
    # Legacy flags
    enable_color_jitter: bool = Form(True),
    enable_rotation: bool = Form(True),
    enable_blur: bool = Form(False),
    enable_crop: bool = Form(False),
):
    """Get augmentation recommendations and generate augmented images"""
    tempdir = None
    try:
        from imgshape.augmentations import AugmentationRecommender, Augmentation, AugmentationPlan
        from imgshape.analyze import analyze_dataset
        try:
            from torchvision import transforms
        except Exception as e:
            logger.warning("torchvision not available, falling back to identity transforms: %s", e)
            transforms = None  # type: ignore
        from PIL import ImageDraw
        
        uploaded = dataset or file
        if uploaded is not None:
            # Handle uploaded file (single image or ZIP)
            contents = await uploaded.read()
            tempdir = Path(tempfile.mkdtemp(prefix="imgshape_aug_"))
            
            # Try to detect if it's a ZIP or a single image
            extracted_dir = tempdir / "extracted"
            extracted_dir.mkdir()
            
            # Try as ZIP first
            try:
                tmp_archive = tempdir / "dataset.zip"
                tmp_archive.write_bytes(contents)
                _safe_extract_zip(tmp_archive, extracted_dir)
                target_path = str(extracted_dir)
            except Exception:
                # If not a ZIP, treat as single image
                try:
                    img = Image.open(io.BytesIO(contents))
                    img_path = extracted_dir / "image.png"
                    img.save(str(img_path))
                    target_path = str(extracted_dir)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Could not process uploaded file: {e}")
        elif dataset_path:
            candidate = Path(dataset_path)
            if not candidate.is_absolute():
                candidate = DATASET_ROOT / candidate
            candidate = candidate.resolve()
            if not candidate.exists():
                raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_path}")
            if not _is_within_directory(DATASET_ROOT, candidate) and not str(candidate).startswith(str(DATASET_CACHE)):
                raise HTTPException(status_code=403, detail="Access to provided dataset_path is forbidden")
            target_path = str(candidate)
        else:
            raise HTTPException(status_code=400, detail="No dataset provided")
        
        # Get base augmentation plan from recommender
        try:
            stats = analyze_dataset(target_path)
        except Exception as e:
            logger.warning(f"analyze_dataset failed: {e}, returning basic stats")
            stats = {"image_count": 0, "error": str(e)}
        
        # Build custom augmentation list
        augmentations = []
        
        color_jitter_flag = color_jitter if color_jitter is not None else enable_color_jitter
        rotate_flag = rotate if rotate is not None else enable_rotation
        blur_flag = blur if blur is not None else enable_blur
        crop_flag = crop if crop is not None else enable_crop

        if color_jitter_flag:
            augmentations.append(Augmentation(
                name="ColorJitter",
                params={
                    "brightness": brightness,
                    "contrast": contrast,
                    "saturation": saturation,
                    "hue": 0.05,
                },
                reason="User-selected color augmentation",
                score=0.85,
            ))
        
        if rotate_flag:
            augmentations.append(Augmentation(
                name="RandomRotation",
                params={"degrees": rotation},
                reason="User-selected rotation augmentation",
                score=0.6,
            ))
        
        if blur_flag:
            augmentations.append(Augmentation(
                name="GaussianBlur",
                params={"kernel_size": 3, "sigma": (0.1, 2.0)},
                reason="User-selected blur augmentation",
                score=0.55,
            ))
        
        if crop_flag:
            augmentations.append(Augmentation(
                name="RandomResizedCrop",
                params={"size": 224, "scale": [0.8, 1.0]},
                reason="User-selected crop augmentation",
                score=0.65,
            ))
        
        # If no augmentations selected, use recommended
        if not augmentations:
            try:
                recommender = AugmentationRecommender()
                aug_plan = recommender.recommend_for_dataset(stats)
                augmentations = aug_plan.augmentations
            except Exception as e:
                logger.warning(f"AugmentationRecommender failed: {e}")
                augmentations = []
        
        # Create augmentation plan
        aug_plan = AugmentationPlan(
            augmentations=augmentations,
            recommended_order=[a.name for a in augmentations],
            seed=42
        )
        
        images: list[Dict[str, Any]] = []

        try:
            image_paths = _collect_image_paths(Path(target_path), limit=max(1, num_to_generate))
        except Exception as e:
            logger.warning("Image collection failed: %s", e)
            image_paths = []

        if not image_paths:
            raise HTTPException(status_code=400, detail="No images found for augmentation")

        # Build torchvision transforms pipeline for generation
        transform_steps = []
        if transforms is not None:
            if color_jitter_flag:
                transform_steps.append(transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=0.05))
            if rotate_flag:
                transform_steps.append(transforms.RandomRotation(degrees=rotation))
            if blur_flag:
                transform_steps.append(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)))
            if crop_flag:
                transform_steps.append(transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)))

        torch_pipeline = transforms.Compose(transform_steps) if transform_steps else None

        for i in range(num_to_generate):
            src = image_paths[i % len(image_paths)]
            img = Image.open(str(src)).convert("RGB")
            if torch_pipeline:
                img = torch_pipeline(img)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            images.append({
                "base64": base64.b64encode(buf.getvalue()).decode("ascii"),
                "label": f"augmented_{i+1}",
            })

        result = {
            "augmentations": aug_plan.as_dict(),
            "dataset_stats": stats,
            "parameters": {
                "num_to_generate": num_to_generate,
                "brightness": brightness,
                "contrast": contrast,
                "saturation": saturation,
                "rotation": rotation,
                "enabled": {
                    "color_jitter": color_jitter_flag,
                    "rotation": rotate_flag,
                    "blur": blur_flag,
                    "crop": crop_flag,
                }
            },
            "message": f"Augmentation plan with {len(augmentations)} transforms generated {len(images)} images",
            "images": images,
        }
        
        return JSONResponse(make_serializable_response(result))
    
    finally:
        if tempdir:
            shutil.rmtree(tempdir, ignore_errors=True)


@app.post("/generate_report", response_class=JSONResponse)
async def generate_report_endpoint(
    request: Request,
    dataset_path: Optional[str] = Form(None),
    dataset: Optional[UploadFile] = File(None),
    format: Optional[str] = Form(None),
    include_metadata: Optional[bool] = Form(None),
    include_charts: Optional[bool] = Form(None),
):
    """Generate a dataset report (markdown/html/pdf). Accepts JSON payloads (UI) or form-data (legacy)."""
    tempdir = None
    analysis_results: Optional[Dict[str, Any]] = None
    try:
        from imgshape.report import generate_markdown_report, generate_html_report
        from imgshape.analyze import analyze_dataset

        payload: Dict[str, Any] = {}
        if request.headers.get("content-type", "").lower().startswith("application/json"):
            try:
                payload = await request.json()
            except Exception:
                payload = {}

        if payload:
            dataset_path = payload.get("dataset_path") or dataset_path
            analysis_results = payload.get("results")
            format = payload.get("format") or format
            include_metadata = payload.get("include_metadata") if "include_metadata" in payload else include_metadata
            include_charts = payload.get("include_charts") if "include_charts" in payload else include_charts

        fmt = (format or "markdown").lower()
        include_metadata = _coerce_bool(include_metadata, True)
        include_charts = _coerce_bool(include_charts, False)
        report_id = uuid.uuid4().hex

        target_path: Optional[str] = None
        stats: Dict[str, Any] = {}

        if dataset is not None:
            contents = await dataset.read()
            tempdir = Path(tempfile.mkdtemp(prefix="imgshape_report_"))

            extracted_dir = tempdir / "extracted"
            extracted_dir.mkdir()

            try:
                tmp_archive = tempdir / "dataset.zip"
                tmp_archive.write_bytes(contents)
                _safe_extract_zip(tmp_archive, extracted_dir)
                target_path = str(extracted_dir)
            except Exception:
                try:
                    img = Image.open(io.BytesIO(contents))
                    img_path = extracted_dir / "image.png"
                    img.save(str(img_path))
                    target_path = str(extracted_dir)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Could not process uploaded file: {e}")
        elif dataset_path:
            candidate = Path(dataset_path)
            if not candidate.is_absolute():
                candidate = DATASET_ROOT / candidate
            candidate = candidate.resolve()
            if not candidate.exists():
                raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_path}")
            if not _is_within_directory(DATASET_ROOT, candidate) and not str(candidate).startswith(str(DATASET_CACHE)):
                raise HTTPException(status_code=403, detail="Access to provided dataset_path is forbidden")
            target_path = str(candidate)

        if target_path:
            try:
                stats = analyze_dataset(target_path)
            except Exception as e:
                logger.warning("analyze_dataset failed: %s", e)
                stats = {"image_count": 0, "error": str(e)}
            analysis_results = analysis_results or stats
        elif analysis_results is None:
            raise HTTPException(status_code=400, detail="No dataset or analysis results provided")

        # Generate report content
        if fmt == "html":
            try:
                out_html = tempdir / "report.html" if tempdir else Path(tempfile.mktemp(suffix=".html"))
                html_content = ""
                if target_path:
                    _ = generate_html_report(target_path, str(out_html), analysis=analysis_results)
                    html_content = out_html.read_text(encoding="utf-8") if out_html.exists() else "<html><body>No report generated</body></html>"
                else:
                    html_content = f"<html><body><pre>{json.dumps(analysis_results, indent=2)}</pre></body></html>"
                return JSONResponse({"id": report_id, "format": "html", "content": html_content})
            except Exception as e:
                logger.error("HTML report generation failed: %s", e)
                return JSONResponse({"id": report_id, "format": "html", "content": f"<html><body>Error generating HTML report: {e}</body></html>"})
        elif fmt == "pdf":
            # Placeholder: PDF rendering not implemented; return HTML content and let client handle download
            html_body = f"<html><body><pre>{json.dumps(analysis_results, indent=2)}</pre></body></html>"
            return JSONResponse({"id": report_id, "format": "pdf", "content": html_body, "url": None})
        else:
            try:
                md_content = ""
                if target_path:
                    out_md = tempdir / "report.md" if tempdir else Path(tempfile.mktemp(suffix=".md"))
                    md_path = generate_markdown_report(target_path, str(out_md), analysis=analysis_results)
                    md_content = Path(md_path).read_text(encoding="utf-8") if Path(md_path).exists() else "# Report\n\nNo report generated"
                else:
                    md_content = "# Report\n\n" + json.dumps({"include_metadata": include_metadata, "include_charts": include_charts, "results": analysis_results}, indent=2)
                return JSONResponse({"id": report_id, "format": "markdown", "content": md_content})
            except Exception as e:
                logger.error("Markdown report generation failed: %s", e)
                return JSONResponse({"id": report_id, "format": "markdown", "content": f"# Error\n\n{e}"})

    finally:
        if tempdir:
            shutil.rmtree(tempdir, ignore_errors=True)


@app.get("/datasets", response_class=JSONResponse)
def list_datasets():
    local = []
    for p in sorted(DATASET_ROOT.iterdir()):
        if p.is_dir():
            local.append({"name": p.name, "path": str(p)})
    registry_list = [{"name": k, "url": v} for k, v in DATASET_REGISTRY.items()]
    return JSONResponse(make_serializable_response({"local": local, "registry": registry_list}))


@app.post("/compatibility", response_class=JSONResponse)
async def check_dataset(
    model: str = Form(...),
    dataset_name: Optional[str] = Form(None),
    dataset_path: Optional[str] = Form(None),
    dataset: Optional[UploadFile] = File(None),
):
    tempdir = None
    tmp_archive = None
    try:
        if dataset is not None:
            contents = await dataset.read()
            tmp_archive = Path(tempfile.mkstemp(prefix="imgshape_upload_", suffix="")[1])
            with open(tmp_archive, "wb") as f:
                f.write(contents)
            tmp_extract = Path(tempfile.mkdtemp(prefix="imgshape_ds_"))
            try:
                arch_type = _detect_archive_type_from_path(tmp_archive)
                if arch_type == "zip":
                    extracted_count = _safe_extract_zip(tmp_archive, tmp_extract)
                elif arch_type == "tar":
                    extracted_count = _safe_extract_tar(tmp_archive, tmp_extract)
                else:
                    try:
                        extracted_count = _safe_extract_zip(tmp_archive, tmp_extract)
                    except Exception:
                        extracted_count = _safe_extract_tar(tmp_archive, tmp_extract)
                logger.info("Extracted %d uploaded files to %s", extracted_count, tmp_extract)
                tempdir = tmp_extract
                dataset_path_to_use = str(tmp_extract)
            finally:
                try:
                    if tmp_archive.exists():
                        tmp_archive.unlink(missing_ok=True)
                except Exception:
                    pass
        elif dataset_name:
            dataset_dir = ensure_local_dataset(dataset_name)
            dataset_path_to_use = str(dataset_dir)
        elif dataset_path:
            candidate = Path(dataset_path)
            if not candidate.is_absolute():
                candidate = DATASET_ROOT / candidate
            candidate = candidate.resolve()
            if not candidate.exists():
                raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_path}")
            if not _is_within_directory(DATASET_ROOT, candidate) and not str(candidate).startswith(str(DATASET_CACHE)):
                raise HTTPException(status_code=403, detail="Access to provided dataset_path is forbidden")
            dataset_path_to_use = str(candidate)
        else:
            raise HTTPException(status_code=400, detail="Provide dataset_name, dataset_path, or upload an archive under 'dataset'")

        p = Path(dataset_path_to_use)
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_path_to_use}")

        report = check_compatibility(model=model, dataset_path=str(p))
        payload = {"status": "ok", "report": report}
        return JSONResponse(make_serializable_response(payload))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("compatibility check failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tempdir is not None and tempdir.exists():
            try:
                shutil.rmtree(tempdir)
                logger.info("Removed temp dataset dir %s", tempdir)
            except Exception as e:
                logger.warning("Failed to remove tempdir %s: %s", tempdir, e)


# --------------------- v4 Atlas endpoints ---------------------
@app.post("/v4/fingerprint", response_class=JSONResponse)
async def v4_fingerprint(
    file: Optional[UploadFile] = File(None),
    dataset_path: Optional[str] = Form(None),
    sample_limit: Optional[int] = Form(None),
    format: str = Form("json"),  # "json" or "pretty"
):
    """
    Extract v4 dataset fingerprint from uploaded file or server path.
    Returns canonical dataset identity with 5 profiles.
    
    format: "json" for structured data, "pretty" for human-readable text
    """
    if not V4_AVAILABLE:
        raise HTTPException(status_code=503, detail="v4 modules not available")
    
    tempdir = None
    try:
        # Determine dataset source
        if file:
            tempdir = Path(tempfile.mkdtemp(prefix="imgshape_v4_"))
            upload_path = tempdir / file.filename
            with open(upload_path, "wb") as f:
                f.write(await file.read())
            dataset_source = upload_path
        elif dataset_path:
            dataset_source = Path(dataset_path)
            if not dataset_source.exists():
                raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_path}")
        else:
            raise HTTPException(status_code=400, detail="Provide either file or dataset_path")
        
        # Extract fingerprint
        extractor = FingerprintExtractor(sample_limit=sample_limit)
        fingerprint = extractor.extract(dataset_source)
        
        # Convert to serializable format
        result = fingerprint.to_dict()
        
        # Format output based on request
        if format == "pretty":
            pretty_text = format_fingerprint_pretty(result)
            return JSONResponse({"format": "pretty", "text": pretty_text, "data": make_serializable_response(result)})
        else:
            return JSONResponse(make_serializable_response(result))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("v4_fingerprint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tempdir and tempdir.exists():
            try:
                shutil.rmtree(tempdir)
            except Exception as e:
                logger.warning("Failed to remove tempdir %s: %s", tempdir, e)


@app.post("/v4/analyze", response_class=JSONResponse)
async def v4_analyze(
    file: Optional[UploadFile] = File(None),
    dataset_path: Optional[str] = Form(None),
    task: str = Form("classification"),
    deployment: str = Form("cloud"),
    priority: str = Form("balanced"),
    sample_limit: Optional[int] = Form(None),
    max_model_size_mb: Optional[float] = Form(None),
    format: str = Form("json"),  # "json" or "pretty"
):
    """
    Complete v4 Atlas analysis: fingerprint + decisions + artifacts.
    Returns fingerprint, decisions, and generates artifacts in temp directory.
    
    format: "json" for structured data, "pretty" for human-readable text
    """
    if not V4_AVAILABLE:
        raise HTTPException(status_code=503, detail="v4 modules not available")
    
    tempdir = None
    try:
        # Determine dataset source
        if file:
            tempdir = Path(tempfile.mkdtemp(prefix="imgshape_v4_"))
            upload_path = tempdir / file.filename
            with open(upload_path, "wb") as f:
                f.write(await file.read())
            dataset_source = upload_path
            output_dir = tempdir / "output"
        elif dataset_path:
            dataset_source = Path(dataset_path)
            if not dataset_source.exists():
                raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_path}")
            tempdir = Path(tempfile.mkdtemp(prefix="imgshape_v4_"))
            output_dir = tempdir / "output"
        else:
            raise HTTPException(status_code=400, detail="Provide either file or dataset_path")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parse user intent
        try:
            intent = UserIntent(
                task=TaskType(task),
                deployment_target=DeploymentTarget(deployment),
                priority=Priority(priority)
            )
        except (KeyError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid parameter: {e}")
        
        # Parse constraints
        constraints = UserConstraints(max_model_size_mb=max_model_size_mb) if max_model_size_mb else None
        
        # Run analysis
        atlas = Atlas(sample_limit=sample_limit)
        result = atlas.analyze(dataset_source, intent, output_dir, constraints)
        
        # Build response
        response_data = {
            "fingerprint": result['fingerprint'].to_dict(),
            "decisions": result['decisions'].to_dict(),
            "artifacts": {
                name: str(path.name) for name, path in result['artifacts'].items()
            }
        }
        
        # Format output based on request
        if format == "pretty":
            pretty_text = format_analysis_pretty(response_data)
            return JSONResponse({"format": "pretty", "text": pretty_text, "data": make_serializable_response(response_data)})
        else:
            return JSONResponse(make_serializable_response(response_data))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("v4_analyze failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tempdir and tempdir.exists():
            try:
                shutil.rmtree(tempdir)
            except Exception as e:
                logger.warning("Failed to remove tempdir %s: %s", tempdir, e)


@app.get("/v4/info", response_class=JSONResponse)
def v4_info():
    """Get v4 availability and capability information"""
    if not V4_AVAILABLE:
        return JSONResponse({
            "available": False,
            "message": "v4 modules not installed"
        })
    
    return JSONResponse({
        "available": True,
        "version": "4.1.0",
        "codename": "Atlas Reinforcement",
        "features": [
            "deterministic_fingerprinting",
            "five_profile_system",
            "rule_based_decisions",
            "deployable_artifacts",
            "gpu_acceleration",
            "drift_analysis",
            "similarity_indexing"
        ],
        "gpu_available": GPUHandler.is_cuda_available(),
        "task_types": [t.value for t in TaskType],
        "deployment_targets": [d.value for d in DeploymentTarget],
        "priorities": [p.value for p in Priority],
    })

@app.post("/v4/compare", response_class=JSONResponse)
async def v4_compare(
    baseline_path: str = Form(...),
    current_path: str = Form(...),
):
    """Compare two datasets and return similarity and drift metrics"""
    if not V4_AVAILABLE:
        raise HTTPException(status_code=503, detail="v4 modules not available")
    
    try:
        atlas = Atlas()
        result = atlas.compare(Path(baseline_path), Path(current_path))
        # result contains drift (DriftScore) and similarity (SimilarityIndex)
        # _to_serializable will handle dataclasses
        return JSONResponse(make_serializable_response(result))
    except Exception as e:
        logger.exception("v4_compare failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v4/drift", response_class=JSONResponse)
async def v4_drift(
    baseline_path: str = Form(...),
    current_path: str = Form(...),
):
    """Alias for compare, focused on drift rationale"""
    return await v4_compare(baseline_path, current_path)

@app.post("/v4/benchmark", response_class=JSONResponse)
async def v4_benchmark(
    dataset_path: str = Form(...),
):
    """Run performance benchmark on a dataset"""
    if not V4_AVAILABLE:
        raise HTTPException(status_code=503, detail="v4 modules not available")
    
    import time
    try:
        start_time = time.time()
        atlas = Atlas()
        fps = atlas.extract_fingerprint(Path(dataset_path))
        duration = time.time() - start_time
        
        return JSONResponse({
            "status": "ok",
            "gpu_used": GPUHandler.is_cuda_available(),
            "duration_seconds": duration,
            "sample_count": fps.metadata.get("sample_count", 0)
        })
    except Exception as e:
        logger.exception("v4_benchmark failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_class=JSONResponse)
def health():
    return JSONResponse({
        "status": "healthy", 
        "version": app.version,
        "v4_available": V4_AVAILABLE
    })


@app.post("/download_report", response_class=JSONResponse)
async def download_report(model: Optional[str] = Form(None), dataset_name: Optional[str] = Form(None), dataset_path: Optional[str] = Form(None)):
    try:
        if model and (dataset_name or dataset_path):
            if dataset_name:
                dataset_dir = ensure_local_dataset(dataset_name)
                report = check_compatibility(model=model, dataset_path=str(dataset_dir))
            else:
                p = Path(dataset_path)
                if not p.exists():
                    raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_path}")
                report = check_compatibility(model=model, dataset_path=str(p))
            payload = {"type": "compatibility_report", "model": model, "report": report}
        else:
            payload = {"type": "meta", "message": "no model/dataset provided"}
        return JSONResponse(make_serializable_response(payload))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("download_report failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# --------------------- debug routes ---------------------
@app.get("/_debug_static_list", response_class=JSONResponse)
def _debug_static_list():
    s = Path(__file__).resolve().parent / "static"
    if not s.exists():
        return JSONResponse({"static_exists": False, "static_dir": str(s)})
    files = []
    for p in sorted(s.rglob("*")):
        if p.is_file():
            files.append(str(p.relative_to(s)))
    return JSONResponse({"static_exists": True, "static_dir": str(s), "files": files})


# --------------------- startup logging ---------------------
@app.on_event("startup")
def _startup():
    logger.info("Starting imgshape API v%s", app.version)
    logger.info("BASE_DIR=%s", BASE_DIR)
    logger.info("DATASET_ROOT=%s DATASET_CACHE=%s", DATASET_ROOT, DATASET_CACHE)
    logger.info("MAX_REMOTE_BYTES=%d MAX_EXTRACT_FILES=%d", MAX_REMOTE_BYTES, MAX_EXTRACT_FILES)
    logger.info("CORS_ORIGINS=%s", CORS_ORIGINS)
