# service/app.py
"""
FastAPI wrapper for imgshape v2.2.0 — dataset-name driven compatibility

Enhancements in this version:
- Introduces DATASET_ROOT and DATASET_CACHE (configurable via env vars).
- Adds `/datasets` endpoint that lists local datasets and known remote dataset registry.
- `/compatibility` accepts:
    * dataset_name (preferred) — resolves to a local folder under DATASET_ROOT
    * if missing locally and a registry mapping (DATASET_REGISTRY) knows a URL for the name,
      the server will download & extract the dataset into DATASET_ROOT/<dataset_name> (cached)
    * or user can still upload ZIP via `dataset` multipart form
- Safe extraction, path traversal checks, download timeouts and size limit checks included.
- Keeps analyze/recommend endpoints and JSON-serializable responses.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

# imgshape internals
from imgshape.analyze import analyze_type
from imgshape.compatibility import check_compatibility
from imgshape.recommender import recommend_preprocessing, recommend_dataset
from imgshape.shape import get_shape

# -------------------------
# Configuration (env-driven)
# -------------------------
# Root directory where server-side datasets are stored/served from.
DATASET_ROOT = Path(os.environ.get("IMGSHAPE_DATASET_ROOT", "./datasets")).resolve()
# Cache directory for downloads / temporary operations (will be created)
DATASET_CACHE = Path(os.environ.get("IMGSHAPE_DATASET_CACHE", "./.dataset_cache")).resolve()
# Maximum allowed bytes to download for a remote dataset zip (default 2GB)
MAX_REMOTE_BYTES = int(os.environ.get("IMGSHAPE_MAX_REMOTE_BYTES", str(2 * 1024 * 1024 * 1024)))
# Known dataset registry mapping short name -> archive URL (optional)
# Example env var: IMGSHAPE_DATASET_REGISTRY='{"cifar10":"https://.../cifar10.zip"}'
try:
    _registry_raw = os.environ.get("IMGSHAPE_DATASET_REGISTRY", "{}")
    DATASET_REGISTRY = json.loads(_registry_raw)
except Exception:
    DATASET_REGISTRY = {}

# make sure directories exist
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

# reduce noise from non-essential libs (streamlit)
logging.getLogger("streamlit").setLevel(logging.ERROR)

app = FastAPI(
    title="imgshape API",
    description="FastAPI wrapper for imgshape — analyze, recommend, and compatibility (dataset-name driven)",
    version="2.2.0",
)


# --------------------- serialization helpers ---------------------


def _to_serializable_scalar(x: Any) -> Any:
    try:
        import numpy as np
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
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    try:
        import numpy as np
    except Exception:
        np = None

    if np is not None and isinstance(obj, np.ndarray):
        try:
            return obj.tolist()
        except Exception:
            return [_to_serializable_scalar(x) for x in obj.flatten().tolist()]

    if np is not None and isinstance(obj, (np.integer, np.floating, np.bool_, np.generic)):
        return _to_serializable_scalar(obj)

    if isinstance(obj, (bytes, bytearray)):
        try:
            return base64.b64encode(obj).decode("ascii")
        except Exception:
            return str(obj)

    if isinstance(obj, Image.Image):
        try:
            w, h = obj.size
            buf = io.BytesIO()
            obj.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return {"_pil_image": True, "width": w, "height": h, "mode": obj.mode, "preview_base64": f"data:image/png;base64,{b64}"}
        except Exception:
            return {"_pil_image": True, "info": str(obj)}

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


# --------------------- optional UI mounting ---------------------
tpl_dir = Path(__file__).resolve().parent / "templates"
static_dir = Path(__file__).resolve().parent / "static"
if tpl_dir.exists() and static_dir.exists():
    templates = Jinja2Templates(directory=str(tpl_dir))
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def ui_index(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})
else:
    @app.get("/", response_class=JSONResponse)
    def root():
        return JSONResponse({"service": "imgshape", "version": "2.2.0"})


# --------------------- safe zip helpers ---------------------


def _is_within_directory(directory: Path, target: Path) -> bool:
    try:
        directory = directory.resolve()
        target = target.resolve()
        return str(target).startswith(str(directory))
    except Exception:
        return False


def _safe_extract_zip_bytes(zip_bytes: bytes, dest_dir: Path, max_files: int = 5000) -> int:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
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


# --------------------- remote downloader (safe streaming) ---------------------


def _download_remote_to_bytes(url: str, max_bytes: int = MAX_REMOTE_BYTES, timeout: int = 60) -> bytes:
    """Download a file via streaming into memory (bounded by max_bytes)."""
    logger.info("Downloading remote dataset from %s", url)
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            total = 0
            chunks = []
            for chunk in r.iter_content(chunk_size=1024 * 64):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(status_code=400, detail=f"Remote file exceeds size limit ({max_bytes} bytes)")
                chunks.append(chunk)
            return b"".join(chunks)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Remote download failed: %s", e)
        raise HTTPException(status_code=502, detail=f"Failed to download remote dataset: {e}")


# --------------------- helper: ensure dataset exists locally ---------------------


def ensure_local_dataset(dataset_name: str) -> Path:
    """
    Ensure DATASET_ROOT/dataset_name exists. If not present:
      - If DATASET_REGISTRY has a URL for dataset_name, download & extract into DATASET_ROOT/dataset_name
      - Otherwise raises HTTPException(404)
    Returns the Path to the dataset folder.
    """
    dataset_dir = (DATASET_ROOT / dataset_name).resolve()
    if dataset_dir.exists():
        return dataset_dir

    # try registry
    url = DATASET_REGISTRY.get(dataset_name)
    if not url:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found locally and no registry URL configured.")

    # create target dir
    dataset_dir.mkdir(parents=True, exist_ok=True)
    # download & extract into a temp dir first
    tmp = Path(tempfile.mkdtemp(prefix="imgshape_dl_"))
    try:
        data = _download_remote_to_bytes(url)
        extracted_count = _safe_extract_zip_bytes(data, tmp)
        # move extracted files into dataset_dir atomically (move children)
        for child in tmp.iterdir():
            target = dataset_dir / child.name
            if target.exists():
                # if name collision, remove target then move
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(child), str(target))
        logger.info("Downloaded & prepared dataset '%s' (%d files) -> %s", dataset_name, extracted_count, dataset_dir)
        return dataset_dir
    finally:
        try:
            if tmp.exists():
                shutil.rmtree(tmp)
        except Exception:
            pass


# --------------------- API endpoints ---------------------


@app.post("/analyze", response_class=JSONResponse)
async def analyze_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        shape = get_shape(img)
        analysis = analyze_type(img) if analyze_type is not None else {"error": "analyze_unavailable"}
        payload = {"status": "ok", "shape": shape, "analysis": analysis}
        return JSONResponse(make_serializable_response(payload))
    except Exception as e:
        logger.exception("analyze failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/recommend", response_class=JSONResponse)
async def recommend_image(file: UploadFile = File(...), prefs: Optional[str] = Form(None)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        user_prefs = [p.strip() for p in prefs.split(",") if p.strip()] if prefs else None
        rec = recommend_preprocessing(img, user_prefs=user_prefs) if recommend_preprocessing is not None else {"error": "recommender_unavailable"}
        payload = {"status": "ok", "recommendations": rec}
        return JSONResponse(make_serializable_response(payload))
    except Exception as e:
        logger.exception("recommend failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/datasets", response_class=JSONResponse)
def list_datasets():
    """
    Return a JSON with:
      - local: list of dataset folder names under DATASET_ROOT
      - registry: known remote datasets from DATASET_REGISTRY
    """
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
    """
    Compatibility check with flexible dataset acquisition:

      - If `dataset` (zip) is uploaded, it's extracted to a temp dir and used.
      - Else if `dataset_name` is provided, it resolves to DATASET_ROOT/dataset_name (and may auto-download via registry).
      - Else if `dataset_path` (absolute/server path) is provided, use it (legacy behavior).
      - Otherwise 400.

    Returns sanitized JSON report.
    """
    tempdir = None
    try:
        if dataset is not None:
            # uploaded zip
            filename = getattr(dataset, "filename", "") or ""
            contents = await dataset.read()
            # basic check for zip signature
            try:
                zipfile.ZipFile(io.BytesIO(contents)).close()
            except Exception:
                raise HTTPException(status_code=400, detail="Uploaded dataset must be a ZIP archive")
            tempdir = Path(tempfile.mkdtemp(prefix="imgshape_ds_"))
            extracted_count = _safe_extract_zip_bytes(contents, tempdir)
            logger.info("Extracted %d files to %s from upload", extracted_count, tempdir)
            dataset_path_to_use = str(tempdir)
        elif dataset_name:
            # dataset name resolved under DATASET_ROOT (and auto-download if registry known)
            dataset_dir = ensure_local_dataset(dataset_name)
            dataset_path_to_use = str(dataset_dir)
        elif dataset_path:
            # legacy direct path
            dataset_path_to_use = dataset_path
        else:
            raise HTTPException(status_code=400, detail="Provide dataset_name, dataset_path, or upload a dataset zip under 'dataset'")

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


@app.get("/health", response_class=JSONResponse)
def health():
    return JSONResponse({"status": "healthy"})


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
