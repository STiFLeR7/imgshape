# service/app.py
"""
FastAPI wrapper for imgshape v2.2.0 — robust archive support

This version extends dataset acquisition to support:
 - .zip
 - .tar
 - .tar.gz / .tgz
 - .tar.bz2

Behavior:
 - When given a dataset_name that maps to a remote URL in DATASET_REGISTRY,
   the server downloads the archive to a temp file (streaming, size-limited),
   then safely extracts it (no path traversal) into the target folder.
 - Uploaded dataset archives (zip/tar/..) are also accepted and extracted safely.
 - Legacy: dataset_path (server-side absolute path) still supported.
 - Temporary extraction directories are cleaned up after use.

Security & limits:
 - Max remote download size is controlled by MAX_REMOTE_BYTES (env override).
 - Max extracted files (per archive) limited via MAX_EXTRACT_FILES to avoid DoS.
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
import tarfile
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
DATASET_ROOT = Path(os.environ.get("IMGSHAPE_DATASET_ROOT", "./datasets")).resolve()
DATASET_CACHE = Path(os.environ.get("IMGSHAPE_DATASET_CACHE", "./.dataset_cache")).resolve()
MAX_REMOTE_BYTES = int(os.environ.get("IMGSHAPE_MAX_REMOTE_BYTES", str(2 * 1024 * 1024 * 1024)))  # 2 GB default
MAX_EXTRACT_FILES = int(os.environ.get("IMGSHAPE_MAX_EXTRACT_FILES", str(100_000)))
try:
    _registry_raw = os.environ.get("IMGSHAPE_DATASET_REGISTRY", "{}")
    DATASET_REGISTRY = json.loads(_registry_raw)
except Exception:
    DATASET_REGISTRY = {}

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

# reduce streamlit noise if present
logging.getLogger("streamlit").setLevel(logging.ERROR)

app = FastAPI(
    title="imgshape API",
    description="FastAPI wrapper for imgshape — analyze, recommend, and compatibility (archive-safe)",
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
    # try tar
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
        # remove partial file
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

    # download archive to temp file
    tmp_archive = _download_remote_to_file(url)
    try:
        # create a temp extraction dir, extract safely, then move to final dataset_dir
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
            # move children of tmp_extract into dataset_dir
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
    try:
        if dataset is not None:
            # uploaded archive: write to temp file then extract based on detected type
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
                    # attempt both
                    try:
                        extracted_count = _safe_extract_zip(tmp_archive, tmp_extract)
                    except Exception:
                        extracted_count = _safe_extract_tar(tmp_archive, tmp_extract)
                logger.info("Extracted %d uploaded files to %s", extracted_count, tmp_extract)
                tempdir = tmp_extract
                dataset_path_to_use = str(tmp_extract)
            finally:
                try:
                    tmp_archive.unlink(missing_ok=True)
                except Exception:
                    pass
        elif dataset_name:
            dataset_dir = ensure_local_dataset(dataset_name)
            dataset_path_to_use = str(dataset_dir)
        elif dataset_path:
            dataset_path_to_use = dataset_path
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
