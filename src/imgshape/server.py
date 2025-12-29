# src/imgshape/server.py
"""
FastAPI server for imgshape v3

Endpoints:
- POST /analyze           -> analyze a dataset or single image
- POST /recommend         -> recommend preprocessing (accepts analysis or dataset_path)
- POST /apply             -> apply a pipeline (background job)
- POST /snapshot/save     -> save dataset snapshot (analysis)
- POST /snapshot/diff     -> diff two snapshot payloads
- GET  /plugins           -> list discovered plugins
- GET  /profiles          -> list available profiles
- GET  /health            -> quick health check

Run:
    python -m uvicorn imgshape.server:app --reload
Or call run_server(host, port) in scripts.
"""
from __future__ import annotations
import os
import json
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import traceback
import uuid
import time

logger = logging.getLogger("imgshape.server")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

app = FastAPI(title="imgshape-server", version="0.1")

# Defensive imports for optional heavy modules
try:
    from imgshape.analyze import analyze_dataset, analyze_type
except Exception:
    analyze_dataset = None
    analyze_type = None

try:
    from imgshape.recommender import RecommendEngine, recommend_dataset, recommend_preprocessing
except Exception:
    RecommendEngine = None
    recommend_dataset = None
    recommend_preprocessing = None

try:
    from imgshape.pipeline import RecommendationPipeline
except Exception:
    RecommendationPipeline = None

try:
    from imgshape.plugins import load_plugins_from_dir
except Exception:
    load_plugins_from_dir = None

# Locations
ROOT = Path(__file__).resolve().parent
PLUGINS_DIR = ROOT.joinpath("plugins")
PROFILES_DIR = ROOT.joinpath("profiles")
SNAPSHOT_DIR = ROOT.joinpath("snapshots")
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# Simple in-memory job registry for background tasks
_JOB_REGISTRY: Dict[str, Dict[str, Any]] = {}


# -----------------------
# Pydantic models
# -----------------------
class AnalyzeRequest(BaseModel):
    dataset_path: Optional[str] = None  # either dataset_path or image_bytes_base64
    image_bytes_base64: Optional[str] = None


class RecommendRequest(BaseModel):
    analysis: Optional[Dict[str, Any]] = None
    dataset_path: Optional[str] = None
    profile: Optional[str] = None
    user_prefs: Optional[List[str]] = None


class ApplyRequest(BaseModel):
    pipeline_json: Dict[str, Any]
    input_dir: str
    output_dir: str
    dry_run: Optional[bool] = True


class SnapshotSaveRequest(BaseModel):
    dataset_path: str
    out_path: Optional[str] = None  # optional override


class SnapshotDiffRequest(BaseModel):
    old_snapshot: Dict[str, Any]
    new_snapshot: Dict[str, Any]


# -----------------------
# Helpers
# -----------------------
def _safe_analyze(dataset_path: Optional[str] = None, image_bytes_base64: Optional[str] = None) -> Dict[str, Any]:
    """
    Try to run analyze_dataset (dir) or analyze_type (single). Fall back to minimal stats.
    """
    try:
        if dataset_path and analyze_dataset:
            return analyze_dataset(dataset_path)
        if image_bytes_base64 and analyze_type:
            import base64
            from io import BytesIO
            b = base64.b64decode(image_bytes_base64)
            return analyze_type(BytesIO(b))
    except Exception as e:
        logger.exception("analyze failed: %s", e)
    # fallback: minimal snapshot (file count)
    try:
        if dataset_path:
            p = Path(dataset_path)
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
            count = sum(1 for _ in p.rglob("*") if _.suffix.lower() in exts)
            return {"image_count": count}
    except Exception:
        logger.exception("fallback analyze failed")
    return {"image_count": 0, "note": "fallback"}


def _safe_recommend(analysis: Optional[Dict[str, Any]] = None, dataset_path: Optional[str] = None, profile: Optional[str] = None, user_prefs: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Try RecommendEngine or recommend_preprocessing/recommend_dataset; fallback heuristics.
    """
    try:
        if RecommendEngine:
            engine = RecommendEngine(profile=profile)
            if analysis:
                rec = engine.recommend_from_analysis(analysis)
            elif dataset_path:
                # prefer recommend_dataset if available
                if recommend_dataset:
                    rec = recommend_dataset(dataset_path)
                else:
                    # analyze then recommend
                    a = _safe_analyze(dataset_path=dataset_path)
                    rec = engine.recommend_from_analysis(a)
            else:
                rec = engine.recommend_from_analysis(analysis or {})
            # coerce to dict if needed
            try:
                return rec.as_dict() if hasattr(rec, "as_dict") else dict(rec)
            except Exception:
                return rec if isinstance(rec, dict) else {"preprocessing": rec}
        # fallback to recommend_preprocessing if present (single-image)
        if analysis and recommend_preprocessing:
            return recommend_preprocessing(analysis)
    except Exception:
        logger.exception("Recommend engine failed; falling back.")

    # basic fallback heuristics
    try:
        if profile:
            # try to load profile YAML
            import yaml
            pfile = PROFILES_DIR.joinpath(profile)
            if pfile.exists():
                try:
                    doc = yaml.safe_load(pfile.read_text(encoding="utf-8"))
                    return {"preprocessing": doc.get("preprocessing", []), "augmentations": doc.get("augmentations", []), "meta": {"source": "profile"}}
                except Exception:
                    logger.exception("failed parsing profile yaml")
        # generic fallback
        return {"preprocessing": [{"name": "resize", "spec": {"resize": [256, 256]}}], "augmentations": [], "meta": {"source": "fallback"}}
    except Exception:
        return {"preprocessing": [], "augmentations": [], "meta": {"error": "fallback_failed"}}


def _save_snapshot(dataset_path: str, out_path: Optional[str] = None) -> str:
    snap = _safe_analyze(dataset_path=dataset_path)
    fname = out_path or str(SNAPSHOT_DIR.joinpath(f"snapshot_{int(time.time())}.json"))
    Path(fname).parent.mkdir(parents=True, exist_ok=True)
    Path(fname).write_text(json.dumps(snap, indent=2), encoding="utf-8")
    return fname


def _diff_snapshots(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    diffs = {}
    keys = set(list(old.keys()) + list(new.keys()))
    for k in keys:
        if old.get(k) != new.get(k):
            diffs[k] = {"old": old.get(k), "new": new.get(k)}
    return diffs


def _discover_plugins() -> List[Dict[str, Any]]:
    if not load_plugins_from_dir:
        return []
    try:
        plugins = load_plugins_from_dir(str(PLUGINS_DIR))
        out = []
        for p in plugins:
            out.append({"name": getattr(p, "NAME", p.__class__.__name__), "class": f"{p.__class__.__module__}.{p.__class__.__name__}"})
        return out
    except Exception:
        logger.exception("plugin discovery failed")
        return []


def _list_profiles() -> List[str]:
    if not PROFILES_DIR.exists():
        return []
    return [p.name for p in sorted(PROFILES_DIR.glob("*.yaml"))]


# -----------------------
# Endpoints
# -----------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "version": app.version if hasattr(app, "version") else "0.1",
        "v4_available": False,
        "plugins_dir": str(PLUGINS_DIR),
        "profiles_dir": str(PROFILES_DIR),
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    try:
        res = _safe_analyze(dataset_path=req.dataset_path, image_bytes_base64=req.image_bytes_base64)
        return res
    except Exception as e:
        logger.exception("analyze endpoint error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend")
def recommend(req: RecommendRequest):
    try:
        rec = _safe_recommend(analysis=req.analysis, dataset_path=req.dataset_path, profile=req.profile, user_prefs=req.user_prefs)
        return rec
    except Exception as e:
        logger.exception("recommend endpoint error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/apply")
def apply_pipeline(req: ApplyRequest, bg: BackgroundTasks):
    """
    Apply a pipeline JSON to a folder. Pipeline is run in background as a best-effort job.
    Returns a job_id to poll for status (simple in-memory registry).
    """
    job_id = str(uuid.uuid4())
    _JOB_REGISTRY[job_id] = {"status": "queued", "created_at": time.time(), "detail": None}

    def _job(pipeline_json: Dict[str, Any], input_dir: str, output_dir: str, dry_run: bool, job_id_local: str):
        try:
            _JOB_REGISTRY[job_id_local]["status"] = "running"
            if RecommendationPipeline:
                pipeline = RecommendationPipeline.from_recommender_output(pipeline_json)
            else:
                # minimal conversion
                steps = []
                for p in pipeline_json.get("preprocessing", []):
                    steps.append(p)
                pipeline = None

            if pipeline is None:
                # fallback: walk and simulate
                count = 0
                for _ in Path(input_dir).rglob("*"):
                    count += 1
                _JOB_REGISTRY[job_id_local]["status"] = "finished"
                _JOB_REGISTRY[job_id_local]["detail"] = {"processed_count": count, "note": "fallback_no_pipeline"}
                return

            # attempt to apply
            pipeline.apply(input_dir, output_dir, dry_run=dry_run)
            _JOB_REGISTRY[job_id_local]["status"] = "finished"
            _JOB_REGISTRY[job_id_local]["detail"] = {"processed": True, "dry_run": bool(dry_run)}
        except Exception as e:
            logger.exception("pipeline job failed: %s", e)
            _JOB_REGISTRY[job_id_local]["status"] = "failed"
            _JOB_REGISTRY[job_id_local]["detail"] = {"error": str(e), "trace": traceback.format_exc()}

    bg.add_task(_job, req.pipeline_json, req.input_dir, req.output_dir, bool(req.dry_run), job_id)
    return {"job_id": job_id, "status": _JOB_REGISTRY[job_id]["status"]}


@app.get("/job/{job_id}")
def job_status(job_id: str):
    return _JOB_REGISTRY.get(job_id, {"error": "not_found"})


@app.post("/snapshot/save")
def snapshot_save(req: SnapshotSaveRequest):
    try:
        path = _save_snapshot(req.dataset_path, req.out_path)
        return {"snapshot_path": path}
    except Exception as e:
        logger.exception("snapshot save failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/snapshot/diff")
def snapshot_diff(req: SnapshotDiffRequest):
    try:
        diffs = _diff_snapshots(req.old_snapshot, req.new_snapshot)
        return {"diff": diffs}
    except Exception as e:
        logger.exception("snapshot diff failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/plugins")
def plugins_list():
    try:
        return {"plugins": _discover_plugins()}
    except Exception as e:
        logger.exception("plugins list failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/profiles")
def profiles_list():
    try:
        return {"profiles": _list_profiles()}
    except Exception as e:
        logger.exception("profiles list failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------
# Helper runner
# -----------------------
def run_server(host: str = "127.0.0.1", port: int = 8008):
    """
    Convenience runner for local development. Uses uvicorn if available,
    otherwise raises helpful message.
    """
    try:
        import uvicorn
    except Exception as e:
        logger.error("uvicorn not installed; please run: pip install uvicorn[standard]")
        raise RuntimeError("uvicorn not installed") from e

    logger.info("Starting imgshape server on %s:%s", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


# If run as a module: python -m imgshape.server
if __name__ == "__main__":
    try:
        run_server()
    except Exception as e:
        logger.exception("server failed to start: %s", e)
