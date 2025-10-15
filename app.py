# app.py — Streamlit front-end for imgshape v3.0.0 (updated)
# Place this file at the repo root and run: streamlit run app.py
from __future__ import annotations
import streamlit as st
from pathlib import Path
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import json
import logging
import traceback
import random
import tempfile
import textwrap
from typing import Any, Dict, Optional, Tuple

# defensive optional imports
try:
    import yaml
except Exception:
    yaml = None

# local imgshape imports (defensive)
from imgshape.shape import get_shape
from imgshape.analyze import analyze_type, analyze_dataset  # analyze_dataset may be heavy
from imgshape.augmentations import AugmentationRecommender
from imgshape.viz import plot_shape_distribution

# pipeline/plugins may be optional in some installs — import defensively
try:
    from imgshape.pipeline import RecommendationPipeline, PipelineStep
except Exception:
    RecommendationPipeline = None
    PipelineStep = None

try:
    from imgshape.plugins import load_plugins_from_dir
except Exception:
    load_plugins_from_dir = None

# small module-level import (should be cheap)
from imgshape import __version__  # type: ignore

# logging setup
logger = logging.getLogger("imgshape.streamlit.v3")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

st.set_page_config(page_title=f"imgshape v3 (Aurora) — {__version__}", layout="wide")
st.title("🖼️ imgshape — Dataset Intelligence (v3.0.0 • Aurora)")

st.markdown(
    "Interactive dataset assistant — analyze, recommend, preview augmentations, export executable pipelines, "
    "and run lightweight local operations. This UI is defensive: optional features degrade gracefully."
)

# ---------------------------
# Sidebar: global inputs
# ---------------------------
st.sidebar.header("📂 Dataset & Input")
uploaded_file = st.sidebar.file_uploader(
    "Upload a single image (preview)",
    type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
)
dataset_path = st.sidebar.text_input("Dataset folder path (local)", value="assets/sample_images")

# profiles/plugins live next to package in src/imgshape/
REPO_ROOT = Path(__file__).resolve().parent
profiles_dir = REPO_ROOT.joinpath("src", "imgshape", "profiles")
plugins_dir = REPO_ROOT.joinpath("src", "imgshape", "plugins")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Quick actions")
profile_files = []
if profiles_dir.exists():
    profile_files = sorted([p.name for p in profiles_dir.glob("*.yaml")])
selected_profile = st.sidebar.selectbox("Profile (preset)", options=["(none)"] + profile_files, index=0)
st.sidebar.text("Profiles loaded from: " + str(profiles_dir) if profile_files else "No profiles found")

# Plugin discovery
st.sidebar.markdown("---")
st.sidebar.header("🔌 Plugins")
plugins = []
try:
    if plugins_dir.exists() and load_plugins_from_dir is not None:
        plugins = load_plugins_from_dir(str(plugins_dir))
        st.sidebar.write(f"Discovered {len(plugins)} plugins")
        for p in plugins:
            st.sidebar.write(f"- {getattr(p, 'NAME', p.__class__.__name__)}")
    elif plugins_dir.exists():
        st.sidebar.info("Plugins folder exists but plugin loader unavailable (optional dep missing).")
    else:
        st.sidebar.info("No plugins folder detected.")
except Exception as e:
    st.sidebar.error(f"Plugin load failed: {e}")
    logger.exception("Plugin loading error: %s", e)

# ---------------------------
# Utility helpers
# ---------------------------
def cache_uploaded_bytes():
    """Cache bytes in session_state to avoid re-reading the Upload multiple times."""
    if uploaded_file is None:
        return None
    if "uploaded_bytes" not in st.session_state:
        try:
            st.session_state["uploaded_bytes"] = uploaded_file.read()
        except Exception as e:
            st.session_state["uploaded_bytes"] = None
            st.error(f"Error reading upload: {e}")
            logger.exception("file_uploader read failed")
            return None
    return st.session_state["uploaded_bytes"]


def load_image_from_bytes(bytes_data: Optional[bytes]) -> Tuple[Optional[Image.Image], Optional[BytesIO]]:
    """Return (PIL.Image, BytesIO) or (None, None) on failure."""
    if not bytes_data:
        return None, None
    try:
        buf = BytesIO(bytes_data)
        img = Image.open(buf).convert("RGB")
        buf.seek(0)
        return img, buf
    except UnidentifiedImageError:
        return None, None
    except Exception as e:
        st.error(f"Unexpected error opening image: {e}")
        logger.exception("open image error: %s", e)
        return None, None


def load_profile_yaml(profile_name: str) -> Optional[Dict[str, Any]]:
    if profile_name in ("", "(none)"):
        return None
    p = profiles_dir.joinpath(profile_name)
    if not p.exists():
        return None
    if yaml is None:
        st.warning("PyYAML not installed — cannot parse profiles. Install `pyyaml` for full profile support.")
        return None
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception as e:
        st.error(f"Failed to load profile {profile_name}: {e}")
        logger.exception("profile load error")
        return None


def recommend_from_buffer_or_pil(buf_or_pil: Any, user_prefs: Optional[list] = None, profile: Optional[dict] = None) -> Dict[str, Any]:
    """
    Try to call RecommendEngine if available; otherwise fallback to heuristics.
    Returns a dict: {'preprocessing':..., 'augmentations':..., 'meta':...}
    """
    # Try RecommendEngine path (defensive)
    try:
        from imgshape.recommender import RecommendEngine  # type: ignore

        engine = RecommendEngine(profile=profile if profile else None)
        # if we got a BytesIO/bytes prefer engine.recommend_from_bytes if available
        try:
            if isinstance(buf_or_pil, (bytes, bytearray)):
                if hasattr(engine, "recommend_from_bytes"):
                    rec = engine.recommend_from_bytes(bytes(buf_or_pil))
                else:
                    analysis = analyze_type(BytesIO(buf_or_pil))
                    rec = engine.recommend_from_analysis(analysis)
            elif isinstance(buf_or_pil, BytesIO):
                analysis = analyze_type(buf_or_pil)
                rec = engine.recommend_from_analysis(analysis)
            elif isinstance(buf_or_pil, Image.Image):
                analysis = analyze_type(buf_or_pil)
                rec = engine.recommend_from_analysis(analysis)
            elif isinstance(buf_or_pil, (str, Path)):
                # treat as dataset path
                analysis = analyze_dataset(str(buf_or_pil))
                rec = engine.recommend_from_analysis(analysis)
            else:
                rec = engine.recommend_from_analysis({})
        except Exception:
            rec = engine.recommend_from_analysis({})
        # coerce to dict
        if hasattr(rec, "as_dict"):
            try:
                rec = rec.as_dict()
            except Exception:
                rec = dict(rec) if isinstance(rec, dict) else {"preprocessing": [], "augmentations": [], "meta": {"source": "engine"}}
        return rec
    except Exception as e:
        logger.debug("RecommendEngine path failed: %s", e, exc_info=True)

    # fallback heuristics (simple, deterministic)
    try:
        fallback = {"preprocessing": [], "augmentations": [], "meta": {"source": "fallback"}}
        if profile and isinstance(profile, dict):
            fallback["preprocessing"] = profile.get("preprocessing", []) or []
            fallback["augmentations"] = profile.get("augmentations", []) or []
            fallback["meta"]["source"] = "profile"
            return fallback

        # basic default: resize + flip
        fallback["preprocessing"].append({"name": "resize", "spec": {"resize": [256, 256]}})
        fallback["augmentations"].append({"name": "RandomHorizontalFlip", "spec": {"p": 0.5}})
        return fallback
    except Exception as e2:
        logger.exception("Fallback recommend failed: %s", e2)
        return {"preprocessing": [], "augmentations": [], "meta": {"error": str(e2)}}


# pipeline helpers — defensive in case RecommendationPipeline / PipelineStep are missing
def pipeline_from_recommendation(rec: Dict[str, Any]):
    """
    Return a pipeline-like object with `.steps` and `.as_dict()` and `.apply()` semantics.
    If RecommendationPipeline is available, use it; otherwise supply a minimal fallback.
    """
    # If real pipeline class available, prefer it
    try:
        if RecommendationPipeline is not None:
            # try factory method if exists
            if hasattr(RecommendationPipeline, "from_recommender_output"):
                return RecommendationPipeline.from_recommender_output(rec)
            # otherwise try constructor
            steps = []
            if PipelineStep is not None:
                for p in rec.get("preprocessing", []):
                    steps.append(PipelineStep(name=p.get("name", "pre"), spec=p.get("spec", p)))
                for a in rec.get("augmentations", []):
                    steps.append(PipelineStep(name=a.get("name", "aug"), spec=a.get("spec", a)))
                return RecommendationPipeline(steps=steps, meta=rec.get("meta", {}))
    except Exception:
        logger.exception("RecommendationPipeline.from_recommender_output failed")

    # Fallback lightweight pipeline object
    class _Step:
        def __init__(self, name: str, spec: dict):
            self.name = name
            self.spec = spec

        def apply(self, img: Image.Image) -> Image.Image:
            # best-effort: try common simple operations; otherwise return image unchanged
            try:
                # resize spec
                if isinstance(self.spec, dict) and "resize" in self.spec:
                    size = self.spec.get("resize")
                    if isinstance(size, (list, tuple)) and len(size) == 2:
                        return img.resize((int(size[0]), int(size[1])))
                # flip
                if self.name.lower().find("flip") >= 0 and self.spec.get("p", 1.0) > 0:
                    return img.transpose(Image.FLIP_LEFT_RIGHT)
            except Exception:
                logger.debug("Fallback step apply failed for %s", self.name, exc_info=True)
            return img

        def as_dict(self):
            return {"name": self.name, "spec": self.spec}

    class _Pipeline:
        def __init__(self, steps):
            self.steps = steps
            self.meta = rec.get("meta", {})

        def as_dict(self):
            return {"steps": [s.as_dict() for s in self.steps], "meta": self.meta}

        def export(self, format="torchvision"):
            # naive export: JSON pretty-print
            return json.dumps(self.as_dict(), indent=2)

        def save(self, path: str):
            Path(path).write_text(self.export(format="json"), encoding="utf-8")

        def apply(self, input_dir: str, output_dir: str, dry_run: bool = True):
            # dry-run: just iterate files and log actions
            input_path = Path(input_dir)
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            actions = []
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
            for p in sorted(input_path.rglob("*")):
                if not p.is_file() or p.suffix.lower() not in exts:
                    continue
                actions.append({"src": str(p), "dst": str(out_path.joinpath(p.name)), "steps": [s.name for s in self.steps]})
            logger.info("Pipeline apply (dry_run=%s) simulated actions: %d files", bool(dry_run), len(actions))
            return actions

    steps = []
    for p in rec.get("preprocessing", []):
        steps.append(_Step(name=p.get("name", "pre"), spec=p.get("spec", p)))
    for a in rec.get("augmentations", []):
        steps.append(_Step(name=a.get("name", "aug"), spec=a.get("spec", a)))
    return _Pipeline(steps=steps)


def export_pipeline_snippet(pipeline_obj, fmt: str = "torchvision") -> Tuple[str, str]:
    """
    Export pipeline as code snippet (or JSON) and produce a filename suggestion.
    """
    try:
        if hasattr(pipeline_obj, "export"):
            snippet = pipeline_obj.export(format=fmt)
        elif hasattr(pipeline_obj, "as_dict"):
            if fmt == "torchvision":
                # naive textual representation
                snippet = "# torchvision-compatible pipeline export unavailable; falling back to JSON\n"
                snippet += json.dumps(pipeline_obj.as_dict(), indent=2)
            else:
                snippet = json.dumps(pipeline_obj.as_dict(), indent=2)
        else:
            snippet = repr(pipeline_obj)
    except Exception as e:
        logger.exception("Pipeline export failed: %s", e)
        snippet = f"# Export failed: {e}\n\n{repr(pipeline_obj)}"
    fname = f"imgshape_pipeline.{ 'py' if fmt == 'torchvision' else ('yaml' if fmt == 'yaml' else 'json') }"
    return snippet, fname


# ---------------------------
# Persist last recommendation across tabs
# ---------------------------
if "last_rec" not in st.session_state:
    st.session_state["last_rec"] = None


# ---------------------------
# Tabs
# ---------------------------
tabs = st.tabs(["📐 Shape", "🔍 Analyze", "🧠 Recommend", "🎨 Augment Visualizer", "📄 Reports", "🔗 Pipeline Export"])

# ---------------------------
# SHAPE TAB
# ---------------------------
with tabs[0]:
    st.subheader("📐 Shape Detection")
    bytes_data = cache_uploaded_bytes()
    if bytes_data:
        pil_img, buf = load_image_from_bytes(bytes_data)
        if pil_img is None:
            st.error("Uploaded file is not a valid image.")
        else:
            st.image(pil_img, caption="Uploaded Image", width='stretch')
            try:
                shape = get_shape(pil_img)
                st.json({"shape": shape})
            except Exception as e:
                st.error(f"Error detecting shape: {e}")
                logger.exception("shape detection error")
    else:
        st.info("Upload an image to inspect its shape.")

# ---------------------------
# ANALYZE TAB
# ---------------------------
with tabs[1]:
    st.subheader("🔍 Analyze — Image & Dataset")
    bytes_data = cache_uploaded_bytes()
    pil_img, buf = load_image_from_bytes(bytes_data) if bytes_data else (None, None)

    if bytes_data or pil_img:
        try:
            # prefer analyze_type on bytes/PIL
            if bytes_data:
                analysis = analyze_type(BytesIO(bytes_data) if isinstance(bytes_data, (bytes, bytearray)) else bytes_data)
            else:
                analysis = analyze_type(pil_img)
            st.json(analysis)
        except Exception as e:
            logger.exception("analyze_type error: %s", e)
            st.error(f"analyze_type failed: {e}")
    else:
        st.info("Upload an image to run lightweight analysis.")

    st.markdown("### Dataset-level analysis")
    if st.button("Run dataset analysis (may be slow)"):
        try:
            res = analyze_dataset(dataset_path)
            st.success("Analysis complete")
            st.json(res)
        except Exception as e:
            st.error(f"Dataset analysis failed: {e}")
            logger.exception("dataset analysis failed")

    st.markdown("### Quick visual: shape distribution")
    if st.button("Plot shape distribution"):
        try:
            out = plot_shape_distribution(dataset_path, save=False)
            st.success("Plotting triggered (server-side). Check logs for the output path.")
        except Exception as e:
            st.error(f"Plot failed: {e}")
            logger.exception("plot failed")

# ---------------------------
# RECOMMEND TAB
# ---------------------------
with tabs[2]:
    st.subheader("🧠 Recommend — Preprocessing & Augmentation")
    bytes_data = cache_uploaded_bytes()
    pil_img, buf = load_image_from_bytes(bytes_data) if bytes_data else (None, None)
    profile_yaml = load_profile_yaml(selected_profile)

    st.markdown("**User preferences (optional):** comma-separated tokens (e.g. `preserve_aspect,low_res`) ")
    user_prefs_input = st.text_input("Prefs", value="")
    user_prefs = [p.strip() for p in user_prefs_input.split(",") if p.strip()] if user_prefs_input else None

    recommend_clicked = False
    if st.button("Recommend from input"):
        recommend_clicked = True

    if bytes_data or pil_img or recommend_clicked:
        try:
            source = buf if buf else (pil_img if pil_img else dataset_path)
            rec = recommend_from_buffer_or_pil(source, user_prefs=user_prefs, profile=profile_yaml)
            # persist for export tab
            st.session_state["last_rec"] = rec
            st.markdown("#### Recommendation (structured)")
            st.json(rec)
            # pipeline preview
            pipeline = pipeline_from_recommendation(rec)
            st.markdown("#### Pipeline preview (steps)")
            try:
                st.json(pipeline.as_dict())
            except Exception:
                st.write(repr(pipeline))
            st.markdown("You can export this pipeline in the 'Pipeline Export' tab (select format and download).")
        except Exception as e:
            st.error(f"Recommendation failed: {e}")
            logger.exception("recommend failed")

# ---------------------------
# AUGMENT VISUALIZER TAB
# ---------------------------
with tabs[3]:
    st.subheader("🎨 Augment Visualizer")
    st.markdown(
        "Preview augmentations with deterministic seeds. Move intensity slider to see weaker→stronger augmentations. "
        "Use `Apply (dry-run)` to validate the transform on a dataset."
    )

    # seed control for reproducible previews
    seed = st.number_input("Preview seed", value=42, step=1)
    intensity = st.slider("Augmentation intensity", min_value=0.0, max_value=1.0, value=0.5)
    n_examples = st.slider("Examples to show", min_value=1, max_value=8, value=4)

    bytes_data = cache_uploaded_bytes()
    pil_img, buf = load_image_from_bytes(bytes_data) if bytes_data else (None, None)

    profile_yaml = load_profile_yaml(selected_profile)
    # pick rec from session_state if available; else create from profile/defaults
    rec = st.session_state.get("last_rec") or recommend_from_buffer_or_pil(profile_yaml if profile_yaml else {}, profile=profile_yaml)

    pipeline = pipeline_from_recommendation(rec)

    # intensity scaling heuristics
    def scale_spec(spec: dict, intensity_val: float):
        out = {}
        for k, v in (spec or {}).items():
            try:
                if isinstance(v, (int, float)):
                    out[k] = float(v) * float(max(0.0, min(1.0, intensity_val)))
                elif isinstance(v, list):
                    out[k] = [float(x) * float(max(0.0, min(1.0, intensity_val))) if isinstance(x, (int, float)) else x for x in v]
                else:
                    out[k] = v
            except Exception:
                out[k] = v
        return out

    # build preview pipeline
    try:
        preview_steps = []
        for s in getattr(pipeline, "steps", []):
            spec = getattr(s, "spec", {}) or {}
            scaled = scale_spec(spec, intensity)
            name = getattr(s, "name", getattr(s, "__class__", {}).get("__name__", "step"))
            # prefer PipelineStep if available
            if PipelineStep is not None:
                preview_steps.append(PipelineStep(name=name, spec=scaled))
            else:
                # create ad-hoc step with apply method as in fallback
                class _S:
                    def __init__(self, name, spec):
                        self.name = name
                        self.spec = spec

                    def apply(self, img):
                        # reuse the fallback step logic inside pipeline_from_recommendation
                        try:
                            if isinstance(self.spec, dict) and "resize" in self.spec:
                                size = self.spec.get("resize")
                                if isinstance(size, (list, tuple)) and len(size) == 2:
                                    return img.resize((int(size[0]), int(size[1])))
                            if self.name.lower().find("flip") >= 0 and self.spec.get("p", 1.0) > 0:
                                return img.transpose(Image.FLIP_LEFT_RIGHT)
                        except Exception:
                            logger.debug("preview apply failed for %s", self.name, exc_info=True)
                        return img

                    def as_dict(self):
                        return {"name": self.name, "spec": self.spec}

                preview_steps.append(_S(name, scaled))
        # create preview pipeline object compatible with pipeline_from_recommendation output
        class _PreviewPipeline:
            def __init__(self, steps, meta):
                self.steps = steps
                self.meta = meta

            def as_dict(self):
                return {"steps": [getattr(s, "as_dict", lambda: {"name": getattr(s,"name",""), "spec": getattr(s,"spec",{})})() for s in self.steps], "meta": getattr(self, "meta", {})}

        preview_pipeline = _PreviewPipeline(preview_steps, getattr(pipeline, "meta", {}))
    except Exception:
        logger.exception("Building preview pipeline failed")
        preview_pipeline = pipeline

    st.markdown("**Pipeline (preview)**")
    try:
        st.code(json.dumps(preview_pipeline.as_dict(), indent=2))
    except Exception:
        st.write(repr(preview_pipeline))

    # thumbnails
    cols = st.columns(min(n_examples, 4))
    random.seed(int(seed))
    sample_images = []
    if bytes_data or pil_img:
        base_img = pil_img
        for i in range(n_examples):
            try:
                img = base_img.copy()
                for step in preview_pipeline.steps:
                    img = step.apply(img)
                sample_images.append(img)
            except Exception:
                logger.exception("preview apply failed")
    else:
        # sample images from dataset_path
        try:
            images = [p for p in Path(dataset_path).rglob("*") if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")]
            images = images[:max(1, n_examples)]
            for p in images:
                try:
                    img = Image.open(p).convert("RGB")
                    for step in preview_pipeline.steps:
                        img = step.apply(img)
                    sample_images.append(img)
                except Exception:
                    logger.exception("failed to open sample image %s", p)
        except Exception:
            st.warning("No sample images found in dataset path.")

    for i, img in enumerate(sample_images):
        try:
            cols[i % len(cols)].image(img, width='stretch')
        except Exception:
            logger.exception("display thumb failed")

    st.markdown("---")
    st.markdown("**Apply pipeline (dry-run)**")
    out_dir = st.text_input("Output folder (for apply)", value=str(Path.cwd().joinpath("imgshape_out")))
    apply_button = st.button("Apply pipeline (dry-run)")

    if apply_button:
        try:
            tmp_out = Path(out_dir)
            tmp_out.mkdir(parents=True, exist_ok=True)
            preview_pipeline.apply(dataset_path, str(tmp_out), dry_run=True)
            st.success(f"Dry-run complete. Files would be written to: {tmp_out} (dry-run shows actions in logs)")
        except Exception as e:
            st.error(f"Apply failed: {e}")
            logger.exception("apply pipeline error")

# ---------------------------
# REPORTS TAB
# ---------------------------
with tabs[4]:
    st.subheader("📄 Reports")
    st.markdown("Generate Markdown and interactive HTML reports. Report generation may require optional dependencies.")
    # defensive import of report helpers
    try:
        from imgshape.report import generate_markdown_report, generate_html_report  # type: ignore
    except Exception as e:
        generate_markdown_report = None
        generate_html_report = None
        st.error("Report helpers unavailable (optional deps missing). See logs.")
        logger.debug("report import failed: %s", e)

    if generate_markdown_report and generate_html_report:
        if st.button("Generate Reports"):
            try:
                md_path = Path("imgshape_report.md")
                html_path = Path("imgshape_report.html")
                generate_markdown_report(dataset_path, str(md_path))
                generate_html_report(dataset_path, str(html_path))
                st.success("Reports generated")
                st.download_button("⬇️ Download Markdown", data=md_path.read_text(encoding="utf-8"), file_name="imgshape_report.md")
                st.download_button("⬇️ Download HTML", data=html_path.read_text(encoding="utf-8"), file_name="imgshape_report.html")
            except Exception as e:
                st.error(f"Report generation failed: {e}")
                logger.exception("report generation error")
    else:
        st.info("Install report extras to enable generation: pip install imgshape[pdf] or pip install pyyaml reportlab weasyprint (see docs)")

# ---------------------------
# PIPELINE EXPORT TAB
# ---------------------------
with tabs[5]:
    st.subheader("🔗 Pipeline Export")
    st.markdown("Export the current recommendation as runnable code (torchvision / yaml / json).")

    # compute rec_for_export from session state or recompute
    rec_for_export = st.session_state.get("last_rec")
    if rec_for_export is None:
        # generate a safe default recommendation
        rec_for_export = {"preprocessing": [{"name": "resize", "spec": {"resize": [256, 256]}}], "augmentations": [], "meta": {"source": "default"}}

    pipeline_for_export = pipeline_from_recommendation(rec_for_export)

    fmt_options = ["torchvision", "json"]
    if yaml:
        fmt_options.append("yaml")
    fmt = st.selectbox("Export format", options=fmt_options, index=0)

    snippet, fname = export_pipeline_snippet(pipeline_for_export, fmt=fmt)

    st.markdown("**Snippet preview**")
    st.code(snippet, language="python" if fmt == "torchvision" else ("yaml" if fmt == "yaml" else "json"))

    # show direct download button (always present) — avoids nested click/streamlit race
    st.download_button("⬇️ Download pipeline snippet", data=snippet, file_name=fname)

    st.markdown("**Save pipeline (v3 JSON)**")
    save_path = Path.cwd().joinpath("imgshape_pipeline_v3.json")
    try:
        pipeline_for_export.save(str(save_path))
        saved_text = save_path.read_text(encoding="utf-8")
        st.success(f"Saved pipeline to {save_path}")
        st.download_button("⬇️ Download pipeline JSON", data=saved_text, file_name=save_path.name)
    except Exception:
        # fallback: try to write JSON manually if pipeline obj doesn't support .save()
        try:
            exported = pipeline_for_export.export(format="json") if hasattr(pipeline_for_export, "export") else json.dumps(getattr(pipeline_for_export, "as_dict", lambda: {})(), indent=2)
            save_path.write_text(exported, encoding="utf-8")
            st.success(f"Saved pipeline to {save_path} (fallback write)")
            st.download_button("⬇️ Download pipeline JSON", data=exported, file_name=save_path.name)
        except Exception as e:
            st.error(f"Save failed: {e}")
            logger.exception("save pipeline failed: %s", e)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown(
    textwrap.dedent(
        """
        <div style="text-align:center;font-size:12px;">
        <b>imgshape</b> v3.0.0 — Aurora • Built for researchers and engineers.
        • <a href="https://github.com/STiFLeR7/imgshape" target="_blank">GitHub</a>
        </div>
        """
    ),
    unsafe_allow_html=True,
)
