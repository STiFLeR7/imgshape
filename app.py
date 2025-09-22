# app.py — Streamlit UI for imgshape v2.2.0 (updated, robust, deploy-ready)
from __future__ import annotations

import logging
import tempfile
import os
from pathlib import Path
from io import BytesIO
from typing import Optional, Dict, Any

import streamlit as st
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt

# imgshape internals (safe imports)
try:
    from imgshape.shape import get_shape
except Exception:
    get_shape = None

try:
    from imgshape.analyze import analyze_type, analyze_dataset
except Exception:
    analyze_type = None
    analyze_dataset = None

try:
    from imgshape.recommender import recommend_preprocessing, recommend_dataset
except Exception:
    recommend_preprocessing = None
    recommend_dataset = None

try:
    from imgshape.augmentations import AugmentationRecommender
except Exception:
    AugmentationRecommender = None

try:
    from imgshape.report import generate_markdown_report, generate_html_report
except Exception:
    generate_markdown_report = None
    generate_html_report = None

try:
    from imgshape.torchloader import to_torch_transform
except Exception:
    to_torch_transform = None

try:
    from imgshape.compatibility import check_compatibility
except Exception:
    check_compatibility = None

# basic logger
logger = logging.getLogger("imgshape.streamlit")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Streamlit page config
st.set_page_config(page_title="imgshape v2.2.0", layout="wide", initial_sidebar_state="expanded")
st.title("🖼️ imgshape — Smart Dataset Assistant (v2.2.0)")
st.caption("Analyze images & datasets, get preprocessing + augmentation recommendations, generate reports, and export Torch transforms.")

# ------------------------------
# Sidebar / Inputs
# ------------------------------
st.sidebar.header("Input & Options")

uploaded_file = st.sidebar.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg", "bmp", "tiff"])
dataset_path_input = st.sidebar.text_input("Dataset folder (local)", value="assets/sample_images")
auto_detect = st.sidebar.checkbox("Auto-detect common dataset paths", value=False)

st.sidebar.markdown("---")
st.sidebar.header("TorchLoader")
prefer_snippet = st.sidebar.checkbox("Prefer snippet (return code)", value=False)
st.sidebar.markdown("---")

st.sidebar.header("Compatibility check")
model_name = st.sidebar.text_input("Model name (for compatibility)", value="mobilenet_v2")
run_compatibility = st.sidebar.button("Run compatibility check")

# ------------------------------
# Helpers
# ------------------------------
@st.cache_data(show_spinner=False)
def _read_uploaded_bytes(uploaded) -> Optional[bytes]:
    if uploaded is None:
        return None
    try:
        uploaded.seek(0)
    except Exception:
        pass
    try:
        data = uploaded.read()
        return data
    except Exception as e:
        logger.warning("failed read upload: %s", e)
        return None


def _open_image_from_bytes(b: bytes) -> Optional[Image.Image]:
    if not b:
        return None
    try:
        img = Image.open(BytesIO(b))
        # always convert to RGB where sensible; keep as-is if conversion fails
        try:
            img = img.convert("RGB")
        except Exception:
            pass
        return img
    except UnidentifiedImageError:
        return None
    except Exception:
        logger.exception("unexpected error opening uploaded image")
        return None


def _safe_analyze_image(img_or_buf) -> Dict[str, Any]:
    if analyze_type is None:
        return {"error": "analyze_unavailable"}
    try:
        return analyze_type(img_or_buf)
    except Exception as e:
        logger.exception("analyze_type error: %s", e)
        return {"error": "analyze_failed", "detail": str(e)}


@st.cache_data(show_spinner=False)
def _safe_analyze_dataset(path: str, sample_limit: int = 200) -> Dict[str, Any]:
    if analyze_dataset is None:
        return {"error": "analyze_dataset_unavailable"}
    p = Path(path)
    if not p.exists() or not p.is_dir():
        return {"error": "dataset_not_found", "message": f"{path} not found or not a directory"}
    try:
        return analyze_dataset(path, sample_limit=sample_limit)
    except Exception as e:
        logger.exception("analyze_dataset error: %s", e)
        return {"error": "analysis_failed", "detail": str(e)}


def _plot_shape_histogram_from_analysis(analysis: Dict[str, Any]):
    # analysis expected to have "unique_shapes" mapping or "shapes"
    shapes = analysis.get("unique_shapes") or analysis.get("shapes") or {}
    # normalize: if dict mapping "WxH"->count
    if isinstance(shapes, dict):
        labels = list(shapes.keys())
        counts = list(shapes.values())
    else:
        # list of shapes tuples -> convert
        labels = []
        counts = []
        try:
            from collections import Counter
            c = Counter()
            for s in shapes:
                if isinstance(s, (list, tuple)) and len(s) >= 2:
                    label = f"{s[1]}x{s[0]}"  # WxH
                else:
                    label = str(s)
                c[label] += 1
            labels = list(c.keys())
            counts = list(c.values())
        except Exception:
            labels, counts = [], []

    fig, ax = plt.subplots(figsize=(8, 3.5))
    if not labels:
        ax.text(0.5, 0.5, "No shapes found", ha="center", va="center")
    else:
        ax.bar(labels, counts)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Count")
        ax.set_title("Image shape (W×H) distribution")
        fig.tight_layout()
    return fig


def _ensure_dir_for_file(path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)


# ------------------------------
# Main UI Columns
# ------------------------------
left, right = st.columns([1, 2])

with left:
    st.subheader("Input Preview")
    uploaded_bytes = _read_uploaded_bytes(uploaded_file)
    pil_img = None
    if uploaded_bytes:
        pil_img = _open_image_from_bytes(uploaded_bytes)
        if pil_img is None:
            st.error("Uploaded file is not a valid image.")
        else:
            st.image(pil_img, caption="Uploaded image", use_column_width=True)
            try:
                if get_shape:
                    shape = get_shape(pil_img)
                    st.write("Detected shape (H, W, C):", shape)
            except Exception:
                pass

    st.markdown("---")
    st.subheader("Dataset selection")
    dataset_path = dataset_path_input
    if auto_detect:
        # heuristic checks for common local folders
        candidates = [
            Path(dataset_path_input),
            Path("assets/sample_images"),
            Path("data"),
            Path("dataset"),
            Path.cwd() / "images",
        ]
        chosen = None
        for c in candidates:
            if c.exists() and c.is_dir():
                chosen = c
                break
        if chosen:
            st.success(f"Auto-detected dataset folder: {chosen}")
            dataset_path = str(chosen)
        else:
            st.info("No common dataset folder auto-detected.")

    st.write("Dataset path:", dataset_path)

    st.markdown("---")
    st.subheader("Quick actions")
    if st.button("Analyze dataset (sample)"):
        with st.spinner("Analyzing dataset..."):
            ds_analysis = _safe_analyze_dataset(dataset_path)
            if ds_analysis.get("error"):
                st.error(ds_analysis.get("message") or ds_analysis.get("detail") or ds_analysis.get("error"))
            else:
                st.success(f"Found ~{ds_analysis.get('image_count', 0)} images (sampled).")
                fig = _plot_shape_histogram_from_analysis(ds_analysis)
                st.pyplot(fig)

    if st.button("Recommend dataset (representative)"):
        with st.spinner("Generating dataset recommendation..."):
            if recommend_dataset is None:
                st.error("recommend_dataset unavailable in this environment.")
            else:
                try:
                    rec = recommend_dataset(dataset_path)
                    st.json(rec)
                except Exception as e:
                    st.error(f"recommend_dataset failed: {e}")

with right:
    st.subheader("Image / Preprocessing tools")

    # Analyze single image
    st.markdown("### Analyze uploaded image")
    if uploaded_bytes:
        if pil_img is None:
            st.warning("Cannot analyze: uploaded file unreadable.")
        else:
            with st.spinner("Analyzing image..."):
                analysis = _safe_analyze_image(pil_img)
                st.json(analysis)

    else:
        st.info("Upload an image to analyze or get preprocessing suggestions.")

    st.markdown("### Preprocessing recommendation")
    if uploaded_bytes and pil_img is not None:
        if recommend_preprocessing is None:
            st.warning("recommend_preprocessing not available in this environment.")
        else:
            with st.spinner("Recommending preprocessing..."):
                try:
                    rec = recommend_preprocessing(pil_img)
                    st.json(rec)
                except Exception as e:
                    st.error(f"recommend_preprocessing failed: {e}")
    else:
        st.info("Upload an image to get a preprocessing recommendation.")

    st.markdown("### Augmentation plan (heuristic)")
    if AugmentationRecommender is not None:
        try:
            ar = AugmentationRecommender(seed=42)
            # get a small stats dict for plan (best-effort)
            fake_stats = {"entropy_mean": (analysis.get("entropy") if isinstance(analysis, dict) else 5.0), "image_count": 10}
            plan = ar.recommend_for_dataset(fake_stats)
            st.json({"order": plan.recommended_order, "augmentations": [a.__dict__ for a in plan.augmentations]})
        except Exception as e:
            st.error(f"AugmentationRecommender error: {e}")
    else:
        st.info("Augmentation recommender not present.")

    st.markdown("### Export Torch transform")
    if uploaded_bytes and pil_img is not None:
        if to_torch_transform is None:
            st.warning("to_torch_transform not available in this environment.")
        else:
            try:
                # use the preprocessing recommendation as input if available
                rec_input = rec if 'rec' in locals() and isinstance(rec, dict) else {}
                t = to_torch_transform({}, rec_input, prefer_snippet=prefer_snippet)
                if isinstance(t, str):
                    st.code(t, language="python")
                elif callable(t):
                    st.success("Transform is callable (torchvision Compose or no-op).")
                    st.write(t)
                else:
                    st.write("Transform:", t)
            except Exception as e:
                st.error(f"to_torch_transform failed: {e}")
    else:
        st.info("Upload an image to export Torch transforms.")

# ------------------------------
# Compatibility panel (bottom)
# ------------------------------
st.markdown("---")
st.header("🔬 Model / Dataset Compatibility")

compat_col1, compat_col2 = st.columns([1, 2])
with compat_col1:
    st.write("Model")
    model_input = st.text_input("Model name for compatibility check", value=model_name)
    st.write("Dataset (local folder)")
    dataset_input_for_check = st.text_input("Dataset folder for compatibility", value=dataset_path)
    run_btn = st.button("Check compatibility")
with compat_col2:
    if run_btn or run_compatibility:
        if check_compatibility is None:
            st.error("Compatibility checker unavailable in this environment.")
        else:
            with st.spinner("Running compatibility check..."):
                try:
                    report = check_compatibility(model=model_input, dataset_path=dataset_input_for_check)
                    st.json(report)
                    # summary
                    total = report.get("total", 0)
                    passed = report.get("passed", 0)
                    status = report.get("compatibility_report", {}).get("status", "unknown")
                    st.metric("Overall status", status)
                    st.write(f"Checks passed: {passed}/{total if total else 'N/A'}")
                except Exception as e:
                    st.error(f"check_compatibility failed: {e}")

# ------------------------------
# Report generation
# ------------------------------
st.markdown("---")
st.header("📄 Generate reports")

report_col1, report_col2 = st.columns([1, 1])
with report_col1:
    md_file = st.text_input("Markdown output filename", value="imgshape_report.md")
    html_file = st.text_input("HTML output filename", value="imgshape_report.html")
    if st.button("Generate reports"):
        try:
            # Prepare minimal data for report generation
            stats = {"image_count": 0, "source_dir": dataset_path}
            if uploaded_bytes and pil_img is not None:
                rec_for_report = recommend_preprocessing(pil_img) if recommend_preprocessing else {}
            else:
                rec_for_report = {}
            if generate_markdown_report is None or generate_html_report is None:
                st.warning("Report generators not available.")
            else:
                # Streamlit friendly write paths
                md_path = Path(md_file)
                html_path = Path(html_file)
                _ensure_dir_for_file(str(md_path))
                _ensure_dir_for_file(str(html_path))

                # generate_markdown_report accepts legacy param ordering in our refactor
                generate_markdown_report(str(md_path), stats, {}, rec_for_report, {"augmentations": []})
                generate_html_report(str(md_path), str(html_path))
                st.success("Reports written.")
                st.download_button("Download Markdown", md_path.read_text(encoding="utf-8"), file_name=md_path.name)
                st.download_button("Download HTML", html_path.read_text(encoding="utf-8"), file_name=html_path.name)
        except Exception as e:
            st.error(f"Report generation failed: {e}")

with report_col2:
    st.write("Quick preview of dataset analysis (sample)")
    if st.button("Preview dataset analysis"):
        with st.spinner("Analyzing (sample)..."):
            ds = _safe_analyze_dataset(dataset_path)
            if ds.get("error"):
                st.error(ds.get("message") or ds.get("detail") or ds.get("error"))
            else:
                st.json(ds)
                fig = _plot_shape_histogram_from_analysis(ds)
                st.pyplot(fig)

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("imgshape v2.2.0 — designed for reproducible, compact dataset checks. Report issues at the repository.")
