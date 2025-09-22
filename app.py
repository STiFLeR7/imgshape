# app_streamlit.py — Streamlit front-end (updated & defensive)
import streamlit as st
from pathlib import Path
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
import inspect
import json
import logging
import importlib

from imgshape.shape import get_shape
from imgshape.analyze import analyze_type
from imgshape.recommender import recommend_preprocessing
from imgshape.augmentations import AugmentationRecommender
from imgshape.report import generate_markdown_report, generate_html_report
from imgshape.viz import plot_shape_distribution

logger = logging.getLogger("imgshape.streamlit")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

st.set_page_config(page_title="imgshape v2.2.0", layout="wide")
st.title("🖼️ imgshape — Smart Dataset Assistant (v2.2.0)")

st.markdown(
    "Upload an image or provide a dataset folder to analyze, "
    "recommend preprocessing, generate reports, and export Torch transforms."
)

# Sidebar inputs
st.sidebar.header("📂 Input")
uploaded_file = st.sidebar.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png", "bmp", "tiff"]
)
dataset_path = st.sidebar.text_input("Dataset folder path", "assets/sample_images")

tabs = st.tabs(["📐 Shape", "🔍 Analyze", "🧠 Recommend", "📄 Report", "🔗 TorchLoader"])


# ------------------------- Helpers -------------------------
def cache_uploaded_bytes():
    """Cache bytes in session_state so we don't re-read large files repeatedly."""
    if uploaded_file is None:
        return None
    if "uploaded_bytes" not in st.session_state:
        try:
            st.session_state["uploaded_bytes"] = uploaded_file.read()
        except Exception as e:
            st.session_state["uploaded_bytes"] = None
            st.error(f"Error reading upload: {e}")
            return None
    return st.session_state["uploaded_bytes"]


def load_uploaded_image_from_bytes(bytes_data):
    """Return (PIL.Image, BytesIO) or (None, None) on failure."""
    if not bytes_data:
        return None, None
    try:
        buf = BytesIO(bytes_data)
        pil_img = Image.open(buf).convert("RGB")
        buf.seek(0)
        return pil_img, buf
    except UnidentifiedImageError:
        return None, None
    except Exception as e:
        st.error(f"Unexpected error opening image: {e}")
        return None, None


def safe_analyze_from_bytes_or_pil(bytes_data, pil_img):
    """
    Try analyze_type on BytesIO (preferred), else try PIL.Image.
    Return a dict (either analysis or an error dict).
    """
    last_exc = None
    if bytes_data:
        buf = BytesIO(bytes_data)
        try:
            buf.seek(0)
            return analyze_type(buf)
        except Exception as e:
            last_exc = e
            logger.debug("analyze_type(bytes) failed: %s", e, exc_info=True)
    if pil_img is not None:
        try:
            return analyze_type(pil_img)
        except Exception as e:
            last_exc = e
            logger.debug("analyze_type(PIL) failed: %s", e, exc_info=True)
    return {"error": "analyze_failed", "detail": str(last_exc) if last_exc else "no input"}


def safe_recommend_from_bytes_or_pil(bytes_data, pil_img, user_prefs=None):
    """
    Call recommend_preprocessing defensively:
      - If recommend_preprocessing accepts user_prefs, pass it.
      - Else call with single arg (buffer or PIL).
    Returns recommendation dict or error dict.
    """
    last_exc = None
    if recommend_preprocessing is None:
        return {"error": "recommender_missing", "detail": "recommend_preprocessing not available"}

    # Inspect signature to decide whether to pass user_prefs
    try:
        sig = inspect.signature(recommend_preprocessing)
        accepts_user_prefs = "user_prefs" in sig.parameters
    except Exception:
        accepts_user_prefs = False

    # Try bytes (preferred)
    if bytes_data:
        buf = BytesIO(bytes_data)
        try:
            buf.seek(0)
            if accepts_user_prefs and user_prefs is not None:
                return recommend_preprocessing(buf, user_prefs=user_prefs)
            else:
                return recommend_preprocessing(buf)
        except Exception as e:
            last_exc = e
            logger.debug("recommend_preprocessing(bytes) failed: %s", e, exc_info=True)

    # Try PIL
    if pil_img is not None:
        try:
            if accepts_user_prefs and user_prefs is not None:
                return recommend_preprocessing(pil_img, user_prefs=user_prefs)
            else:
                return recommend_preprocessing(pil_img)
        except Exception as e:
            last_exc = e
            logger.debug("recommend_preprocessing(PIL) failed: %s", e, exc_info=True)

    return {"error": "recommend_failed", "detail": str(last_exc) if last_exc else "no input"}


def safe_to_torch_transform(plan_or_config, rec, prefer_snippet: bool = False):
    """
    Wrapper that calls to_torch_transform in a backward-compatible way.
    - If to_torch_transform supports prefer_snippet kw -> use it.
    - Else try legacy calling style (plan, preprocessing) or (config, rec).
    - Catch errors and return informative dict {'error':..., 'detail':...}
    """
    try:
        from imgshape.torchloader import to_torch_transform
    except Exception as e:
        return {"error": "torchloader_missing", "detail": str(e)}

    try:
        sig = inspect.signature(to_torch_transform)
        params = sig.parameters
        # If prefer_snippet supported as named param, call with it
        if "prefer_snippet" in params:
            try:
                return to_torch_transform(plan_or_config, rec, prefer_snippet=prefer_snippet)
            except TypeError:
                # fallback to calling without keyword
                return to_torch_transform(plan_or_config, rec)
        else:
            # legacy style: try (plan, preprocessing) -> many tests call this
            try:
                return to_torch_transform(plan_or_config, rec)
            except TypeError:
                # try single arg style
                try:
                    return to_torch_transform(rec)
                except Exception as e:
                    return {"error": "transform_call_failed", "detail": str(e)}
    except Exception as e:
        logger.exception("safe_to_torch_transform failed: %s", e)
        return {"error": "transform_wrapper_failed", "detail": str(e)}


# ------------------------- SHAPE TAB -------------------------
with tabs[0]:
    st.subheader("📐 Shape Detection")
    bytes_data = cache_uploaded_bytes()
    if bytes_data:
        pil_img, buf = load_uploaded_image_from_bytes(bytes_data)
        if pil_img is None:
            st.error("Uploaded file is not a valid image. Please upload a PNG/JPEG/etc.")
        else:
            st.image(pil_img, caption="Uploaded Image", use_column_width=True)
            try:
                # prefer passing PIL image to get immediate shape
                shape = get_shape(pil_img)
                st.json({"shape": shape})
            except Exception as e:
                st.error(f"Error in shape detection: {e}")
    else:
        st.info("Upload an image to see its shape.")


# ------------------------- ANALYZE TAB -------------------------
with tabs[1]:
    st.subheader("🔍 Image Analysis")
    bytes_data = cache_uploaded_bytes()
    pil_img, buf = load_uploaded_image_from_bytes(bytes_data) if bytes_data else (None, None)

    if bytes_data or pil_img:
        # prefer analyzing from BytesIO so analyze_type can treat it consistently
        res = safe_analyze_from_bytes_or_pil(bytes_data, pil_img)
        if isinstance(res, dict) and res.get("error"):
            st.error(json.dumps(res, indent=2))
        else:
            st.json(res)
    else:
        st.info("Upload an image to analyze.")

    st.subheader("📊 Dataset Visualization")
    if st.button("Plot Shape Distribution"):
        try:
            out = plot_shape_distribution(dataset_path, save=False)
            if out:
                st.success(f"Saved plot: {out}")
            else:
                st.info("Plot created (displayed on server side).")
        except Exception as e:
            st.error(f"Error plotting dataset: {e}")


# ------------------------- RECOMMEND TAB -------------------------
with tabs[2]:
    st.subheader("🧠 Preprocessing + Augmentation Recommendations")
    bytes_data = cache_uploaded_bytes()
    pil_img, buf = load_uploaded_image_from_bytes(bytes_data) if bytes_data else (None, None)

    if bytes_data or pil_img:
        # user prefs input (left in sidebar)
        user_prefs_input = st.sidebar.text_input("Prefs (comma-separated)", "")
        user_prefs = [p.strip() for p in user_prefs_input.split(",") if p.strip()] if user_prefs_input else None

        rec = safe_recommend_from_bytes_or_pil(bytes_data, pil_img, user_prefs=user_prefs)
        if isinstance(rec, dict) and rec.get("error"):
            st.error(json.dumps(rec, indent=2))
        else:
            st.json({"preprocessing": rec})

        # Augmentation plan (deterministic heuristic)
        try:
            ar = AugmentationRecommender(seed=42)
            # create a minimal stats object for the recommender
            analysis = {}
            if isinstance(rec, dict) and rec:
                analysis["entropy_mean"] = rec.get("entropy") or rec.get("meta", {}).get("entropy") or rec.get("entropy_mean")
            plan = ar.recommend_for_dataset({"entropy_mean": analysis.get("entropy_mean", 5.0), "image_count": 1})

            # Build a safe dict for UI display (don't rely on plan.as_dict existing)
            try:
                plan_dict = plan.as_dict()  # preferred if available
            except Exception:
                plan_dict = {
                    "order": getattr(plan, "recommended_order", []),
                    "augmentations": [a.__dict__ if hasattr(a, "__dict__") else dict(a) for a in getattr(plan, "augmentations", [])],
                    "seed": getattr(plan, "seed", None),
                }

            st.markdown("#### Augmentation plan (heuristic)")
            st.json(plan_dict)
        except Exception as e:
            st.error(f"Augmentation generation failed: {e}")
    else:
        st.info("Upload an image to get recommendations.")


# ------------------------- REPORT TAB -------------------------
with tabs[3]:
    st.subheader("📄 Dataset Report")
    if st.button("Generate Markdown + HTML Report"):
        try:
            md_path = Path("report.md")
            html_path = Path("report.html")

            # prefer using dataset path; if none exists, try creating reports from a sample recommendation
            generate_markdown_report(dataset_path, str(md_path))
            generate_html_report(dataset_path, str(html_path))

            st.success("Reports generated!")
            st.download_button("⬇️ Download Markdown", md_path.read_text(), file_name="report.md")
            st.download_button("⬇️ Download HTML", html_path.read_text(), file_name="report.html")
        except Exception as e:
            st.error(f"Error generating report: {e}")


# ------------------------- TORCHLOADER TAB -------------------------
with tabs[4]:
    st.subheader("🔗 TorchLoader Export")
    bytes_data = cache_uploaded_bytes()
    pil_img, buf = load_uploaded_image_from_bytes(bytes_data) if bytes_data else (None, None)

    if bytes_data or pil_img:
        # Build a representative recommendation (prefers rec if available)
        rec = None
        try:
            rec = safe_recommend_from_bytes_or_pil(bytes_data, pil_img)
        except Exception as e:
            rec = {"error": "recommend_fail", "detail": str(e)}

        # Attempt to produce transform (callable or snippet)
        transform_result = safe_to_torch_transform({}, rec or {}, prefer_snippet=False)

        if isinstance(transform_result, str):
            st.code(transform_result, language="python")
        elif isinstance(transform_result, dict) and transform_result.get("error"):
            st.error(json.dumps(transform_result, indent=2))
        else:
            # Most likely a Compose-like callable/object
            st.success("✅ torchvision.transforms.Compose (or equivalent) created")
            st.write(transform_result)
    else:
        st.info("Upload an image to export Torch transforms.")


# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        <p><b>Connect with me</b></p>
        <a href="https://instagram.com/stifler.xd" target="_blank" style="margin: 0 10px; text-decoration: none;">
            <img src="https://cdn-icons-png.flaticon.com/512/2111/2111463.png" width="24"/> Instagram
        </a>
        <a href="https://github.com/STiFLeR7" target="_blank" style="margin: 0 10px; text-decoration: none;">
            <img src="https://cdn-icons-png.flaticon.com/512/733/733553.png" width="24"/> GitHub
        </a>
        <a href="https://huggingface.co/STiFLeR7" target="_blank" style="margin: 0 10px; text-decoration: none;">
            <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="24"/> HuggingFace
        </a>
        <a href="https://medium.com/@stiflerxd" target="_blank" style="margin: 0 10px; text-decoration: none;">
            <img src="https://cdn-icons-png.flaticon.com/512/5968/5968906.png" width="24"/> Medium
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)
