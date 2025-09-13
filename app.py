import streamlit as st
from pathlib import Path
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt

from imgshape.shape import get_shape
from imgshape.analyze import analyze_type
from imgshape.recommender import recommend_preprocessing
from imgshape.augmentations import AugmentationRecommender
from imgshape.report import generate_markdown_report, generate_html_report
from imgshape.viz import plot_shape_distribution
from imgshape.torchloader import to_torch_transform

# Page config
st.set_page_config(page_title="imgshape v2.1.3", layout="wide")
st.title("🖼️ imgshape — Smart Dataset Assistant (v2.1.3)")

st.markdown(
    "Upload an image or provide a dataset folder to analyze, "
    "recommend preprocessing, generate reports, and even get PyTorch transforms."
)

# Sidebar for inputs
st.sidebar.header("📂 Input")
uploaded_file = st.sidebar.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png", "bmp", "tiff"]
)
dataset_path = st.sidebar.text_input("Dataset folder path", "assets/sample_images")

tabs = st.tabs(["📐 Shape", "🔍 Analyze", "🧠 Recommend", "📄 Report", "🔗 TorchLoader"])


# ------------------------- helpers -------------------------
def cache_uploaded_bytes():
    """
    Read uploaded_file once and cache raw bytes in session_state['uploaded_bytes'].
    Returns bytes or None.
    """
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
    """
    Build fresh BytesIO and PIL.Image from raw bytes.
    Returns (PIL.Image, BytesIO) or (None, None) on error.
    """
    if not bytes_data:
        return None, None
    try:
        buf = BytesIO(bytes_data)
        pil_img = Image.open(BytesIO(bytes_data)).convert("RGB")
        return pil_img, buf
    except UnidentifiedImageError:
        return None, None
    except Exception as e:
        st.error(f"Unexpected error opening image: {e}")
        return None, None


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
                # get_shape accepts a PIL.Image or path depending on implementation
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
    if bytes_data:
        pil_img, buf = load_uploaded_image_from_bytes(bytes_data)
        if pil_img is None:
            st.error("Uploaded file is not a valid image. Please upload a PNG/JPEG/etc.")
        else:
            buf.seek(0)
            try:
                analysis = analyze_type(buf)
                st.json(analysis)
            except Exception as e:
                st.error(f"Error in analysis: {e}")
    else:
        st.info("Upload an image to analyze.")

    st.subheader("📊 Dataset Visualization")
    if st.button("Plot Shape Distribution"):
        try:
            fig = plt.figure()
            plot_shape_distribution(dataset_path, save=False)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error plotting dataset: {e}")


# ------------------------- RECOMMEND TAB -------------------------
with tabs[2]:
    st.subheader("🧠 Preprocessing + Augmentation Recommendations")
    bytes_data = cache_uploaded_bytes()
    if bytes_data:
        pil_img, buf = load_uploaded_image_from_bytes(bytes_data)
        if pil_img is None:
            st.error("Uploaded file is not a valid image. Please upload a PNG/JPEG/etc.")
        else:
            try:
                rec = recommend_preprocessing(pil_img)  # ✅ pass PIL.Image
                st.json({"preprocessing": rec})
            except Exception as e:
                st.error(f"Error in preprocessing recommendation: {e}")

            # Augmentation plan
            try:
                ar = AugmentationRecommender(seed=42)
                buf.seek(0)
                analysis = analyze_type(buf)  # still safe with BytesIO
                plan = ar.recommend_for_dataset(
                    {"entropy_mean": analysis.get("entropy", 5.0), "image_count": 1}
                )
                st.json({
                    "augmentation_plan": {
                        "order": plan.recommended_order,
                        "augmentations": [a.__dict__ for a in plan.augmentations]
                    }
                })
            except Exception as e:
                st.error(f"Error in augmentation plan: {e}")
    else:
        st.info("Upload an image to get recommendations.")


# ------------------------- REPORT TAB -------------------------
with tabs[3]:
    st.subheader("📄 Dataset Report")
    if st.button("Generate Markdown + HTML Report"):
        try:
            stats = {"image_count": 1, "source_dir": dataset_path}
            rec = {}
            bytes_data = cache_uploaded_bytes()
            if bytes_data:
                # use bytes for preprocessing recommendation
                _, buf = load_uploaded_image_from_bytes(bytes_data)
                if buf is not None:
                    buf.seek(0)
                    rec = recommend_preprocessing(buf)

            ar = AugmentationRecommender(seed=42)
            plan = ar.recommend_for_dataset({"entropy_mean": 5.0, "image_count": 10})

            md_path = Path("report.md")
            html_path = Path("report.html")

            generate_markdown_report(
                md_path, stats, {}, rec,
                {"augmentations": [a.__dict__ for a in plan.augmentations]}
            )
            generate_html_report(md_path, html_path)

            st.success("Reports generated!")
            st.download_button("⬇️ Download Markdown", md_path.read_text(), file_name="report.md")
            st.download_button("⬇️ Download HTML", html_path.read_text(), file_name="report.html")
        except Exception as e:
            st.error(f"Error generating report: {e}")


# ------------------------- TORCHLOADER TAB -------------------------
with tabs[4]:
    st.subheader("🔗 TorchLoader Export")
    bytes_data = cache_uploaded_bytes()
    if bytes_data:
        pil_img, buf = load_uploaded_image_from_bytes(bytes_data)
        if pil_img is None:
            st.error("Uploaded file is not a valid image. Please upload a PNG/JPEG/etc.")
        else:
            try:
                rec = recommend_preprocessing(pil_img)  # ✅ use PIL.Image
                snippet_or_transform = to_torch_transform({}, rec)

                if isinstance(snippet_or_transform, str):
                    st.code(snippet_or_transform, language="python")
                else:
                    st.success("✅ torchvision.transforms.Compose object created")
                    st.write(snippet_or_transform)
            except Exception as e:
                st.error(f"Error building Torch transform: {e}")
    else:
        st.info("Upload an image to export Torch transforms.")

# ------------------------- FOOTER -------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        <p><b>Connect with me</b></p>
        <a href="https://instagram.com/stifler.xd" target="_blank" style="margin: 0 10px; text-decoration: none;">
            <img src="https://cdn-icons-png.flaticon.com/512/2111/2111463.png" width="30"/> Instagram
        </a>
        <a href="https://github.com/STiFLeR7" target="_blank" style="margin: 0 10px; text-decoration: none;">
            <img src="https://cdn-icons-png.flaticon.com/512/733/733553.png" width="30"/> GitHub
        </a>
        <a href="https://huggingface.co/STiFLeR7" target="_blank" style="margin: 0 10px; text-decoration: none;">
            <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="30"/> HuggingFace
        </a>
        <a href="https://medium.com/@stiflerxd" target="_blank" style="margin: 0 10px; text-decoration: none;">
            <img src="https://cdn-icons-png.flaticon.com/512/5968/5968906.png" width="30"/> Medium
        </a>
        <a href="https://www.kaggle.com/stiflerxd" target="_blank" style="margin: 0 10px; text-decoration: none;">
            <img src="https://cdn-icons-png.flaticon.com/512/2111/2111290.png" width="30"/> Kaggle
        </a>
        <br><br>
        📧 <a href="mailto:hillaniljppatel@gmail.com">hillaniljppatel@gmail.com</a> |
        🌐 <a href="https://hillpatel.tech" target="_blank">hillpatel.tech</a>
    </div>
    """,
    unsafe_allow_html=True
)