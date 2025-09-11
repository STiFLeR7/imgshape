import streamlit as st
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from imgshape.shape import get_shape
from imgshape.analyze import analyze_type
from imgshape.recommender import recommend_preprocessing
from imgshape.augmentations import AugmentationRecommender
from imgshape.report import generate_markdown_report, generate_html_report
from imgshape.viz import plot_shape_distribution
from imgshape.torchloader import to_torch_transform


st.set_page_config(page_title="imgshape v2.1.0", layout="wide")
st.title("🖼️ imgshape — Smart Dataset Assistant (v2.1.0)")

st.markdown(
    "Upload an image or provide a dataset folder to analyze, "
    "recommend preprocessing, generate reports, and even get PyTorch transforms."
)

# Sidebar for inputs
st.sidebar.header("📂 Input")
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
dataset_path = st.sidebar.text_input("Dataset folder path", "assets/sample_images")

tabs = st.tabs(["📐 Shape", "🔍 Analyze", "🧠 Recommend", "📄 Report", "🔗 TorchLoader"])


# ------------------------- SHAPE TAB -------------------------
with tabs[0]:
    st.subheader("📐 Shape Detection")
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        shape = get_shape(uploaded_file)
        st.json({"shape": shape})
    else:
        st.info("Upload an image to see its shape.")


# ------------------------- ANALYZE TAB -------------------------
with tabs[1]:
    st.subheader("🔍 Image Analysis")
    if uploaded_file:
        analysis = analyze_type(uploaded_file)
        st.json(analysis)
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
    if uploaded_file:
        rec = recommend_preprocessing(uploaded_file)
        st.json({"preprocessing": rec})

        # Augmentation plan
        ar = AugmentationRecommender(seed=42)
        analysis = analyze_type(uploaded_file)
        plan = ar.recommend_for_dataset({"entropy_mean": analysis.get("entropy", 5.0), "image_count": 1})
        st.json({
            "augmentation_plan": {
                "order": plan.recommended_order,
                "augmentations": [a.__dict__ for a in plan.augmentations]
            }
        })
    else:
        st.info("Upload an image to get recommendations.")


# ------------------------- REPORT TAB -------------------------
with tabs[3]:
    st.subheader("📄 Dataset Report")
    if st.button("Generate Markdown + HTML Report"):
        try:
            stats = {"image_count": 1, "source_dir": dataset_path}
            rec = recommend_preprocessing(uploaded_file) if uploaded_file else {}
            ar = AugmentationRecommender(seed=42)
            plan = ar.recommend_for_dataset({"entropy_mean": 5.0, "image_count": 10})

            md_path = Path("report.md")
            html_path = Path("report.html")

            generate_markdown_report(md_path, stats, {}, rec, {"augmentations": [a.__dict__ for a in plan.augmentations]})
            generate_html_report(md_path, html_path)

            st.success("Reports generated!")
            st.download_button("⬇️ Download Markdown", md_path.read_text(), file_name="report.md")
            st.download_button("⬇️ Download HTML", html_path.read_text(), file_name="report.html")
        except Exception as e:
            st.error(f"Error generating report: {e}")


# ------------------------- TORCHLOADER TAB -------------------------
with tabs[4]:
    st.subheader("🔗 TorchLoader Export")
    try:
        rec = recommend_preprocessing(uploaded_file) if uploaded_file else {}
        snippet_or_transform = to_torch_transform({}, rec)

        if isinstance(snippet_or_transform, str):
            st.code(snippet_or_transform, language="python")
        else:
            st.success("✅ torchvision.transforms.Compose object created")
            st.write(snippet_or_transform)
    except Exception as e:
        st.error(f"Error building Torch transform: {e}")
