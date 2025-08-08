# src/imgshape/gui.py
import gradio as gr
from imgshape.shape import get_shape
from imgshape.analyze import analyze_type
from imgshape.recommender import recommend_preprocessing

def launch_gui():
    def process_image(image):
        shape = get_shape(image)
        analysis = analyze_type(image)
        recommendation = recommend_preprocessing(image)
        return {
            "Shape": shape,
            "Analysis": analysis,
            "Recommendation": recommendation
        }

    demo = gr.Interface(
        fn=process_image,
        inputs=gr.Image(type="filepath", label="Upload Image"),
        outputs="json",
        title="📦 imgshape GUI",
        description="Get image shape, analysis, and preprocessing recommendations."
    )

    demo.launch()

# Allow direct run: `python -m src.imgshape.gui`
if __name__ == "__main__":
    launch_gui()
