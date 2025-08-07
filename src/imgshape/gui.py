# gui.py
import gradio as gr
from imgshape.shape import get_shape
from imgshape.analyze import analyze_type
from imgshape.recommender import recommend_preprocessing

def analyze_image(image):
    shape = get_shape(image)
    analysis = analyze_type(image)
    recommendation = recommend_preprocessing(image)

    return {
        "Shape": shape,
        "Analysis": analysis,
        "Recommendation": recommendation
    }

def main():
    with gr.Blocks(title="imgshape GUI") as demo:
        gr.Markdown("# 🧠 imgshape Analyzer")
        gr.Markdown("Upload an image to analyze its shape, type, and get model recommendations.")

        img_input = gr.Image(type="filepath", label="Upload Image")
        btn = gr.Button("Analyze")

        out_json = gr.JSON(label="Results")

        btn.click(fn=analyze_image, inputs=[img_input], outputs=[out_json])

    demo.launch()

if __name__ == "__main__":
    main()

