# src/imgshape/cli.py
import argparse
from imgshape.shape import get_shape, get_shape_batch
from imgshape.analyze import analyze_type
from imgshape.recommender import recommend_preprocessing
from imgshape.compatibility import check_model_compatibility
from imgshape.viz import plot_shape_distribution
from imgshape.gui import launch_gui

def main():
    parser = argparse.ArgumentParser(description="📦 imgshape v2.0.0 — Image Shape, Analysis & Preprocessing Toolkit")

    parser.add_argument("--path", type=str, help="Path to a single image")
    parser.add_argument("--url", type=str, help="Image URL to analyze")
    parser.add_argument("--batch", action="store_true", help="Get shapes for multiple images")
    parser.add_argument("--analyze", action="store_true", help="Analyze image type and stats")
    parser.add_argument("--recommend", action="store_true", help="Recommend preprocessing for image")
    parser.add_argument("--check", type=str, help="Check compatibility with a model")
    parser.add_argument("--dir", type=str, help="Directory for model compatibility check")
    parser.add_argument("--viz", type=str, help="Plot dataset shape/size distribution")
    parser.add_argument("--web", action="store_true", help="Launch web GUI")

    args = parser.parse_args()

    # Single image shape
    if args.path and not any([args.analyze, args.recommend]):
        print(f"\n📐 Shape for: {args.path}")
        print(get_shape(args.path))

    # Analyze image
    if args.path and args.analyze:
        print(f"\n🔍 Analysis for: {args.path}")
        print(analyze_type(args.path))

    # Recommend preprocessing
    if args.path and args.recommend:
        print(f"\n🧠 Recommendation for: {args.path}")
        print(recommend_preprocessing(args.path))

    # Model compatibility
    if args.dir and args.check:
        passed, failed = check_model_compatibility(args.dir, args.check)
        print(f"\n✅ Model: {args.check}")
        print(f"🖼️ Total Images: {passed + failed}")
        print(f"✔️ Passed: {passed}")
        if failed:
            print(f"❌ Failed: {failed}")
        else:
            print("🎉 All images are compatible!")

    # Visualization
    if args.viz:
        print(f"\n📊 Plotting shape distribution for: {args.viz}")
        plot_shape_distribution(args.viz)

    # Web GUI
    if args.web:
        print("\n🚀 Launching imgshape Web GUI...")
        launch_gui()

if __name__ == "__main__":
    main()
