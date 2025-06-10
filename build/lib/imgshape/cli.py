import argparse
from imgshape.resize import batch_resize

def main():
    parser = argparse.ArgumentParser(description="Resize images in a folder for ML training.")
    parser.add_argument("folder", help="Path to image folder")
    parser.add_argument("--size", type=str, required=True, help="Target size (e.g. 224 or 224x224)")
    parser.add_argument("--format", type=str, default="jpg", help="Output format: jpg, png")
    parser.add_argument("--keep-structure", action="store_true", help="Preserve folder structure")
    parser.add_argument("--save-dir", type=str, help="Directory to save resized images")
    parser.add_argument("--keep-original", action="store_true", help="Keep original image copies")

    args = parser.parse_args()

    results = batch_resize(
        folder_path=args.folder,
        size=args.size,
        fmt=args.format,
        keep_structure=args.keep_structure,
        save_dir=args.save_dir,
        keep_original=args.keep_original
    )

    print(f"✅ Resized {len(results)} images.")

if __name__ == "__main__":
    main()
