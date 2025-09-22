# debug_recommender.py
import sys
from pathlib import Path
import json
from PIL import Image
from imgshape.recommender import recommend_dataset, _open_image_from_input, _shape_from_image, _entropy_from_image

def list_images(dataset):
    p = Path(dataset).expanduser().resolve()
    print("Resolved dataset path:", p)
    print("Exists:", p.exists(), "Is dir:", p.is_dir())
    if not p.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
    files = [f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in exts]
    print(f"Found {len(files)} candidate image files (matching exts). Showing up to 10:")
    for f in files[:10]:
        print("  -", f)
    return files

def try_open(sample_files, n=5):
    print("\nAttempting to open up to", n, "files:")
    for f in sample_files[:n]:
        print("Testing", f)
        try:
            img = _open_image_from_input(f)
            if img is None:
                print("  -> open returned None")
                continue
            shp = _shape_from_image(img)
            ent = _entropy_from_image(img)
            print(f"  -> opened OK: shape={shp}, entropy={ent}, mode={img.mode}")
            # show a tiny pixel-sample (size)
            print("     size:", img.size)
        except Exception as e:
            print("  -> ERROR opening:", repr(e))

def run_recommender(dataset):
    print("\nCalling recommend_dataset()")
    try:
        rec = recommend_dataset(dataset)
        print("recommend_dataset returned keys:", list(rec.keys()))
        print(json.dumps(rec, indent=2)[:2000])
    except Exception as e:
        print("recommend_dataset raised:", repr(e))

if __name__ == '__main__':
    dataset = sys.argv[1] if len(sys.argv) > 1 else "assets/sample_images"
    print("Working dir:", Path.cwd())
    files = list_images(dataset)
    if files:
        try_open(files, n=5)
    run_recommender(dataset)
