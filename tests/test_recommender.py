# tests/test_recommender.py
import sys
import os
from pathlib import Path

# allow running tests from project root: ensure src/ is on sys.path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.joinpath("src")
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest

from imgshape.recommender import recommend_preprocessing, recommend_dataset, RecommendEngine


@pytest.mark.parametrize("size", [(300, 300), (32, 32)])
def test_recommend_preprocessing_basic(tmp_path: Path, size):
    """
    Smoke test for recommend_preprocessing.
    - Creates a tiny image at tmp_path (if Pillow is available).
    - Calls recommend_preprocessing and asserts expected keys are present.
    """
    img_path = tmp_path / "img.png"

    # create a tiny image using Pillow if available
    try:
        from PIL import Image
        img = Image.new("RGB", size, color=(123, 50, 200))
        img.save(img_path, format="PNG")
        created = True
    except Exception:
        # Pillow missing: we still exercise fallback behavior by passing a non-image path
        created = False
        img_path.write_text("not-an-image", encoding="utf-8")

    result = recommend_preprocessing(str(img_path))

    assert isinstance(result, dict), "recommend_preprocessing must return a dict"
    # keys expected by downstream code / report generation
    assert "resize" in result, "Missing 'resize' key in recommendation"
    assert "normalize" in result, "Missing 'normalize' key in recommendation"
    assert "entropy" in result or result.get("error") == "fallback", "Missing 'entropy' (or fallback) in recommendation"


def test_recommend_dataset_and_engine(tmp_path: Path):
    """
    Test recommend_dataset on a tiny directory and RecommendEngine wrapper.
    """
    data_dir = tmp_path / "dataset"
    data_dir.mkdir()

    # create a few sample images (if Pillow available)
    pillow_ok = False
    try:
        from PIL import Image
        pillow_ok = True
        for i, s in enumerate([(64, 64), (128, 128), (64, 64)]):
            p = data_dir / f"img_{i}.png"
            Image.new("RGB", s, color=(i * 30 + 10, 20, 30)).save(p)
    except Exception:
        # fallback: create placeholder files (recommend_dataset should fall back)
        for i in range(2):
            (data_dir / f"file_{i}.txt").write_text("x")

    ds_res = recommend_dataset(str(data_dir), sample_limit=10)
    assert isinstance(ds_res, dict)
    assert "dataset_summary" in ds_res
    assert "representative_preprocessing" in ds_res

    # Test RecommendEngine methods
    engine = RecommendEngine()  # no profile
    # dataset wrapper
    rec_ds = engine.recommend_from_dataset(str(data_dir))
    assert isinstance(rec_ds, dict)
    # image wrapper: pick first image if pillow created
    if pillow_ok:
        first_img = next(data_dir.glob("*.png"))
        from PIL import Image as PILImage
        pil = PILImage.open(first_img).convert("RGB")
        rec_img = engine.recommend_from_image(pil)
        assert isinstance(rec_img, dict)
        assert "resize" in rec_img or "augmentation_plan" in rec_img

    # recommend_from_bytes: read first file as bytes if exists
    some_file = next(data_dir.iterdir(), None)
    if some_file is not None:
        b = some_file.read_bytes()
        rec_bytes = engine.recommend_from_bytes(b)
        assert isinstance(rec_bytes, dict)


if __name__ == "__main__":
    # allow running directly
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
