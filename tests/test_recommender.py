# tests/test_recommender.py

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from imgshape.recommender import recommend_preprocessing


def test_recommend():
    """Test recommend_preprocessing with fallback to a safe image path."""
    test_image_path = "assets/sample_images/Image_created_with_a_mobile_phone.png"

    # If local asset missing, use a known small generated image
    if not os.path.exists(test_image_path):
        from PIL import Image
        os.makedirs(os.path.dirname(test_image_path), exist_ok=True)
        img = Image.new("RGB", (300, 300), color=(255, 0, 0))
        img.save(test_image_path)

    result = recommend_preprocessing(test_image_path)

    assert isinstance(result, dict), "Output should be a dictionary"
    assert "resize" in result, "Missing 'resize' key"
    assert "normalize" in result, "Missing 'normalize' key"
    assert "entropy" in result, "Missing 'entropy' key"

    print(f"✅ Recommender Test Passed: {result}")


if __name__ == "__main__":
    test_recommend()
