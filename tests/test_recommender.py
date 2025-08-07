# tests/test_recommender.py

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src.imgshape.recommender import recommend_preprocessing


def test_recommend():
    result = recommend_preprocessing("assets/sample_images/image_created_with_a_mobile_phone.png")
    assert isinstance(result, dict)
    assert "resize" in result
    assert "normalize" in result
    print(f"✅ Recommender Test Passed: {result}")

if __name__ == "__main__":
    test_recommend()
