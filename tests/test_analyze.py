# test_analyze.py
from src.imgshape.analyze import analyze_type

def test_analyze_type():
    result = analyze_type("assets/sample_images/image_created_with_a_mobile_phone.png")
    assert isinstance(result, dict)
    assert "entropy" in result
    assert "guess_type" in result
    print(f"✅ Analyze Test Passed: {result}")

if __name__ == "__main__":
    test_analyze_type()
