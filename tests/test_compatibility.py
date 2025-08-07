from src.imgshape.compatibility import check_model_compatibility

def test_check_model_compatibility():
    result = check_model_compatibility("assets/sample_images", model="mobilenet_v2")
    assert isinstance(result, dict)
    assert "total" in result
    print(f"✅ Compatibility Test Passed: {result['passed']}/{result['total']} passed")

if __name__ == "__main__":
    test_check_model_compatibility()
