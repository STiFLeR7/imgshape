import os
from src.imgshape.shape import get_shape

def test_get_shape():
    img_path = "assets/sample_images/image_created_with_a_mobile_phone.png"
    if not os.path.exists(img_path):
        print("❌ test.jpg not found in assets/sample_images/")
        return

    shape = get_shape(img_path)
    assert isinstance(shape, tuple)
    assert len(shape) == 3
    assert all(isinstance(x, int) for x in shape)
    print(f"✅ Shape Test Passed: {shape}")

# Run test when executed directly
if __name__ == "__main__":
    test_get_shape()
