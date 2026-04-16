import pytest
import numpy as np
from pathlib import Path
from PIL import Image
import shutil
import os

def test_atlas_batched_gpu():
    from imgshape.atlas import Atlas
    
    # Create a small dummy dataset
    tmp_dir = Path("tmp_test_dataset_gpu")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        for i in range(5):
            img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(tmp_dir / f"test_{i}.jpg")
    
        # Test initialization with new params
        # This should fail if Atlas.__init__ is not updated
        atlas = Atlas(sample_limit=10, use_gpu=True, batch_size=2)
        
        fingerprint = atlas.extract_fingerprint(tmp_dir)
        
        assert fingerprint.metadata["acceleration"] in ["gpu", "cpu"]
        assert "signal" in fingerprint.profiles
        
        # Verify entropy and blur are not 0 (meaning they were computed)
        signal = fingerprint.profiles["signal"]
        assert signal.entropy.mean > 0
        
        # Check that we have 5 images processed
        assert fingerprint.metadata["sample_count"] == 5
        
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

def test_atlas_no_gpu_backward_compatibility():
    from imgshape.atlas import Atlas
    import numpy as np
    from pathlib import Path
    from PIL import Image
    import shutil
    
    # Create a small dummy dataset
    tmp_dir = Path("tmp_test_dataset_no_gpu")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        for i in range(3):
            img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(tmp_dir / f"test_{i}.jpg")
    
        # Test initialization WITHOUT new params (default values)
        atlas = Atlas(sample_limit=10)
        
        fingerprint = atlas.extract_fingerprint(tmp_dir)
        
        assert "signal" in fingerprint.profiles
        assert fingerprint.metadata["sample_count"] == 3
        
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    test_atlas_batched_gpu()
    test_atlas_no_gpu_backward_compatibility()
