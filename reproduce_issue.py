
import torch
import numpy as np
from imgshape.gpu_v4 import BatchedGPUHandler

def test_heterogeneous_batch():
    handler = BatchedGPUHandler(batch_size=2)
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = np.zeros((200, 200, 3), dtype=np.uint8)
    
    handler.add_image(img1)
    try:
        results = handler.add_image(img2)
        print("Success (Unexpected)")
    except Exception as e:
        print(f"Caught expected error: {e}")

if __name__ == "__main__":
    test_heterogeneous_batch()
