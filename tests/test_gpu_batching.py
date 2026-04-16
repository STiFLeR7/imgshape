import pytest
import torch
import numpy as np
from imgshape.gpu_v4 import BatchedGPUHandler

def test_batch_collection():
    handler = BatchedGPUHandler(batch_size=4)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Should not trigger execution yet
    res = handler.add_image(img)
    assert res is None
    assert len(handler.queue) == 1
    
    # Fill batch
    handler.add_image(img)
    handler.add_image(img)
    results = handler.add_image(img)
    
    assert results is not None
    assert len(results) == 4
    assert len(handler.queue) == 0
