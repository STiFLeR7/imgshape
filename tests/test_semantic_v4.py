import torch
import numpy as np
import pytest
from imgshape.semantic_v4 import SemanticExtractor

def test_semantic_embedding():
    extractor = SemanticExtractor(model_name="squeezenet1_1")
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    embedding = extractor.extract(img)
    
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (512,)
