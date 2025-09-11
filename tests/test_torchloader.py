import pytest
torch = pytest.importorskip("torch")
from imgshape.torchloader import to_torch_transform
def test_to_torch_transform_returns_callable():
    plan = {"order": [], "augmentations": []}
    transforms = to_torch_transform(plan, {})
    assert callable(transforms)
