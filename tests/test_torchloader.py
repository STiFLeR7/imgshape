# tests/test_torchloader.py
import sys
import types
import pytest

from imgshape.torchloader import to_torch_transform


def test_to_torch_transform_with_fake_torchvision(monkeypatch):
    """Simulate torchvision available and ensure we get a callable back."""

    # Fake torchvision.transforms module with a minimal Compose
    fake_T = types.SimpleNamespace()

    class FakeCompose:
        def __init__(self, items=None):
            self.items = items or []
        def __call__(self, x):  # make it callable
            return x

    fake_T.Compose = FakeCompose
    fake_T.RandomHorizontalFlip = lambda p=0.5: (p, "flip")
    fake_T.ColorJitter = lambda **kwargs: ("jitter", kwargs)
    fake_T.RandomResizedCrop = lambda size: ("crop", size)
    fake_T.RandomCrop = lambda size: ("crop", size)
    fake_T.ToTensor = lambda: "to_tensor"
    fake_T.Normalize = lambda mean, std: ("norm", mean, std)

    monkeypatch.setitem(sys.modules, "torchvision.transforms", fake_T)
    monkeypatch.setitem(sys.modules, "torchvision", types.SimpleNamespace(transforms=fake_T))

    plan = {"order": [], "augmentations": [{"name": "RandomHorizontalFlip", "params": {"p": 0.3}}]}
    preprocessing = {"normalize": {"mean": [0.5], "std": [0.5]}}

    tfm = to_torch_transform(plan, preprocessing)
    assert callable(tfm)
    # our fake Compose returns a FakeCompose, so check type
    assert isinstance(tfm, fake_T.Compose)


def test_to_torch_transform_with_only_torch(monkeypatch):
    """Simulate torch present but torchvision missing -> should return a no-op callable."""

    monkeypatch.setitem(sys.modules, "torchvision", None)
    monkeypatch.setitem(sys.modules, "torchvision.transforms", None)

    plan = {"order": [], "augmentations": []}
    preprocessing = {}

    tfm = to_torch_transform(plan, preprocessing)
    assert callable(tfm)
    dummy = object()
    assert tfm(dummy) is dummy  # no-op


def test_to_torch_transform_no_torch(monkeypatch):
    """Simulate no torch at all -> should return snippet string."""

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "torchvision", None)
    monkeypatch.setitem(sys.modules, "torchvision.transforms", None)

    plan = {"order": [], "augmentations": []}
    preprocessing = {}

    tfm = to_torch_transform(plan, preprocessing)
    assert isinstance(tfm, str)
    assert "transforms.Compose" in tfm
