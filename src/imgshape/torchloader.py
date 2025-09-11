from typing import Iterable, Optional, Dict, Any
from pathlib import Path


def to_torch_transform(augmentation_plan: Dict[str, Any], preprocessing: Dict[str, Any]):
    """
    Return either:
      - torchvision.transforms.Compose (if torchvision available), OR
      - a callable no-op transform (if torch available but torchvision missing), OR
      - a string snippet (if neither torch nor torchvision available).
    """
    has_torch = False
    has_torchvision = False

    try:
        import torch  # noqa: F401
        has_torch = True
    except Exception:
        has_torch = False

    try:
        import torchvision.transforms as T  # noqa: F401
        has_torchvision = True
    except Exception:
        has_torchvision = False

    # Case 1: torchvision available
    if has_torchvision:
        import torchvision.transforms as T

        transforms_list = []
        aug_list = augmentation_plan.get("augmentations", []) if isinstance(augmentation_plan, dict) else []

        for a in aug_list:
            name = a.get("name") if isinstance(a, dict) else getattr(a, "name", None)
            params = a.get("params", {}) if isinstance(a, dict) else getattr(a, "params", {})

            if name == "RandomHorizontalFlip":
                p = params.get("p", 0.5)
                transforms_list.append(T.RandomHorizontalFlip(p=p))

            elif name == "ColorJitter":
                def _mid(x):
                    if isinstance(x, list) and len(x) >= 2:
                        return (x[0] + x[1]) / 2.0
                    return x
                b = _mid(params.get("brightness", 0.2))
                c = _mid(params.get("contrast", 0.2))
                s = _mid(params.get("saturation", 0.2))
                transforms_list.append(T.ColorJitter(brightness=b, contrast=c, saturation=s))

            elif name == "RandomCrop":
                size = params.get("size", None)
                if size:
                    transforms_list.append(T.RandomResizedCrop(size))
                else:
                    transforms_list.append(T.RandomCrop(224))

        # Always add tensor conversion + normalization
        transforms_list.append(T.ToTensor())
        transforms_list.append(
            T.Normalize(
                mean=preprocessing.get("normalize", {}).get("mean", [0.485, 0.456, 0.406]),
                std=preprocessing.get("normalize", {}).get("std", [0.229, 0.224, 0.225]),
            )
        )
        return T.Compose(transforms_list)

    # Case 2: torch present but torchvision missing → callable no-op
    if has_torch and not has_torchvision:
        def _noop_transform(img):
            """No-op transform (returns image unchanged)."""
            return img
        return _noop_transform

    # Case 3: neither torch nor torchvision → snippet string
    aug_list = augmentation_plan.get("augmentations", []) if isinstance(augmentation_plan, dict) else []
    snippet_lines = [
        "# torchvision.transforms snippet (torch/torchvision not installed in this env)",
        "from torchvision import transforms",
        "transforms_list = [",
    ]
    for a in aug_list:
        snippet_lines.append(
            f"    # {a.get('name') if isinstance(a, dict) else getattr(a, 'name', None)},"
        )
    snippet_lines.append("    transforms.ToTensor(),")
    snippet_lines.append(
        "    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])"
    )
    snippet_lines.append("]")
    snippet_lines.append("transform = transforms.Compose(transforms_list)")
    return "\n".join(snippet_lines)


def to_dataloader(
    dataset_paths: Iterable[str],
    labels: Optional[Iterable[int]] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    augmentation_plan: Optional[Dict[str, Any]] = None,
    preprocessing: Optional[Dict[str, Any]] = None,
    pin_memory: bool = False,
):
    """
    Minimal to_dataloader helper.

    If torch is installed, returns a DataLoader wrapping a simple Dataset
    built from dataset_paths. If torch isn't installed, raises ImportError.

    This is for prototyping and examples — not a production-grade loader.
    """
    try:
        import torch  # noqa: F401
        from torch.utils.data import DataLoader, Dataset
        from PIL import Image
    except Exception as e:
        raise ImportError(
            "to_dataloader requires torch and PIL. Install them to enable DataLoader creation."
        ) from e

    class _SimpleImageDataset(Dataset):
        def __init__(self, paths, transform=None):
            self.paths = []
            for p in paths:
                p = Path(p)
                if p.is_dir():
                    for f in p.iterdir():
                        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".png"}:
                            self.paths.append(str(f))
                elif p.is_file():
                    self.paths.append(str(p))
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            p = self.paths[idx]
            img = Image.open(p).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img

    transform = to_torch_transform(augmentation_plan or {}, preprocessing or {})
    ds = _SimpleImageDataset(list(dataset_paths), transform=transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
