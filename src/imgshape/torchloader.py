# src/imgshape/torchloader.py
from typing import Iterable, Optional, Dict, Any
from pathlib import Path

def to_torch_transform(augmentation_plan: Dict[str, Any], preprocessing: Dict[str, Any]):
    """
    If torchvision is installed, return a torchvision.transforms.Compose object.
    If not installed, return a string snippet describing the recommended transforms.
    Supports a few canonical names used by AugmentationRecommender: RandomHorizontalFlip, ColorJitter, ClassWiseOversample, RandomCrop.
    """
    # try to import torchvision
    try:
        import torchvision.transforms as T
        has_torch = True
    except Exception:
        has_torch = False

    # Build list of transforms
    transforms = []
    aug_list = augmentation_plan.get("augmentations", []) if isinstance(augmentation_plan, dict) else []

    for a in aug_list:
        name = a.get("name") if isinstance(a, dict) else getattr(a, "name", None)
        params = a.get("params", {}) if isinstance(a, dict) else getattr(a, "params", {})
        if name == "RandomHorizontalFlip":
            p = params.get("p", 0.5)
            if has_torch:
                transforms.append(T.RandomHorizontalFlip(p=p))
            else:
                transforms.append(f"RandomHorizontalFlip(p={p})")
        elif name == "ColorJitter":
            # map ranges to strengths: take middle of range if list provided
            b = params.get("brightness", 0.2)
            c = params.get("contrast", 0.2)
            s = params.get("saturation", 0.2)
            # if values are lists, pick middle
            def _mid(x):
                if isinstance(x, list) and len(x) >= 2:
                    return (x[0] + x[1]) / 2.0
                return x
            b, c, s = _mid(b), _mid(c), _mid(s)
            if has_torch:
                transforms.append(T.ColorJitter(brightness=b, contrast=c, saturation=s))
            else:
                transforms.append(f"ColorJitter(brightness={b}, contrast={c}, saturation={s})")
        elif name == "RandomCrop":
            size = params.get("size", None)
            p = params.get("p", 1.0)
            if has_torch:
                transforms.append(T.RandomResizedCrop(size) if size else T.RandomCrop(224))
            else:
                transforms.append(f"RandomCrop(size={size})")
        else:
            # Unknown augmentation: ignore or annotate
            if has_torch:
                # no-op placeholder
                pass
            else:
                transforms.append(f"UnknownAug({name})")

    # Attach final ToTensor and Normalize if preprocessing suggests it
    if has_torch:
        # basic safe defaults
        transforms.append(T.ToTensor())
        transforms.append(T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]))
        composed = T.Compose(transforms)
        return composed
    else:
        # produce a human-readable snippet
        snippet_lines = [
            "# torchvision.transforms snippet (torch not installed in this env)",
            "from torchvision import transforms",
            "transforms_list = [",
        ]
        for t in transforms:
            snippet_lines.append(f"    #{t},")
        snippet_lines.append("    transforms.ToTensor(),")
        snippet_lines.append("    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])")
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
    Minimal to_dataloader helper: if torch is installed, returns a simple DataLoader
    wrapping an ImageFolder-like ad-hoc dataset constructed from dataset_paths.
    If torch isn't installed, raises ImportError.
    Note: this is a convenience wrapper for development / examples — not a production-grade loader.
    """
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from PIL import Image
    except Exception as e:
        raise ImportError("to_dataloader requires torch and PIL. Install them to enable DataLoader creation.") from e

    # Simple Dataset that expects dataset_paths to be a list of image file paths (or a single directory string)
    class _SimpleImageDataset(Dataset):
        def __init__(self, paths, transform=None):
            self.paths = []
            for p in paths:
                p = Path(p)
                if p.is_dir():
                    for f in p.iterdir():
                        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
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
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return dl
