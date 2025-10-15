# pipeline.py
"""
RecommendationPipeline — v3.0.0
Executable preprocessing + augmentation pipelines
Supports: save/load/export/apply on-disk datasets.
"""
from __future__ import annotations
import os, json, traceback
from typing import List, Callable, Optional, Any, Dict
from PIL import Image
from pathlib import Path

try:
    import torchvision.transforms as tv
    HAS_TORCHVISION = True
except Exception:
    HAS_TORCHVISION = False


class PipelineStep:
    def __init__(self, name: str, fn: Optional[Callable] = None, spec: Optional[Dict] = None):
        self.name = name
        self.fn = fn
        self.spec = spec or {}

    def apply(self, img: Image.Image) -> Image.Image:
        if self.fn:
            return self.fn(img)
        if "resize" in self.spec:
            w, h = self.spec["resize"]
            return img.resize((int(w), int(h)))
        if "rotate" in self.spec:
            return img.rotate(float(self.spec["rotate"]))
        if self.spec.get("flip") == "horizontal":
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def as_dict(self):
        return {"name": self.name, "spec": self.spec}


class RecommendationPipeline:
    def __init__(self, steps: Optional[List[PipelineStep]] = None, meta: Optional[Dict] = None):
        self.steps = steps or []
        self.meta = meta or {}

    def add_step(self, step: PipelineStep):
        self.steps.append(step)

    def as_dict(self):
        return {"meta": self.meta, "steps": [s.as_dict() for s in self.steps]}

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.as_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "RecommendationPipeline":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        steps = [PipelineStep(s["name"], spec=s.get("spec")) for s in data.get("steps", [])]
        return cls(steps=steps, meta=data.get("meta", {}))

    def export(self, format: str = "torchvision") -> str:
        if format == "json":
            return json.dumps(self.as_dict(), indent=2)
        if format == "yaml":
            import yaml
            return yaml.safe_dump(self.as_dict())
        if format == "torchvision":
            if not HAS_TORCHVISION:
                return "# torchvision not installed\n" + json.dumps(self.as_dict(), indent=2)
            lines = ["import torchvision.transforms as transforms", "", "pipeline = transforms.Compose(["]
            for s in self.steps:
                spec = s.spec
                if "resize" in spec:
                    w, h = spec["resize"]
                    lines.append(f"    transforms.Resize(({int(h)}, {int(w)})),")
                if spec.get("flip") == "horizontal":
                    lines.append("    transforms.RandomHorizontalFlip(p=1.0),")
                if "color_jitter" in spec:
                    cj = spec["color_jitter"]
                    lines.append(
                        f"    transforms.ColorJitter({cj.get('brightness',0)},"
                        f"{cj.get('contrast',0)},{cj.get('saturation',0)},{cj.get('hue',0)}),"
                    )
            lines.append("])")
            return "\n".join(lines)
        raise ValueError("unsupported export format")

    def apply(self, input_dir: str, output_dir: str, pattern: str = "*.*", dry_run: bool = False):
        input_dir, output_dir = Path(input_dir), Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        files = [p for p in input_dir.rglob(pattern) if p.suffix.lower() in {".jpg", ".png", ".jpeg"}]
        for src in files:
            dst = output_dir.joinpath(src.relative_to(input_dir))
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dry_run:
                print(f"[DRY] {src} -> {dst}")
                continue
            try:
                img = Image.open(src).convert("RGB")
                for step in self.steps:
                    img = step.apply(img)
                img.save(dst)
            except Exception as e:
                print(f"[ERR] {src}: {e}\n{traceback.format_exc()}")

    @classmethod
    def from_recommender_output(cls, rec: Dict[str, Any]) -> "RecommendationPipeline":
        steps = []
        for p in rec.get("preprocessing", []):
            steps.append(PipelineStep(p.get("name", "pre"), spec=p.get("spec", p)))
        for a in rec.get("augmentations", []):
            steps.append(PipelineStep(a.get("name", "aug"), spec=a.get("spec", a)))
        return cls(steps=steps, meta=rec.get("meta", {}))
