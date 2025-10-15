# plugins/__init__.py
"""
Plugin architecture for imgshape v3.0.0
Allows drop-in analyzers, recommenders, exporters.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
from typing import List, Dict, Any


class PluginBase:
    NAME = "base"


class AnalyzerPlugin(PluginBase):
    def analyze(self, dataset_path: str) -> Dict[str, Any]:
        raise NotImplementedError


class RecommenderPlugin(PluginBase):
    def recommend(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class ExporterPlugin(PluginBase):
    def export(self, pipeline: Dict[str, Any]) -> str:
        raise NotImplementedError


def load_plugins_from_dir(path: str) -> List[PluginBase]:
    path = Path(path)
    plugins = []
    for file in path.glob("*.py"):
        if file.name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(file.stem, str(file))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
        for attr in dir(module):
            obj = getattr(module, attr)
            if isinstance(obj, type) and issubclass(obj, PluginBase) and obj is not PluginBase:
                plugins.append(obj())
    return plugins
