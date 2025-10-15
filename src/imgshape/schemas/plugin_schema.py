# schemas/plugin_schema.py
from pydantic import BaseModel
from typing import Dict, Any, Optional

class PluginMetadata(BaseModel):
    name: str
    version: Optional[str]
    description: Optional[str]

class PluginSpec(BaseModel):
    metadata: PluginMetadata
    type: str  # analyzer|recommender|exporter
    entry_point: str
    config: Optional[Dict[str, Any]]
