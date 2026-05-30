from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union, Dict


@dataclass
class DatasetClauses:
    expected_channels: Optional[int] = None
    allowed_formats: Optional[List[str]] = None
    resolution_min: Optional[Union[Tuple[int, int], List[int]]] = None
    resolution_max: Optional[Union[Tuple[int, int], List[int]]] = None


@dataclass
class QualityClauses:
    corruption_max: Optional[float] = None
    duplicate_max: Optional[float] = None
    blur_threshold: Optional[float] = None


@dataclass
class DistributionClauses:
    entropy_min: Optional[float] = None
    entropy_max: Optional[float] = None
    channel_consistency: Optional[str] = None  # e.g., 'strict'


@dataclass
class ContractClauses:
    dataset: DatasetClauses = field(default_factory=DatasetClauses)
    quality: QualityClauses = field(default_factory=QualityClauses)
    distribution: DistributionClauses = field(default_factory=DistributionClauses)
    severities: Dict[str, str] = field(default_factory=dict)
