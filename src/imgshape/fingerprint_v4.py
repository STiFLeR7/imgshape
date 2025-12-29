"""
imgshape.v4.fingerprint — Dataset Fingerprint Extraction System

This module implements the core fingerprint extraction system for imgshape v4.0.0 (Atlas).
Fingerprints are canonical semantic identities derived deterministically from dataset inspection.

Principles:
- Stable across runs (deterministic)
- Human-readable and machine-actionable
- Schema-versioned (v4.0)
- Never depends on user intent (describes what IS, not what is WANTED)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import math

import numpy as np
from PIL import Image, ImageStat

logger = logging.getLogger("imgshape.fingerprint_v4")


# ============================================================================
# Enumerations for Profile Classifications
# ============================================================================

class ResizeRisk(Enum):
    """Risk level for resizing operations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class GeometryClass(Enum):
    """Classification of geometric diversity"""
    UNIFORM = "uniform"
    DIVERSE = "diverse"
    EXTREME = "extreme"


class CapacityCeiling(Enum):
    """Maximum recommended model capacity"""
    LIGHTWEIGHT = "lightweight"
    STANDARD = "standard"
    HEAVY = "heavy"
    UNLIMITED = "unlimited"


class InformationDensity(Enum):
    """Information density classification"""
    SPARSE = "sparse"
    MODERATE = "moderate"
    DENSE = "dense"
    VERY_DENSE = "very_dense"


class ClassBalanceType(Enum):
    """Class balance classification"""
    BALANCED = "balanced"
    SLIGHTLY_IMBALANCED = "slightly_imbalanced"
    HIGHLY_IMBALANCED = "highly_imbalanced"
    EXTREME_IMBALANCE = "extreme_imbalance"
    UNKNOWN = "unknown"


class SamplingStrategy(Enum):
    """Recommended sampling strategy"""
    UNIFORM = "uniform"
    WEIGHTED = "weighted"
    OVERSAMPLING = "oversampling"
    UNDERSAMPLING = "undersampling"


class LossRecommendation(Enum):
    """Recommended loss function"""
    CROSS_ENTROPY = "cross_entropy"
    FOCAL_LOSS = "focal_loss"
    WEIGHTED_CE = "weighted_ce"
    BALANCED_CE = "balanced_ce"


class QualityLevel(Enum):
    """Overall quality assessment"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"


class WarningSeverity(Enum):
    """Warning severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class WarningCategory(Enum):
    """Warning categories"""
    CORRUPTION = "corruption"
    DUPLICATION = "duplication"
    RESOLUTION = "resolution"
    FORMAT = "format"
    MISSING_DATA = "missing_data"


class PrimaryDomain(Enum):
    """Primary semantic domain"""
    PHOTOGRAPHIC = "photographic"
    MEDICAL = "medical"
    DOCUMENT = "document"
    SYNTHETIC = "synthetic"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ContentType(Enum):
    """Content type classification"""
    NATURAL = "natural"
    TEXT_DENSE = "text_dense"
    LOW_SIGNAL = "low_signal"
    HIGH_ENTROPY = "high_entropy"
    STRUCTURED = "structured"
    UNSTRUCTURED = "unstructured"


# ============================================================================
# Profile Data Classes
# ============================================================================

@dataclass
class Resolution:
    """Resolution specification"""
    width: int
    height: int

    def area(self) -> int:
        return self.width * self.height

    def aspect_ratio(self) -> float:
        return self.width / max(self.height, 1)


@dataclass
class SpatialProfile:
    """Geometric constraints and resizing risk assessment"""
    resolution_range: Dict[str, Resolution]
    aspect_ratio_variance: float
    resize_risk: ResizeRisk
    geometry_class: GeometryClass


@dataclass
class EntropyStats:
    """Entropy statistics"""
    mean: float
    std: float
    range: Dict[str, float]  # min, max


@dataclass
class SignalProfile:
    """Information density and noise characteristics"""
    entropy: EntropyStats
    noise_estimate: float
    capacity_ceiling: CapacityCeiling
    information_density: InformationDensity


@dataclass
class ClassBalance:
    """Class balance information"""
    type: ClassBalanceType
    imbalance_ratio: float
    class_count: int


@dataclass
class DistributionProfile:
    """Dataset structure and imbalance characteristics"""
    class_balance: ClassBalance
    sampling_strategy: SamplingStrategy
    loss_recommendation: LossRecommendation


@dataclass
class QualityWarning:
    """Quality warning"""
    severity: WarningSeverity
    category: WarningCategory
    message: str
    affected_count: int = 0


@dataclass
class QualityProfile:
    """Trustworthiness and data quality assessment"""
    overall_quality: QualityLevel
    warnings: List[QualityWarning]
    corruption_rate: float
    duplication_rate: float


@dataclass
class SemanticProfile:
    """Semantic classification that gates decision branches"""
    primary_domain: PrimaryDomain
    content_type: ContentType
    characteristics: List[str]
    specialization_required: bool


@dataclass
class DatasetFingerprint:
    """
    Complete dataset fingerprint (v4.0).
    
    This is the canonical semantic identity of a dataset.
    It is stable, deterministic, and schema-versioned.
    """
    schema_version: str
    dataset_uri: str
    profiles: Dict[str, Any]  # spatial, signal, distribution, quality, semantic
    derived_class: str
    confidence: float
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        # Convert enums to their string values
        return self._convert_enums_to_strings(result)

    @staticmethod
    def _convert_enums_to_strings(obj: Any) -> Any:
        """Recursively convert enum values to strings"""
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, dict):
            return {k: DatasetFingerprint._convert_enums_to_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [DatasetFingerprint._convert_enums_to_strings(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return DatasetFingerprint._convert_enums_to_strings(asdict(obj))
        return obj


# ============================================================================
# Fingerprint Extractor
# ============================================================================

class FingerprintExtractor:
    """
    Deterministic fingerprint extraction from image datasets.
    
    This class implements the core fingerprint extraction logic according to
    the v4.0 specification. It computes all five profile types and derives
    the canonical dataset class.
    """

    def __init__(self, sample_limit: Optional[int] = None):
        """
        Initialize fingerprint extractor.
        
        Args:
            sample_limit: Optional limit on number of images to analyze
        """
        self.sample_limit = sample_limit
        self.logger = logger

    def extract(self, dataset_path: Path) -> DatasetFingerprint:
        """
        Extract fingerprint from a dataset.
        
        Args:
            dataset_path: Path to dataset directory or image list
            
        Returns:
            DatasetFingerprint object containing all profiles
        """
        # Collect image data
        images = self._collect_images(dataset_path)
        
        if not images:
            raise ValueError(f"No valid images found at {dataset_path}")
        
        self.logger.info(f"Analyzing {len(images)} images from {dataset_path}")
        
        # Extract all profiles
        spatial = self._compute_spatial_profile(images)
        signal = self._compute_signal_profile(images)
        distribution = self._compute_distribution_profile(images)
        quality = self._compute_quality_profile(images)
        semantic = self._compute_semantic_profile(images, spatial, signal)
        
        # Derive canonical class
        derived_class, confidence = self._derive_class(spatial, signal, distribution, quality, semantic)
        
        # Generate dataset URI
        dataset_uri = self._generate_uri(derived_class)
        
        # Build fingerprint
        fingerprint = DatasetFingerprint(
            schema_version="4.0",
            dataset_uri=dataset_uri,
            profiles={
                "spatial": spatial,
                "signal": signal,
                "distribution": distribution,
                "quality": quality,
                "semantic": semantic
            },
            derived_class=derived_class,
            confidence=confidence,
            metadata={
                "sample_count": len(images),
                "imgshape_version": "4.0.0"
            }
        )
        
        return fingerprint

    def _collect_images(self, dataset_path: Path) -> List[Dict[str, Any]]:
        """
        Collect image metadata and statistics.
        
        Returns list of dicts with: path, size, entropy, etc.
        """
        images = []
        
        if dataset_path.is_file():
            # Single image
            img_data = self._analyze_single_image(dataset_path)
            if img_data:
                images.append(img_data)
        else:
            # Directory
            image_files = self._find_images(dataset_path)
            
            count = 0
            for img_path in image_files:
                if self.sample_limit and count >= self.sample_limit:
                    break
                    
                img_data = self._analyze_single_image(img_path)
                if img_data:
                    images.append(img_data)
                    count += 1
        
        return images

    def _find_images(self, directory: Path) -> List[Path]:
        """Find all image files in directory"""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        images = []
        
        for ext in extensions:
            images.extend(directory.rglob(f"*{ext}"))
            images.extend(directory.rglob(f"*{ext.upper()}"))
        
        return images

    def _analyze_single_image(self, img_path: Path) -> Optional[Dict[str, Any]]:
        """Analyze a single image and return metadata"""
        try:
            with Image.open(img_path) as img:
                # Convert to RGB for consistent analysis
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Compute statistics
                stat = ImageStat.Stat(img)
                
                # Shannon entropy (approximation from variance)
                variance = stat.var
                entropy = sum([math.log2(v + 1) for v in variance]) / len(variance)
                
                return {
                    'path': img_path,
                    'size': (img.width, img.height),
                    'mode': img.mode,
                    'entropy': entropy,
                    'mean': stat.mean,
                    'variance': variance,
                    'extrema': stat.extrema
                }
        except Exception as e:
            self.logger.warning(f"Failed to analyze {img_path}: {e}")
            return None

    def _compute_spatial_profile(self, images: List[Dict[str, Any]]) -> SpatialProfile:
        """Compute spatial profile from image data"""
        sizes = [img['size'] for img in images]
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        aspect_ratios = [w / h for w, h in sizes]
        
        # Resolution range
        min_res = Resolution(min(widths), min(heights))
        max_res = Resolution(max(widths), max(heights))
        median_res = Resolution(int(np.median(widths)), int(np.median(heights)))
        
        # Aspect ratio variance
        ar_variance = float(np.var(aspect_ratios))
        
        # Determine geometry class
        if ar_variance < 0.01:
            geometry_class = GeometryClass.UNIFORM
        elif ar_variance < 0.1:
            geometry_class = GeometryClass.DIVERSE
        else:
            geometry_class = GeometryClass.EXTREME
        
        # Determine resize risk
        area_variance = np.var([w * h for w, h in sizes])
        if area_variance < 10000 and ar_variance < 0.01:
            resize_risk = ResizeRisk.LOW
        elif area_variance < 100000 and ar_variance < 0.1:
            resize_risk = ResizeRisk.MEDIUM
        else:
            resize_risk = ResizeRisk.HIGH
        
        return SpatialProfile(
            resolution_range={
                'min': min_res,
                'max': max_res,
                'median': median_res
            },
            aspect_ratio_variance=ar_variance,
            resize_risk=resize_risk,
            geometry_class=geometry_class
        )

    def _compute_signal_profile(self, images: List[Dict[str, Any]]) -> SignalProfile:
        """Compute signal profile from image data"""
        entropies = [img['entropy'] for img in images]
        
        entropy_stats = EntropyStats(
            mean=float(np.mean(entropies)),
            std=float(np.std(entropies)),
            range={'min': float(min(entropies)), 'max': float(max(entropies))}
        )
        
        # Estimate noise from variance
        variances = [np.mean(img['variance']) for img in images]
        noise_estimate = min(1.0, float(np.mean(variances)) / 10000.0)
        
        # Determine information density
        mean_entropy = entropy_stats.mean
        if mean_entropy < 5:
            info_density = InformationDensity.SPARSE
        elif mean_entropy < 7:
            info_density = InformationDensity.MODERATE
        elif mean_entropy < 9:
            info_density = InformationDensity.DENSE
        else:
            info_density = InformationDensity.VERY_DENSE
        
        # Determine capacity ceiling
        if mean_entropy < 6 and noise_estimate < 0.3:
            capacity = CapacityCeiling.LIGHTWEIGHT
        elif mean_entropy < 8:
            capacity = CapacityCeiling.STANDARD
        elif mean_entropy < 10:
            capacity = CapacityCeiling.HEAVY
        else:
            capacity = CapacityCeiling.UNLIMITED
        
        return SignalProfile(
            entropy=entropy_stats,
            noise_estimate=noise_estimate,
            capacity_ceiling=capacity,
            information_density=info_density
        )

    def _compute_distribution_profile(self, images: List[Dict[str, Any]]) -> DistributionProfile:
        """Compute distribution profile from image data"""
        # For now, assume unlabeled dataset (class_count=0)
        # In full implementation, this would analyze directory structure or labels
        
        class_balance = ClassBalance(
            type=ClassBalanceType.UNKNOWN,
            imbalance_ratio=1.0,
            class_count=0
        )
        
        sampling_strategy = SamplingStrategy.UNIFORM
        loss_recommendation = LossRecommendation.CROSS_ENTROPY
        
        return DistributionProfile(
            class_balance=class_balance,
            sampling_strategy=sampling_strategy,
            loss_recommendation=loss_recommendation
        )

    def _compute_quality_profile(self, images: List[Dict[str, Any]]) -> QualityProfile:
        """Compute quality profile from image data"""
        warnings = []
        
        # Check for resolution issues
        sizes = [img['size'] for img in images]
        min_dimension = min([min(s) for s in sizes])
        
        if min_dimension < 32:
            warnings.append(QualityWarning(
                severity=WarningSeverity.WARNING,
                category=WarningCategory.RESOLUTION,
                message="Very small images detected (< 32px)",
                affected_count=sum(1 for s in sizes if min(s) < 32)
            ))
        
        # Corruption rate (images that failed to load would not be in list)
        corruption_rate = 0.0
        
        # Duplication rate (simple placeholder - would need perceptual hashing)
        duplication_rate = 0.0
        
        # Overall quality assessment
        if len(warnings) == 0:
            overall_quality = QualityLevel.EXCELLENT
        elif len(warnings) <= 2:
            overall_quality = QualityLevel.GOOD
        else:
            overall_quality = QualityLevel.ACCEPTABLE
        
        return QualityProfile(
            overall_quality=overall_quality,
            warnings=warnings,
            corruption_rate=corruption_rate,
            duplication_rate=duplication_rate
        )

    def _compute_semantic_profile(
        self, 
        images: List[Dict[str, Any]],
        spatial: SpatialProfile,
        signal: SignalProfile
    ) -> SemanticProfile:
        """Compute semantic profile using heuristics"""
        
        # Determine primary domain (heuristic-based for now)
        # In full implementation, this could use lightweight classification
        
        # Check for grayscale (medical/document indicator)
        modes = [img['mode'] for img in images]
        grayscale_ratio = sum(1 for m in modes if m in ('L', 'LA')) / len(modes)
        
        # Check for high entropy (photographic indicator)
        mean_entropy = signal.entropy.mean
        
        characteristics = []
        
        if grayscale_ratio > 0.5:
            characteristics.append("grayscale")
            if mean_entropy < 7:
                primary_domain = PrimaryDomain.MEDICAL
                content_type = ContentType.LOW_SIGNAL
                specialization_required = True
            else:
                primary_domain = PrimaryDomain.DOCUMENT
                content_type = ContentType.TEXT_DENSE
                specialization_required = True
        else:
            characteristics.append("color")
            if mean_entropy > 8:
                primary_domain = PrimaryDomain.PHOTOGRAPHIC
                content_type = ContentType.HIGH_ENTROPY
                specialization_required = False
            else:
                primary_domain = PrimaryDomain.SYNTHETIC
                content_type = ContentType.STRUCTURED
                specialization_required = False
        
        return SemanticProfile(
            primary_domain=primary_domain,
            content_type=content_type,
            characteristics=characteristics,
            specialization_required=specialization_required
        )

    def _derive_class(
        self,
        spatial: SpatialProfile,
        signal: SignalProfile,
        distribution: DistributionProfile,
        quality: QualityProfile,
        semantic: SemanticProfile
    ) -> Tuple[str, float]:
        """
        Derive canonical dataset class from profiles.
        
        Returns:
            (class_name, confidence)
        """
        # Build class name from semantic profile
        domain = semantic.primary_domain.value
        content = semantic.content_type.value
        
        derived_class = f"vision.{domain}.{content}"
        
        # Compute confidence based on profile consistency
        confidence = 0.9  # Base confidence
        
        # Reduce confidence for quality issues
        if quality.overall_quality in (QualityLevel.POOR, QualityLevel.CRITICAL):
            confidence -= 0.2
        
        # Reduce confidence for mixed characteristics
        if semantic.primary_domain == PrimaryDomain.MIXED:
            confidence -= 0.1
        
        confidence = max(0.0, min(1.0, confidence))
        
        return derived_class, confidence

    def _generate_uri(self, derived_class: str) -> str:
        """Generate canonical dataset URI"""
        return f"imgshape://{derived_class.replace('.', '/')}"
