import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from PIL import Image, ImageStat

from imgshape.atlas.gpu import GPUHandler, BatchedGPUHandler
from imgshape.atlas.spatial import (
    SpatialProfiler,
    SpatialProfile,
)
from imgshape.atlas.signal import (
    SignalProfiler,
    SignalProfile,
)
from imgshape.atlas.distribution import (
    DistributionProfiler,
    DistributionProfile,
)
from imgshape.atlas.quality import (
    QualityProfiler,
    QualityProfile,
    QualityLevel,
)
from imgshape.atlas.semantic import (
    SemanticProfiler,
    SemanticProfile,
    PrimaryDomain,
    MedicalProfile,
    SatelliteProfile,
    OCRProfile,
)

logger = logging.getLogger("imgshape.atlas.orchestrator")


@dataclass
class ProfilesContainer:
    """Type-safe container for the 5 profile types in v5.0.0"""

    spatial: SpatialProfile
    signal: SignalProfile
    distribution: DistributionProfile
    quality: QualityProfile
    semantic: SemanticProfile
    medical: Optional[MedicalProfile] = None
    satellite: Optional[SatelliteProfile] = None
    ocr: Optional[OCRProfile] = None

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            val = getattr(self, key)
            if val is not None:
                return val
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) and getattr(self, key) is not None


@dataclass
class DatasetFingerprint:
    """
    Complete dataset fingerprint (v5.0.0).

    This is the canonical semantic identity of a dataset.
    It is stable, deterministic, and schema-versioned.
    """

    schema_version: str
    dataset_uri: str
    profiles: Union[ProfilesContainer, Dict[str, Any]]
    derived_class: str
    confidence: float
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        return self._convert_enums_to_strings(result)

    @staticmethod
    def _convert_enums_to_strings(obj: Any) -> Any:
        """Recursively convert enum values to strings"""
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, dict):
            return {
                k: DatasetFingerprint._convert_enums_to_strings(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [DatasetFingerprint._convert_enums_to_strings(item) for item in obj]
        elif hasattr(obj, "__dict__"):
            return DatasetFingerprint._convert_enums_to_strings(asdict(obj))
        return obj


def _compute_entropy(img_array: np.ndarray) -> float:
    """Compute Shannon entropy on CPU."""
    _, counts = np.unique(img_array, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy)


def _compute_blur(img_array: np.ndarray) -> float:
    """Compute blur score (Laplacian variance) on CPU."""
    if len(img_array.shape) == 3:
        # Use standard ITU-R 601-2 luma weights matching GPU path
        weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
        gray = np.dot(img_array[..., :3], weights).astype(np.float32)
    else:
        gray = img_array.astype(np.float32)

    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    from scipy.signal import convolve2d

    laplacian = convolve2d(gray, kernel, mode="same")
    return float(np.var(laplacian))


class Atlas:
    """
    Main orchestrator for the refactored Atlas fingerprint engine (v5.0.0).

    This class coordinates the complete pipeline:
    1. Scan and collect image metadata
    2. Batch process image arrays for entropy and blur estimation
    3. Run individual profilers (spatial, signal, distribution, quality, semantic)
    4. Compute specialized profiles if applicable
    5. Derive the canonical dataset class name and confidence
    6. Return a typed DatasetFingerprint
    """

    def __init__(
        self,
        sample_limit: Optional[int] = None,
        use_gpu: bool = False,
        batch_size: int = 32,
        rule_version: str = "5.0.0",
        validate: bool = True,
        **kwargs,
    ):
        self.sample_limit = sample_limit
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.rule_version = rule_version
        self.validate_outputs = validate

    def extract_fingerprint(self, dataset_path: Union[str, Path]) -> Any:
        """
        Extract only the fingerprint (no decisions) - backward compatibility.
        """
        import warnings

        warnings.warn(
            (
                "extract_fingerprint() is deprecated and will be removed in v6.0.0. "
                "Use extract() instead."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        from imgshape.fingerprint_v4 import FingerprintExtractor

        extractor = FingerprintExtractor(
            sample_limit=self.sample_limit,
            use_gpu=self.use_gpu,
            batch_size=self.batch_size,
        )
        return extractor.extract(dataset_path)

    def compare(self, baseline_path: Path, current_path: Path) -> Dict[str, Any]:
        """
        Compare two datasets - backward compatibility.
        """
        import warnings

        warnings.warn(
            (
                "Atlas.compare() is deprecated and will be removed in v6.0.0. "
                "Use DatasetDiff.compare() instead."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        from imgshape.fingerprint_v4 import FingerprintExtractor
        from imgshape.compare_v4 import DatasetComparator

        extractor = FingerprintExtractor(
            sample_limit=self.sample_limit,
            use_gpu=self.use_gpu,
            batch_size=self.batch_size,
        )
        comparator = DatasetComparator()
        f1 = extractor.extract(baseline_path)
        f2 = extractor.extract(current_path)
        return comparator.compare(f1, f2)

    def extract(self, dataset_path: Union[str, Path]) -> DatasetFingerprint:
        """
        Extract the dataset fingerprint.
        """
        path = Path(dataset_path)
        images = self._collect_images(path)
        failed_count = getattr(self, "last_failed_count", 0)

        if not images:
            raise ValueError(f"No valid images found at {dataset_path}")

        logger.info(f"Extracting profiles for {len(images)} images.")

        # Step 3: Run individual profilers
        spatial = SpatialProfiler.profile(images)
        signal = SignalProfiler.profile(images)
        distribution = DistributionProfiler.profile(images)
        quality = QualityProfiler.profile(images, failed_count=failed_count)
        semantic = SemanticProfiler.profile(images, spatial, signal)

        # Step 4: Compute specialized profile if applicable
        specialized = None
        if semantic.primary_domain == PrimaryDomain.MEDICAL:
            specialized = SemanticProfiler.profile_medical(images)
        elif semantic.primary_domain == PrimaryDomain.SATELLITE:
            specialized = SemanticProfiler.profile_satellite(images)
        elif semantic.primary_domain == PrimaryDomain.OCR:
            specialized = SemanticProfiler.profile_ocr(images)

        # Step 5: Derive canonical class name and confidence
        derived_class, confidence = self._derive_class(
            spatial, signal, distribution, quality, semantic
        )
        dataset_uri = self._generate_uri(derived_class)

        profiles = ProfilesContainer(
            spatial=spatial,
            signal=signal,
            distribution=distribution,
            quality=quality,
            semantic=semantic,
        )

        if specialized:
            if semantic.primary_domain == PrimaryDomain.MEDICAL:
                profiles.medical = specialized
            elif semantic.primary_domain == PrimaryDomain.SATELLITE:
                profiles.satellite = specialized
            elif semantic.primary_domain == PrimaryDomain.OCR:
                profiles.ocr = specialized

        # Collect formats and channel_counts
        formats_set = set()
        for img in images:
            if "path" in img:
                ext = Path(img["path"]).suffix.lower().lstrip(".")
                if ext:
                    formats_set.add(ext)
        formats = sorted(list(formats_set))

        all_grayscale = all(img.get("is_grayscale", False) for img in images)
        none_grayscale = all(not img.get("is_grayscale", False) for img in images)
        channel_counts = 1 if all_grayscale else 3

        # Compute provenance ID
        from imgshape.audit.provenance import generate_provenance_id

        try:
            provenance_id = generate_provenance_id(path)
        except Exception as e:
            logger.warning(f"Failed to generate provenance ID: {e}")
            import uuid

            provenance_id = f"prov_err_{uuid.uuid4().hex[:8]}"

        # Step 6: Return DatasetFingerprint
        return DatasetFingerprint(
            schema_version="5.0.0",
            dataset_uri=dataset_uri,
            profiles=profiles,
            derived_class=derived_class,
            confidence=confidence,
            metadata={
                "sample_count": len(images),
                "imgshape_version": "5.0.0",
                "acceleration": (
                    "gpu"
                    if (self.use_gpu and GPUHandler.is_cuda_available())
                    else "cpu"
                ),
                "formats": formats,
                "channel_count": channel_counts,
                "channel_counts": channel_counts,
                "grayscale_consistency": all_grayscale or none_grayscale,
                "grayscale_ratio": sum(
                    1 if img.get("is_grayscale", False) else 0 for img in images
                )
                / len(images),
                "provenance_id": provenance_id,
            },
        )

    def analyze(
        self,
        dataset_path: Union[str, Path],
        user_intent: Optional[Any] = None,
        output_dir: Optional[Union[str, Path]] = None,
        constraints: Optional[Any] = None,
    ) -> Union[DatasetFingerprint, Dict[str, Any]]:
        """
        Extract dataset fingerprint, or run complete analysis pipeline for backward compatibility.
        """
        if user_intent is not None or output_dir is not None:
            # Run v4 pipeline for backward compatibility
            from imgshape.fingerprint_v4 import FingerprintExtractor
            from imgshape.decision_v4 import DecisionEngine
            from imgshape.artifacts_v4 import ArtifactGenerator
            from imgshape.validator_v4 import SchemaValidator

            extractor = FingerprintExtractor(
                sample_limit=self.sample_limit,
                use_gpu=self.use_gpu,
                batch_size=self.batch_size,
            )
            rule_version = getattr(self, "rule_version", "4.2.0")
            engine = DecisionEngine(rule_version=rule_version)
            validator = (
                SchemaValidator() if getattr(self, "validate_outputs", True) else None
            )

            fingerprint = extractor.extract(dataset_path)
            if validator:
                try:
                    data = fingerprint.to_dict()
                    validator.validate_fingerprint(data)
                except Exception as e:
                    logger.warning(f"Fingerprint validation failed: {e}")

            decisions = engine.decide(fingerprint, user_intent, constraints)
            if validator:
                try:
                    data = decisions.to_dict()
                    validator.validate_decisions_collection(data)
                except Exception as e:
                    logger.warning(f"Decisions validation failed: {e}")

            if output_dir is not None:
                generator = ArtifactGenerator(Path(output_dir))
                artifacts = generator.generate_all(fingerprint, decisions)
            else:
                artifacts = {}

            return {
                "fingerprint": fingerprint,
                "decisions": decisions,
                "artifacts": artifacts,
                "version": "4.2.0",
            }
        else:
            # Run v5 pipeline (just extract)
            return self.extract(dataset_path)

    def _collect_images(self, dataset_path: Path) -> List[Dict[str, Any]]:
        images = []
        failed_count = 0

        handler = None
        if self.use_gpu or self.batch_size > 1:
            handler = BatchedGPUHandler(
                batch_size=self.batch_size, use_gpu=self.use_gpu
            )

        pending_batch = []

        def process_batch_results(batch_results, batch_metadata):
            for res, meta in zip(batch_results, batch_metadata):
                meta["entropy"] = res["entropy"]
                meta["blur_score"] = res["blur"]
                if "_img_array" in meta:
                    del meta["_img_array"]

        if dataset_path.is_file():
            img_data = self._analyze_single_image(dataset_path, handler)
            if img_data:
                images.append(img_data)
                if handler:
                    results = handler.add_image(img_data["_img_array"])
                    if results:
                        process_batch_results(results, [img_data])
                    else:
                        results = handler.flush()
                        process_batch_results(results, [img_data])
            else:
                failed_count += 1
        else:
            image_files = self._find_images(dataset_path)

            count = 0
            for img_path in image_files:
                if self.sample_limit and count >= self.sample_limit:
                    break

                img_data = self._analyze_single_image(img_path, handler)
                if img_data:
                    images.append(img_data)
                    count += 1

                    if handler:
                        pending_batch.append(img_data)
                        results = handler.add_image(img_data["_img_array"])
                        if results:
                            process_batch_results(results, pending_batch)
                            pending_batch = []
                else:
                    failed_count += 1

        if handler and pending_batch:
            results = handler.flush()
            process_batch_results(results, pending_batch)

        self.last_failed_count = failed_count
        return images

    def _find_images(self, directory: Path) -> List[Path]:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        images = []
        seen = set()

        for ext in sorted(extensions):
            for pattern in [f"*{ext}", f"*{ext.upper()}"]:
                for img_path in sorted(directory.rglob(pattern)):
                    resolved = img_path.resolve()
                    if resolved not in seen:
                        images.append(img_path)
                        seen.add(resolved)
        images.sort()
        return images

    def _analyze_single_image(
        self, img_path: Path, handler: Optional[BatchedGPUHandler] = None
    ) -> Optional[Dict[str, Any]]:
        try:
            with Image.open(img_path) as img:
                original_mode = img.mode
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Compute statistics
                stat = ImageStat.Stat(img)
                variance = stat.var
                hist = img.histogram()

                is_grayscale = original_mode in ("L", "LA", "I", "F")
                if not is_grayscale and img.mode == "RGB":
                    r, g, b = img.split()
                    if (
                        ImageStat.Stat(r).mean
                        == ImageStat.Stat(g).mean
                        == ImageStat.Stat(b).mean
                    ):
                        if np.array_equal(np.array(r), np.array(g)) and np.array_equal(
                            np.array(g), np.array(b)
                        ):
                            is_grayscale = True

                img_array = np.array(img)

                import hashlib

                img_hash = hashlib.sha256(img_array.tobytes()).hexdigest()

                if handler:
                    entropy = 0.0
                    blur_score = 0.0
                else:
                    entropy = _compute_entropy(img_array)
                    blur_score = _compute_blur(img_array)

                result = {
                    "path": img_path,
                    "size": (img.width, img.height),
                    "mode": img.mode,
                    "is_grayscale": is_grayscale,
                    "entropy": entropy,
                    "mean": stat.mean,
                    "variance": variance,
                    "extrema": stat.extrema,
                    "histogram": hist,
                    "blur_score": blur_score,
                    "hash": img_hash,
                }

                if handler:
                    result["_img_array"] = img_array

                return result
        except Exception as e:
            logger.warning(f"Failed to analyze {img_path}: {e}")
            return None

    def _derive_class(
        self,
        spatial: SpatialProfile,
        signal: SignalProfile,
        distribution: DistributionProfile,
        quality: QualityProfile,
        semantic: SemanticProfile,
    ) -> Tuple[str, float]:
        domain = semantic.primary_domain.value
        content = semantic.content_type.value
        derived_class = f"vision.{domain}.{content}"

        confidence = 0.9  # Base confidence
        if quality.overall_quality in (QualityLevel.POOR, QualityLevel.CRITICAL):
            confidence -= 0.2
        if semantic.primary_domain == PrimaryDomain.MIXED:
            confidence -= 0.1

        confidence = max(0.0, min(1.0, confidence))
        return derived_class, confidence

    def _generate_uri(self, derived_class: str) -> str:
        return f"imgshape://{derived_class.replace('.', '/')}"
