import logging
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image

logger = logging.getLogger("imgshape.atlas.semantic")

try:
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    models = None
    transforms = None
    TORCH_AVAILABLE = False


class PrimaryDomain(Enum):
    """Primary semantic domain"""

    PHOTOGRAPHIC = "photographic"
    MEDICAL = "medical"
    DOCUMENT = "document"
    SATELLITE = "satellite"
    OCR = "ocr"
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


@dataclass
class SemanticProfile:
    """Semantic classification that gates decision branches"""

    primary_domain: PrimaryDomain
    content_type: ContentType
    characteristics: List[str]
    specialization_required: bool
    latent_embedding: Optional[List[float]] = None


@dataclass
class MedicalProfile:
    """Specialized metrics for medical imaging"""

    hu_range: Tuple[float, float]
    slice_consistency: float


@dataclass
class SatelliteProfile:
    """Specialized metrics for remote sensing and satellite imagery"""

    gsd_estimate: float  # Ground Sample Distance
    cloud_cover_estimate: float


@dataclass
class OCRProfile:
    """Specialized metrics for document analysis and OCR"""

    text_density: float
    orientation_variance: float


class SemanticExtractor:
    """
    Lightweight feature extractor using SqueezeNet.
    Wraps torch/torchvision imports to fail gracefully if missing.
    """

    def __init__(self, model_name="squeezenet1_1"):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch and torchvision are required for SemanticExtractor. "
                "Install with 'pip install \"imgshape[torch]\"'."
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            # Try modern weights syntax
            weights = getattr(models, "SqueezeNet1_1_Weights").DEFAULT
            self.model = (
                getattr(models, model_name)(weights=weights).to(self.device).eval()
            )
        except AttributeError:
            # Fallback for older torchvision
            self.model = (
                getattr(models, model_name)(pretrained=True).to(self.device).eval()
            )

        if "squeezenet" in model_name:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.Flatten()
            )

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def extract(self, img_array: np.ndarray) -> np.ndarray:
        if not TORCH_AVAILABLE:
            raise ImportError("SemanticExtractor requires torch and torchvision")

        img = Image.fromarray(img_array)
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(img_t)
        return features.squeeze().cpu().numpy()


class SemanticProfiler:
    """
    Implements semantic profiling using heuristics and optional latent embeddings.
    """

    @staticmethod
    def profile(
        images: List[Dict[str, Any]], spatial_profile, signal_profile
    ) -> SemanticProfile:
        if not images:
            raise ValueError("No images provided for semantic profiling")

        grayscale_ratio = sum(
            1 for img in images if img.get("is_grayscale", False)
        ) / len(images)
        mean_entropy = signal_profile.entropy.mean
        ar_variance = spatial_profile.aspect_ratio_variance

        characteristics = []

        if grayscale_ratio > 0.5:
            characteristics.append("grayscale")
            if mean_entropy < 7:
                primary_domain = PrimaryDomain.MEDICAL
                content_type = ContentType.LOW_SIGNAL
                specialization_required = True
            else:
                primary_domain = PrimaryDomain.OCR
                content_type = ContentType.TEXT_DENSE
                specialization_required = True
        else:
            characteristics.append("color")
            if ar_variance < 0.001 and mean_entropy > 7.0:
                primary_domain = PrimaryDomain.SATELLITE
                content_type = ContentType.HIGH_ENTROPY
                specialization_required = True
            elif mean_entropy > 8:
                primary_domain = PrimaryDomain.PHOTOGRAPHIC
                content_type = ContentType.HIGH_ENTROPY
                specialization_required = False
            else:
                primary_domain = PrimaryDomain.SYNTHETIC
                content_type = ContentType.STRUCTURED
                specialization_required = False

        latent_embedding = None
        if TORCH_AVAILABLE:
            try:
                extractor = SemanticExtractor()
                # Sample up to 10 images for mean embedding
                sample_indices = np.linspace(
                    0, len(images) - 1, min(10, len(images)), dtype=int
                )
                embeddings = []
                for idx in sample_indices:
                    img_data = images[idx]
                    path = img_data["path"]
                    with Image.open(path) as img:
                        img_arr = np.array(img.convert("RGB"))
                        emb = extractor.extract(img_arr)
                        embeddings.append(emb)
                if embeddings:
                    mean_emb = np.mean(embeddings, axis=0)
                    latent_embedding = mean_emb.tolist()
            except Exception as e:
                logger.warning(f"Failed to compute latent embedding: {e}")
        else:
            logger.info(
                "PyTorch and torchvision are not available. Skipping latent embedding extraction."
            )

        return SemanticProfile(
            primary_domain=primary_domain,
            content_type=content_type,
            characteristics=characteristics,
            specialization_required=specialization_required,
            latent_embedding=latent_embedding,
        )

    @staticmethod
    def profile_medical(images: List[Dict[str, Any]]) -> MedicalProfile:
        all_min = []
        all_max = []
        for img in images:
            ext = img["extrema"]
            all_min.append(np.min([e[0] for e in ext]))
            all_max.append(np.max([e[1] for e in ext]))

        hu_range = (float(np.min(all_min)), float(np.max(all_max)))

        sizes = [img["size"] for img in images]
        size_vars = np.var([s[0] for s in sizes]) + np.var([s[1] for s in sizes])
        slice_consistency = 1.0 / (1.0 + float(size_vars))

        return MedicalProfile(hu_range=hu_range, slice_consistency=slice_consistency)

    @staticmethod
    def profile_satellite(images: List[Dict[str, Any]]) -> SatelliteProfile:
        sizes = [img["size"] for img in images]
        avg_dim = np.mean([s[0] + s[1] for s in sizes]) / 2.0
        gsd_estimate = 100.0 / max(1.0, float(avg_dim))

        cloud_ratios = []
        for img in images:
            hist = img["histogram"]
            total_count = sum(hist)
            high_count = 0
            for c in range(3):
                high_count += sum(hist[c * 256 + 240 : (c + 1) * 256])
            cloud_ratios.append(high_count / max(1, total_count))

        cloud_cover = float(np.mean(cloud_ratios))

        return SatelliteProfile(
            gsd_estimate=gsd_estimate, cloud_cover_estimate=cloud_cover
        )

    @staticmethod
    def profile_ocr(images: List[Dict[str, Any]]) -> OCRProfile:
        entropies = [img["entropy"] for img in images]
        text_density = float(np.mean(entropies)) / 10.0

        ars = [img["size"][0] / max(1, img["size"][1]) for img in images]
        orientation_variance = float(np.var(ars))

        return OCRProfile(
            text_density=text_density, orientation_variance=orientation_variance
        )
