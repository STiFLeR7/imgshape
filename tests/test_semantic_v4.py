import torch
import numpy as np
import pytest
from imgshape.semantic_v4 import SemanticExtractor
from imgshape.compare_v4 import DatasetComparator, DriftScore

def test_semantic_embedding():
    extractor = SemanticExtractor(model_name="squeezenet1_1")
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    embedding = extractor.extract(img)
    
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (512,)

def test_semantic_drift_score():
    comparator = DatasetComparator()
    # Baseline: random features
    v1 = np.array([1.0, 0.0, 0.0])
    # Current: shifted features
    v2 = np.array([0.0, 1.0, 0.0])
    
    # This should now pass
    score = comparator.calculate_semantic_drift(v1, v2)
    assert score > 0.5 # High drift for orthogonal vectors

def test_dataset_drift_integration():
    from imgshape.fingerprint_v4 import DatasetFingerprint, SemanticProfile, SignalProfile, SpatialProfile, EntropyStats, Resolution
    from imgshape.fingerprint_v4 import PrimaryDomain, ContentType, ResizeRisk, GeometryClass
    
    comparator = DatasetComparator()
    
    def create_mock_fingerprint(embedding):
        semantic = SemanticProfile(
            primary_domain=PrimaryDomain.PHOTOGRAPHIC,
            content_type=ContentType.NATURAL,
            characteristics=["color"],
            specialization_required=False,
            latent_embedding=embedding.tolist()
        )
        signal = SignalProfile(
            entropy=EntropyStats(mean=7.0, std=0.5, range={"min": 6.0, "max": 8.0}),
            noise_estimate=0.1,
            capacity_ceiling=None,
            information_density=None
        )
        spatial = SpatialProfile(
            resolution_range={"median": Resolution(224, 224)},
            aspect_ratio_variance=0.01,
            resize_risk=ResizeRisk.LOW,
            geometry_class=GeometryClass.UNIFORM
        )
        return DatasetFingerprint(
            schema_version="4.1",
            dataset_uri="test://uri",
            profiles={"semantic": semantic, "signal": signal, "spatial": spatial},
            derived_class="vision.test",
            confidence=0.9
        )
        
    f1 = create_mock_fingerprint(np.array([1.0, 0.0, 0.0]))
    f2 = create_mock_fingerprint(np.array([0.0, 1.0, 0.0]))
    
    drift = comparator.calculate_drift_score(f1, f2)
    assert drift.semantic_drift > 0.5
    assert drift.overall_drift > 0.25 # Overall should be high due to 50% weight
    assert any("semantic drift" in r.lower() for r in drift.rationale)
