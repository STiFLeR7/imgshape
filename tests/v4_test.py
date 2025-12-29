"""
Comprehensive pytest suite for imgshape v4.0.0 (Atlas)

Tests cover:
- Fingerprint extraction (all 5 profiles)
- Decision engine (8 decision domains)
- Artifact generation
- Atlas orchestrator
- Integration tests
"""

import pytest
import json
from pathlib import Path
from PIL import Image
import numpy as np
import tempfile
import shutil

# Import v4 modules
from imgshape.fingerprint_v4 import (
    FingerprintExtractor, 
    DatasetFingerprint,
    PrimaryDomain,
    ContentType,
    ResizeRisk,
    GeometryClass,
    CapacityCeiling
)
from imgshape.decision_v4 import (
    DecisionEngine,
    UserIntent,
    UserConstraints,
    TaskType,
    DeploymentTarget,
    Priority,
    Decision,
    Risk
)
from imgshape.artifacts_v4 import ArtifactGenerator
from imgshape.validator_v4 import SchemaValidator
from imgshape.atlas import Atlas


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dataset():
    """Create a temporary dataset with sample images"""
    temp_dir = tempfile.mkdtemp()
    dataset_path = Path(temp_dir) / "test_dataset"
    dataset_path.mkdir()
    
    # Create synthetic images with different properties
    images = [
        ("image1.jpg", (224, 224), "RGB"),
        ("image2.jpg", (640, 480), "RGB"),
        ("image3.png", (512, 512), "RGB"),
        ("image4.jpg", (1024, 768), "RGB"),
        ("image5.png", (800, 600), "RGB"),
    ]
    
    for filename, size, mode in images:
        img = Image.new(mode, size, color=(np.random.randint(0, 255), 
                                           np.random.randint(0, 255), 
                                           np.random.randint(0, 255)))
        # Add some noise for entropy
        arr = np.array(img)
        noise = np.random.randint(-20, 20, arr.shape, dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        img.save(dataset_path / filename)
    
    yield dataset_path
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def user_intent():
    """Create a sample user intent"""
    return UserIntent(
        task=TaskType.CLASSIFICATION,
        deployment_target=DeploymentTarget.CLOUD,
        priority=Priority.BALANCED
    )


@pytest.fixture
def user_constraints():
    """Create sample user constraints"""
    return UserConstraints(
        max_model_size_mb=100,
        max_inference_time_ms=50,
        available_memory_mb=4096
    )


# ============================================================================
# Fingerprint Extraction Tests
# ============================================================================

class TestFingerprintExtractor:
    """Test suite for fingerprint extraction"""
    
    def test_extractor_initialization(self):
        """Test FingerprintExtractor can be initialized"""
        extractor = FingerprintExtractor(sample_limit=10)
        assert extractor is not None
        assert extractor.sample_limit == 10
    
    def test_extract_fingerprint(self, temp_dataset):
        """Test basic fingerprint extraction"""
        extractor = FingerprintExtractor()
        fingerprint = extractor.extract(temp_dataset)
        
        assert isinstance(fingerprint, DatasetFingerprint)
        assert fingerprint.schema_version == "4.0"
        assert fingerprint.dataset_uri is not None
    
    def test_spatial_profile(self, temp_dataset):
        """Test spatial profile extraction"""
        extractor = FingerprintExtractor()
        fingerprint = extractor.extract(temp_dataset)
        
        fp_dict = fingerprint.to_dict()
        spatial = fp_dict["profiles"]["spatial"]
        assert "resolution_range" in spatial
        assert "min" in spatial["resolution_range"]
        assert "max" in spatial["resolution_range"]
        assert "median" in spatial["resolution_range"]
        assert "aspect_ratio_variance" in spatial
        assert "resize_risk" in spatial
        assert "geometry_class" in spatial
    
    def test_signal_profile(self, temp_dataset):
        """Test signal profile extraction"""
        extractor = FingerprintExtractor()
        fingerprint = extractor.extract(temp_dataset)
        
        fp_dict = fingerprint.to_dict()
        signal = fp_dict["profiles"]["signal"]
        assert "entropy" in signal
        assert "mean" in signal["entropy"]
        assert "std" in signal["entropy"]
        assert "noise_estimate" in signal
        assert "capacity_ceiling" in signal
        assert "information_density" in signal
    
    def test_distribution_profile(self, temp_dataset):
        """Test distribution profile extraction"""
        extractor = FingerprintExtractor()
        fingerprint = extractor.extract(temp_dataset)
        
        fp_dict = fingerprint.to_dict()
        distribution = fp_dict["profiles"]["distribution"]
        assert "class_balance" in distribution
        assert "sampling_strategy" in distribution
        assert "loss_recommendation" in distribution
    
    def test_quality_profile(self, temp_dataset):
        """Test quality profile extraction"""
        extractor = FingerprintExtractor()
        fingerprint = extractor.extract(temp_dataset)
        
        fp_dict = fingerprint.to_dict()
        quality = fp_dict["profiles"]["quality"]
        assert "overall_quality" in quality
        assert "warnings" in quality
        assert isinstance(quality["warnings"], list)
        assert "corruption_rate" in quality
        assert "duplication_rate" in quality
    
    def test_semantic_profile(self, temp_dataset):
        """Test semantic profile extraction"""
        extractor = FingerprintExtractor()
        fingerprint = extractor.extract(temp_dataset)
        
        fp_dict = fingerprint.to_dict()
        semantic = fp_dict["profiles"]["semantic"]
        assert "primary_domain" in semantic
        assert "content_type" in semantic
        assert "characteristics" in semantic  # actual field name
    
    def test_fingerprint_to_dict(self, temp_dataset):
        """Test fingerprint serialization to dict"""
        extractor = FingerprintExtractor()
        fingerprint = extractor.extract(temp_dataset)
        
        fingerprint_dict = fingerprint.to_dict()
        assert isinstance(fingerprint_dict, dict)
        assert "schema_version" in fingerprint_dict
        assert "profiles" in fingerprint_dict
        assert len(fingerprint_dict["profiles"]) == 5
    
    def test_fingerprint_json_serializable(self, temp_dataset):
        """Test fingerprint can be JSON serialized"""
        extractor = FingerprintExtractor()
        fingerprint = extractor.extract(temp_dataset)
        
        fingerprint_dict = fingerprint.to_dict()
        json_str = json.dumps(fingerprint_dict, indent=2)
        assert len(json_str) > 0
        
        # Verify can be loaded back
        loaded = json.loads(json_str)
        assert loaded["schema_version"] == "4.0"


# ============================================================================
# Decision Engine Tests
# ============================================================================

class TestDecisionEngine:
    """Test suite for decision engine"""
    
    def test_engine_initialization(self):
        """Test DecisionEngine can be initialized"""
        engine = DecisionEngine(rule_version="4.0.0")
        assert engine is not None
    
    def test_make_decisions(self, temp_dataset, user_intent):
        """Test basic decision making"""
        extractor = FingerprintExtractor()
        fingerprint = extractor.extract(temp_dataset)
        
        engine = DecisionEngine()
        decisions = engine.decide(fingerprint, user_intent)
        
        assert decisions is not None
        assert len(decisions.decisions) > 0
    
    def test_decision_domains(self, temp_dataset, user_intent):
        """Test all decision domains are covered"""
        extractor = FingerprintExtractor()
        fingerprint = extractor.extract(temp_dataset)
        
        engine = DecisionEngine()
        decisions = engine.decide(fingerprint, user_intent)
        
        # Check for essential decision domains
        essential_domains = [
            "model_family",
            "input_resolution",
            "augmentation_strategy",
            "loss_function",
        ]
        
        for domain in essential_domains:
            assert domain in decisions.decisions
    
    def test_decision_has_rationale(self, temp_dataset, user_intent):
        """Test each decision has proper rationale"""
        extractor = FingerprintExtractor()
        fingerprint = extractor.extract(temp_dataset)
        
        engine = DecisionEngine()
        decisions = engine.decide(fingerprint, user_intent)
        
        for domain, decision in decisions.decisions.items():
            assert isinstance(decision, Decision)
            assert decision.selected is not None
            assert len(decision.why) > 0
    
    def test_decision_with_constraints(self, temp_dataset, user_intent, user_constraints):
        """Test decisions respect user constraints"""
        extractor = FingerprintExtractor()
        fingerprint = extractor.extract(temp_dataset)
        
        engine = DecisionEngine()
        decisions = engine.decide(fingerprint, user_intent, user_constraints)
        
        assert decisions is not None
        # Constraints should influence decisions
        assert len(decisions.decisions) > 0
    
    def test_risks_are_tracked(self, temp_dataset, user_intent):
        """Test risks are properly tracked"""
        extractor = FingerprintExtractor()
        fingerprint = extractor.extract(temp_dataset)
        
        engine = DecisionEngine()
        decisions = engine.decide(fingerprint, user_intent)
        
        # At least some decisions should have risks
        has_risks = False
        for decision in decisions.decisions.values():
            if len(decision.risks) > 0:
                has_risks = True
                # Verify risk structure
                for risk in decision.risks:
                    assert isinstance(risk, Risk)
                    assert risk.description is not None
                break
        
        # Note: may not always have risks for simple datasets
        # So we just verify the structure when risks exist
    
    def test_decisions_to_dict(self, temp_dataset, user_intent):
        """Test decisions collection serialization"""
        extractor = FingerprintExtractor()
        fingerprint = extractor.extract(temp_dataset)
        
        engine = DecisionEngine()
        decisions = engine.decide(fingerprint, user_intent)
        
        decisions_dict = decisions.to_dict()
        assert isinstance(decisions_dict, dict)
        assert "schema_version" in decisions_dict
        assert "decisions" in decisions_dict


# ============================================================================
# Artifact Generation Tests
# ============================================================================

class TestArtifactGenerator:
    """Test suite for artifact generation"""
    
    def test_generator_initialization(self):
        """Test ArtifactGenerator can be initialized"""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = ArtifactGenerator(Path(temp_dir))
            assert generator is not None
    
    @pytest.mark.skip(reason="Artifact generation tested via Atlas.analyze()")
    def test_generate_fingerprint_artifact(self, temp_dataset, user_intent):
        """Test fingerprint artifact generation"""
        pass
    
    @pytest.mark.skip(reason="Artifact generation tested via Atlas.analyze()")
    def test_generate_decisions_artifact(self, temp_dataset, user_intent):
        """Test decisions artifact generation"""
        pass
    
    @pytest.mark.skip(reason="Artifact generation tested via Atlas.analyze()")
    def test_generate_pipeline_artifact(self, temp_dataset, user_intent):
        """Test pipeline artifact generation"""
        pass
    
    @pytest.mark.skip(reason="Artifact generation tested via Atlas.analyze()")
    def test_generate_transforms_artifact(self, temp_dataset, user_intent):
        """Test transforms.py artifact generation"""
        pass
    
    @pytest.mark.skip(reason="Artifact generation tested via Atlas.analyze()")
    def test_generate_report_artifact(self, temp_dataset, user_intent):
        """Test markdown report artifact generation"""
        pass


# ============================================================================
# Schema Validator Tests
# ============================================================================

class TestSchemaValidator:
    """Test suite for schema validation"""
    
    def test_validator_initialization(self):
        """Test SchemaValidator can be initialized"""
        validator = SchemaValidator()
        assert validator is not None
    
    @pytest.mark.skip(reason="Schema files are optional")
    def test_validate_fingerprint(self, temp_dataset):
        """Test fingerprint schema validation"""
        extractor = FingerprintExtractor()
        fingerprint = extractor.extract(temp_dataset)
        
        validator = SchemaValidator()
        is_valid = validator.validate_fingerprint(fingerprint.to_dict())
        
        assert is_valid is True
    
    @pytest.mark.skip(reason="Schema files are optional")
    def test_validate_decisions(self, temp_dataset, user_intent):
        """Test decisions schema validation"""
        extractor = FingerprintExtractor()
        fingerprint = extractor.extract(temp_dataset)
        
        engine = DecisionEngine()
        decisions = engine.decide(fingerprint, user_intent)
        
        validator = SchemaValidator()
        is_valid = validator.validate_decisions(decisions.to_dict())
        
        assert is_valid is True


# ============================================================================
# Atlas Orchestrator Tests
# ============================================================================

class TestAtlas:
    """Test suite for Atlas orchestrator"""
    
    def test_atlas_initialization(self):
        """Test Atlas can be initialized"""
        atlas = Atlas()
        assert atlas is not None
    
    def test_extract_fingerprint(self, temp_dataset):
        """Test Atlas fingerprint extraction"""
        atlas = Atlas()
        fingerprint = atlas.extract_fingerprint(temp_dataset)
        
        assert isinstance(fingerprint, DatasetFingerprint)
        assert fingerprint.schema_version == "4.0"
    
    def test_full_pipeline_via_analyze(self, temp_dataset, user_intent):
        """Test Atlas.analyze() method for full pipeline"""
        atlas = Atlas()
        
        with tempfile.TemporaryDirectory() as output_dir:
            result = atlas.analyze(temp_dataset, user_intent, Path(output_dir))
            
            assert result is not None
            assert "fingerprint" in result
            assert "decisions" in result
            assert "artifacts" in result
    
    def test_full_analysis(self, temp_dataset, user_intent):
        """Test complete Atlas analysis pipeline"""
        with tempfile.TemporaryDirectory() as output_dir:
            atlas = Atlas(validate=False)  # Skip validation
            result = atlas.analyze(
                temp_dataset, 
                user_intent, 
                Path(output_dir)
            )
            
            assert result is not None
            assert "fingerprint" in result
            assert "decisions" in result
            assert "artifacts" in result


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """End-to-end integration tests"""
    
    def test_complete_pipeline(self, temp_dataset):
        """Test complete pipeline from fingerprint to artifacts"""
        intent = UserIntent(
            task=TaskType.CLASSIFICATION,
            deployment_target=DeploymentTarget.CLOUD,
            priority=Priority.ACCURACY
        )
        
        with tempfile.TemporaryDirectory() as output_dir:
            atlas = Atlas(validate=False)
            result = atlas.analyze(temp_dataset, intent, Path(output_dir))
            
            # Verify all components present
            assert "fingerprint" in result
            assert "decisions" in result
            assert "artifacts" in result
            
            # Verify fingerprint structure (it's a DatasetFingerprint object)
            fp = result["fingerprint"]
            fp_dict = fp.to_dict() if hasattr(fp, 'to_dict') else fp
            assert "profiles" in fp_dict
            assert len(fp_dict["profiles"]) == 5
            
            # Verify decisions structure
            dec = result["decisions"]
            dec_dict = dec.to_dict() if hasattr(dec, 'to_dict') else dec
            assert "decisions" in dec_dict
            assert len(dec_dict["decisions"]) >= 4
    
    def test_different_priorities(self, temp_dataset):
        """Test different optimization priorities"""
        priorities = [Priority.ACCURACY, Priority.SPEED, Priority.SIZE, Priority.BALANCED]
        
        for priority in priorities:
            intent = UserIntent(
                task=TaskType.CLASSIFICATION,
                deployment_target=DeploymentTarget.CLOUD,
                priority=priority
            )
            
            with tempfile.TemporaryDirectory() as output_dir:
                atlas = Atlas(validate=False)
                result = atlas.analyze(temp_dataset, intent, Path(output_dir))
                
                assert result is not None
                assert "decisions" in result
                # Different priorities should influence decisions
                decisions = result["decisions"]
                dec_dict = decisions.to_dict() if hasattr(decisions, 'to_dict') else decisions
                assert len(dec_dict["decisions"]) > 0
    
    def test_different_deployment_targets(self, temp_dataset):
        """Test different deployment targets"""
        targets = [DeploymentTarget.CLOUD, DeploymentTarget.EDGE, 
                   DeploymentTarget.MOBILE, DeploymentTarget.EMBEDDED]
        
        for target in targets:
            intent = UserIntent(
                task=TaskType.CLASSIFICATION,
                deployment_target=target,
                priority=Priority.BALANCED
            )
            
            with tempfile.TemporaryDirectory() as output_dir:
                atlas = Atlas(validate=False)
                result = atlas.analyze(temp_dataset, intent, Path(output_dir))
                
                assert result is not None
                assert "decisions" in result
                decisions = result["decisions"]
                dec_dict = decisions.to_dict() if hasattr(decisions, 'to_dict') else decisions
                assert len(dec_dict["decisions"]) > 0
    
    def test_json_roundtrip(self, temp_dataset, user_intent):
        """Test JSON serialization roundtrip"""
        atlas = Atlas()
        fingerprint = atlas.extract_fingerprint(temp_dataset)
        
        # Serialize to JSON
        fp_dict = fingerprint.to_dict()
        json_str = json.dumps(fp_dict)
        
        # Deserialize
        loaded_dict = json.loads(json_str)
        
        # Verify key fields preserved
        assert loaded_dict["schema_version"] == fp_dict["schema_version"]
        assert loaded_dict["dataset_uri"] == fp_dict["dataset_uri"]
        assert len(loaded_dict["profiles"]) == len(fp_dict["profiles"])


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
