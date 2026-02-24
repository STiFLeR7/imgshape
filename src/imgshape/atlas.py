"""
imgshape.v4.atlas — Main Atlas Orchestrator

This is the entry point for imgshape v4.0.0 (Atlas).
It orchestrates the complete pipeline: fingerprint → decide → artifacts.

Usage:
    from imgshape.v4 import Atlas
    
    atlas = Atlas()
    atlas.analyze(dataset_path, user_intent, output_dir)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict, Any

from imgshape.fingerprint_v4 import FingerprintExtractor, DatasetFingerprint
from imgshape.decision_v4 import DecisionEngine, UserIntent, UserConstraints, TaskType, DeploymentTarget, Priority
from imgshape.artifacts_v4 import ArtifactGenerator
from imgshape.validator_v4 import SchemaValidator
from imgshape.compare_v4 import DatasetComparator

logger = logging.getLogger("imgshape.atlas")


class Atlas:
    """
    Main orchestrator for imgshape v4.0.0 (Atlas).
    
    This class coordinates the entire pipeline:
    1. Extract dataset fingerprint
    2. Make decisions based on fingerprint + user intent
    3. Generate deployable artifacts
    4. Validate all outputs
    
    Everything is deterministic, traceable, and reproducible.
    """

    def __init__(
        self,
        sample_limit: Optional[int] = None,
        rule_version: str = "4.0.0",
        validate: bool = True
    ):
        """
        Initialize Atlas.
        
        Args:
            sample_limit: Optional limit on images to analyze
            rule_version: Version of decision rules to use
            validate: Whether to validate outputs against schemas
        """
        self.extractor = FingerprintExtractor(sample_limit=sample_limit)
        self.engine = DecisionEngine(rule_version=rule_version)
        self.comparator = DatasetComparator()
        self.validator = SchemaValidator() if validate else None
        self.validate_outputs = validate
        self.logger = logger

    def analyze(
        self,
        dataset_path: Path,
        user_intent: UserIntent,
        output_dir: Path,
        constraints: Optional[UserConstraints] = None
    ) -> Dict[str, Any]:
        """
        Complete analysis pipeline.
        
        Args:
            dataset_path: Path to dataset
            user_intent: User's stated intent (task, target, priority)
            output_dir: Where to write artifacts
            constraints: Optional user constraints
            
        Returns:
            Dictionary with fingerprint, decisions, and artifact paths
        """
        self.logger.info(f"Starting Atlas analysis for {dataset_path}")
        
        # Phase 1: Extract fingerprint
        self.logger.info("Phase 1: Extracting dataset fingerprint...")
        fingerprint = self.extractor.extract(dataset_path)
        
        if self.validate_outputs:
            self._validate_fingerprint(fingerprint)
        
        # Phase 2: Make decisions
        self.logger.info("Phase 2: Making decisions...")
        decisions = self.engine.decide(fingerprint, user_intent, constraints)
        
        if self.validate_outputs:
            self._validate_decisions(decisions)
        
        # Phase 3: Generate artifacts
        self.logger.info("Phase 3: Generating artifacts...")
        generator = ArtifactGenerator(output_dir)
        artifacts = generator.generate_all(fingerprint, decisions)
        
        self.logger.info(f"✓ Atlas analysis complete. Artifacts in {output_dir}")
        
        return {
            "fingerprint": fingerprint,
            "decisions": decisions,
            "artifacts": artifacts,
            "version": "4.1.0"
        }

    def compare(self, baseline_path: Path, current_path: Path) -> Dict[str, Any]:
        """
        Compare two datasets (or fingerprints) and return drift/similarity.
        """
        f1 = self.extractor.extract(baseline_path)
        f2 = self.extractor.extract(current_path)
        return self.comparator.compare(f1, f2)

    def extract_fingerprint(self, dataset_path: Path) -> DatasetFingerprint:
        """
        Extract only the fingerprint (no decisions).
        
        Args:
            dataset_path: Path to dataset
            
        Returns:
            DatasetFingerprint
        """
        return self.extractor.extract(dataset_path)

    def _validate_fingerprint(self, fingerprint: DatasetFingerprint):
        """Validate fingerprint against schema"""
        if not self.validator:
            return
        
        try:
            data = fingerprint.to_dict()
            self.validator.validate_fingerprint(data)
            self.logger.debug("✓ Fingerprint validation passed")
        except Exception as e:
            self.logger.warning(f"Fingerprint validation failed: {e}")

    def _validate_decisions(self, decisions):
        """Validate decisions against schema"""
        if not self.validator:
            return
        
        try:
            data = decisions.to_dict()
            self.validator.validate_decisions_collection(data)
            self.logger.debug("✓ Decisions validation passed")
        except Exception as e:
            self.logger.warning(f"Decisions validation failed: {e}")


# Convenience functions for common use cases

def analyze_dataset(
    dataset_path: str | Path,
    task: str = "classification",
    deployment: str = "cloud",
    priority: str = "balanced",
    output_dir: Optional[str | Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for complete dataset analysis.
    
    Args:
        dataset_path: Path to dataset
        task: ML task (classification, detection, segmentation, etc.)
        deployment: Deployment target (cloud, edge, mobile, embedded)
        priority: Optimization priority (accuracy, speed, size, balanced)
        output_dir: Where to write artifacts (defaults to ./imgshape_v4_output)
        **kwargs: Additional options (sample_limit, max_model_size_mb, etc.)
        
    Returns:
        Dictionary with fingerprint, decisions, and artifacts
    """
    dataset_path = Path(dataset_path)
    
    if output_dir is None:
        output_dir = Path("./imgshape_v4_output")
    else:
        output_dir = Path(output_dir)
    
    # Parse user intent
    intent = UserIntent(
        task=TaskType[task.upper()],
        deployment_target=DeploymentTarget[deployment.upper()],
        priority=Priority[priority.upper()]
    )
    
    # Parse constraints
    constraints = UserConstraints(
        max_model_size_mb=kwargs.get('max_model_size_mb'),
        max_inference_time_ms=kwargs.get('max_inference_time_ms'),
        available_memory_mb=kwargs.get('available_memory_mb')
    )
    
    # Run analysis
    atlas = Atlas(sample_limit=kwargs.get('sample_limit'))
    return atlas.analyze(dataset_path, intent, output_dir, constraints)


def fingerprint_only(
    dataset_path: str | Path,
    sample_limit: Optional[int] = None
) -> DatasetFingerprint:
    """
    Extract only the fingerprint (no decisions or artifacts).
    
    Args:
        dataset_path: Path to dataset
        sample_limit: Optional limit on images to analyze
        
    Returns:
        DatasetFingerprint
    """
    atlas = Atlas(sample_limit=sample_limit, validate=False)
    return atlas.extract_fingerprint(Path(dataset_path))
