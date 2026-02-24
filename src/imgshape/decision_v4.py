"""
imgshape.v4.decision — Decision Engine and Decision Objects

The decision engine converts fingerprints into actionable decisions using deterministic rules.
Every decision includes full rationale, risks, tradeoffs, and alternatives.

Principles:
- Rules exist because explanations must be exact
- Decisions must be repeatable
- Audits must be trivial
- Regressions must be obvious
- No hidden inputs, no implicit state
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
import hashlib
import json

from imgshape.fingerprint_v4 import (
    DatasetFingerprint, 
    PrimaryDomain, 
    ContentType,
    CapacityCeiling,
    ResizeRisk,
    SamplingStrategy,
    LossRecommendation
)

logger = logging.getLogger("imgshape.decision_v4")


# ============================================================================
# Decision-Related Enumerations
# ============================================================================

class RiskSeverity(Enum):
    """Risk severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskType(Enum):
    """ML task types"""
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    GENERATION = "generation"
    OTHER = "other"


class DeploymentTarget(Enum):
    """Deployment environments"""
    CLOUD = "cloud"
    EDGE = "edge"
    MOBILE = "mobile"
    EMBEDDED = "embedded"
    OTHER = "other"


class Priority(Enum):
    """Optimization priorities"""
    ACCURACY = "accuracy"
    SPEED = "speed"
    SIZE = "size"
    BALANCED = "balanced"


# ============================================================================
# Decision Data Classes
# ============================================================================

@dataclass
class Risk:
    """A risk associated with a decision"""
    severity: RiskSeverity
    description: str
    mitigation: Optional[str] = None


@dataclass
class Tradeoff:
    """A tradeoff in a decision"""
    aspect: str
    gain: str
    cost: str


@dataclass
class Alternative:
    """An alternative that was considered but not selected"""
    option: Union[str, int, float, Dict[str, Any]]
    rejected_because: str
    score: Optional[float] = None  # v4.1: Higher is better


@dataclass
class Decision:
    """
    A single decision object with full rationale.
    
    This is the core output of the decision engine.
    Every decision is explainable, traceable, and reproducible.
    """
    decision_id: str
    selected: Union[str, int, float, Dict[str, Any], List[Any]]
    why: List[str]
    risks: List[Risk]
    tradeoffs: List[Tradeoff]
    alternatives_considered: List[Alternative]
    provenance_id: Optional[str] = None  # v4.1: Rule ID
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        # Convert enums
        if 'risks' in result:
            for risk in result['risks']:
                if isinstance(risk.get('severity'), Enum):
                    risk['severity'] = risk['severity'].value
        return result


@dataclass
class UserIntent:
    """User's stated intent for the ML task"""
    task: TaskType
    deployment_target: DeploymentTarget
    priority: Priority


@dataclass
class UserConstraints:
    """User-provided constraints"""
    max_model_size_mb: Optional[float] = None
    max_inference_time_ms: Optional[float] = None
    available_memory_mb: Optional[float] = None


@dataclass
class DecisionsCollection:
    """
    Complete collection of all decisions for a dataset.
    
    This is the authoritative output that drives all downstream artifacts.
    """
    schema_version: str
    fingerprint_id: str
    decisions: Dict[str, Decision]
    reproducibility_hash: str  # v4.1: SHA-256 of fingerprint + decisions
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "schema_version": self.schema_version,
            "fingerprint_id": self.fingerprint_id,
            "decisions": {k: v.to_dict() for k, v in self.decisions.items()},
            "metadata": self.metadata
        }
        return result


# ============================================================================
# Decision Engine
# ============================================================================

class DecisionEngine:
    """
    Deterministic rule-based decision engine.
    
    This engine takes a fingerprint and user intent, then produces a complete
    set of decisions with rationale. All decisions are deterministic and repeatable.
    
    The engine NEVER uses AI/ML to make decisions - only explicit rules.
    """

    def __init__(self, rule_version: str = "4.0.0"):
        """
        Initialize decision engine.
        
        Args:
            rule_version: Version of the rule tables to use
        """
        self.rule_version = rule_version
        self.logger = logger

    def decide(
        self,
        fingerprint: DatasetFingerprint,
        intent: UserIntent,
        constraints: Optional[UserConstraints] = None
    ) -> DecisionsCollection:
        """
        Make all decisions for a dataset.
        
        Args:
            fingerprint: Dataset fingerprint
            intent: User's stated intent
            constraints: Optional user constraints
            
        Returns:
            Complete collection of decisions
        """
        if constraints is None:
            constraints = UserConstraints()

        self.logger.info(f"Making decisions for {fingerprint.dataset_uri}")

        # Make individual decisions
        model_family = self._decide_model_family(fingerprint, intent, constraints)
        input_resolution = self._decide_input_resolution(fingerprint, intent, constraints)
        augmentation = self._decide_augmentation_strategy(fingerprint, intent, constraints)
        regularization = self._decide_regularization(fingerprint, intent, constraints)
        quantization = self._decide_quantization(fingerprint, intent, constraints)
        preprocessing = self._decide_preprocessing(fingerprint, intent, constraints)
        loss_function = self._decide_loss_function(fingerprint, intent, constraints)
        sampling_strategy = self._decide_sampling_strategy(fingerprint, intent, constraints)

        # Build collection
        collection = DecisionsCollection(
            schema_version="4.1",  # v4.1
            fingerprint_id=fingerprint.dataset_uri,
            decisions={
                "model_family": model_family,
                "input_resolution": input_resolution,
                "augmentation_strategy": augmentation,
                "regularization": regularization,
                "quantization": quantization,
                "preprocessing": preprocessing,
                "loss_function": loss_function,
                "sampling_strategy": sampling_strategy
            },
            reproducibility_hash="", # Set below
            metadata={
                "rule_engine_version": self.rule_version,
                "user_intent": {
                    "task": intent.task.value,
                    "deployment_target": intent.deployment_target.value,
                    "priority": intent.priority.value
                },
                "constraints": {
                    k: v for k, v in asdict(constraints).items() if v is not None
                }
            }
        )
        
        # Calculate reproducibility hash (v4.1)
        collection.reproducibility_hash = self._calculate_reproducibility_hash(fingerprint, collection)
        
        return collection

    def _calculate_reproducibility_hash(self, fingerprint: DatasetFingerprint, collection: DecisionsCollection) -> str:
        """Calculate SHA-256 hash of fingerprint and decisions for reproducibility tracking"""
        # We use a stable JSON representation for hashing
        data = {
            "fingerprint": fingerprint.dataset_uri,
            "rule_version": self.rule_version,
            "decisions": {k: str(d.selected) for k, d in collection.decisions.items()}
        }
        encoded = json.dumps(data, sort_keys=True).encode('utf-8')
        return hashlib.sha256(encoded).hexdigest()

    def _decide_model_family(
        self,
        fingerprint: DatasetFingerprint,
        intent: UserIntent,
        constraints: UserConstraints
    ) -> Decision:
        """Decide on model family"""
        
        semantic = fingerprint.profiles['semantic']
        signal = fingerprint.profiles['signal']
        
        why = []
        risks = []
        tradeoffs = []
        alternatives = []
        provenance_id = "BASE"

        # Primary rule: capacity ceiling gates model choice
        capacity = signal.capacity_ceiling
        
        if constraints.max_model_size_mb and constraints.max_model_size_mb < 20:
            # Severe size constraint
            selected = "MobileNetV3-Small"
            why.append("Strict model size constraint (< 20MB) requires lightweight architecture")
            why.append(f"Dataset capacity ceiling is {capacity.value}, which permits lightweight models")
            
            risks.append(Risk(
                severity=RiskSeverity.MEDIUM,
                description="Lightweight model may underfit high-complexity datasets",
                mitigation="Monitor validation metrics and consider knowledge distillation"
            ))
            
            tradeoffs.append(Tradeoff(
                aspect="model_size",
                gain="Fits deployment constraint",
                cost="Reduced representational capacity"
            ))
            
            provenance_id = "R-MODEL-SIZE-LT-20"
            alternatives.append(Alternative(
                option="MobileNetV3-Large",
                rejected_because="Exceeds model size constraint",
                score=0.4
            ))
            
        elif capacity == CapacityCeiling.LIGHTWEIGHT:
            selected = "MobileNetV3"
            why.append("Signal profile indicates lightweight model is sufficient")
            why.append(f"Information density is {signal.information_density.value}")
            
            risks.append(Risk(
                severity=RiskSeverity.LOW,
                description="May need capacity increase if dataset grows in complexity"
            ))
            
            provenance_id = "R-MODEL-CAP-LW"
            alternatives.append(Alternative(
                option="EfficientNetB0",
                rejected_because="More capacity than needed based on signal analysis",
                score=0.6
            ))
            
        elif capacity == CapacityCeiling.STANDARD:
            if intent.deployment_target in (DeploymentTarget.EDGE, DeploymentTarget.MOBILE):
                selected = "EfficientNetB1"
                why.append("Standard capacity needed, edge deployment requires efficiency")
            else:
                selected = "ResNet50"
                why.append("Standard capacity, cloud deployment permits larger models")
            
            provenance_id = f"R-MODEL-CAP-STD-{'EDGE' if intent.deployment_target in (DeploymentTarget.EDGE, DeploymentTarget.MOBILE) else 'CLOUD'}"
            alternatives.append(Alternative(
                option="VGG16",
                rejected_because="Inefficient architecture for modern deployments",
                score=0.2
            ))
            
        elif capacity == CapacityCeiling.HEAVY:
            selected = "EfficientNetB4"
            why.append("High signal density requires heavy model capacity")
            
            provenance_id = "R-MODEL-CAP-HEAVY"
            risks.append(Risk(
                severity=RiskSeverity.MEDIUM,
                description="Longer training time and higher compute requirements"
            ))
            
        else:  # UNLIMITED
            selected = "EfficientNetB7"
            why.append("Very high complexity dataset requires maximum capacity")
            
            provenance_id = "R-MODEL-CAP-UNLIMITED"
            risks.append(Risk(
                severity=RiskSeverity.HIGH,
                description="Very high computational cost",
                mitigation="Consider progressive training or model pruning"
            ))

        # Semantic domain can override
        if semantic.primary_domain == PrimaryDomain.MEDICAL:
            if semantic.specialization_required:
                selected = "MedicalNet (specialized)"
                why.append("Medical domain requires specialized architecture")

        return Decision(
            decision_id="model_family",
            selected=selected,
            why=why,
            risks=risks,
            tradeoffs=tradeoffs,
            alternatives_considered=alternatives,
            provenance_id=f"v4.1.0-rules-model-{provenance_id}",
            metadata={
                "rule_version": self.rule_version,
                "fingerprint_inputs": ["signal.capacity_ceiling", "semantic.primary_domain"]
            }
        )

    def _decide_input_resolution(
        self,
        fingerprint: DatasetFingerprint,
        intent: UserIntent,
        constraints: UserConstraints
    ) -> Decision:
        """Decide on input resolution"""
        
        spatial = fingerprint.profiles['spatial']
        
        median_res = spatial.resolution_range['median']
        resize_risk = spatial.resize_risk
        
        why = []
        risks = []
        alternatives = []
        
        # Rule: minimize resizing when risky
        if resize_risk == ResizeRisk.HIGH:
            selected = {"width": median_res.width, "height": median_res.height}
            why.append("High resize risk detected - using median native resolution")
            why.append(f"Aspect ratio variance: {spatial.aspect_ratio_variance:.3f}")
            
            risks.append(Risk(
                severity=RiskSeverity.MEDIUM,
                description="Variable input sizes increase training complexity"
            ))
        else:
            # Use standard resolution based on task
            if intent.task == TaskType.SEGMENTATION:
                selected = {"width": 512, "height": 512}
                why.append("Segmentation task benefits from higher resolution")
            elif intent.deployment_target == DeploymentTarget.MOBILE:
                selected = {"width": 224, "height": 224}
                why.append("Mobile deployment requires smaller input size")
            else:
                selected = {"width": 384, "height": 384}
                why.append("Standard resolution for balanced performance")
            
            alternatives.append(Alternative(
                option={"width": median_res.width, "height": median_res.height},
                rejected_because="Standardized resolution preferred when resize risk is acceptable"
            ))
        
        return Decision(
            decision_id="input_resolution",
            selected=selected,
            why=why,
            risks=risks,
            tradeoffs=[],
            alternatives_considered=alternatives,
            metadata={
                "rule_version": self.rule_version,
                "fingerprint_inputs": ["spatial.resize_risk", "spatial.resolution_range"]
            }
        )

    def _decide_augmentation_strategy(
        self,
        fingerprint: DatasetFingerprint,
        intent: UserIntent,
        constraints: UserConstraints
    ) -> Decision:
        """Decide on augmentation strategy"""
        
        signal = fingerprint.profiles['signal']
        semantic = fingerprint.profiles['semantic']
        quality = fingerprint.profiles['quality']
        
        why = []
        risks = []
        
        # Rule: augmentation strength inversely proportional to signal quality
        if signal.noise_estimate > 0.5:
            selected = "conservative"
            why.append("High noise level requires conservative augmentation")
            why.append(f"Noise estimate: {signal.noise_estimate:.2f}")
        elif semantic.primary_domain == PrimaryDomain.MEDICAL:
            selected = "domain_specific_medical"
            why.append("Medical images require specialized augmentations")
            why.append("Avoiding unrealistic transformations that could mislead diagnosis")
        else:
            selected = "standard"
            why.append("Clean dataset permits standard augmentation suite")
        
        return Decision(
            decision_id="augmentation_strategy",
            selected=selected,
            why=why,
            risks=risks,
            tradeoffs=[],
            alternatives_considered=[],
            metadata={
                "rule_version": self.rule_version,
                "fingerprint_inputs": ["signal.noise_estimate", "semantic.primary_domain"]
            }
        )

    def _decide_regularization(
        self,
        fingerprint: DatasetFingerprint,
        intent: UserIntent,
        constraints: UserConstraints
    ) -> Decision:
        """Decide on regularization strategy"""
        
        signal = fingerprint.profiles['signal']
        
        # Rule: regularization strength based on capacity and data quality
        if signal.capacity_ceiling == CapacityCeiling.HEAVY:
            selected = {"dropout": 0.5, "weight_decay": 1e-4}
            why = ["Heavy models require strong regularization"]
        else:
            selected = {"dropout": 0.2, "weight_decay": 1e-5}
            why = ["Lightweight models need moderate regularization"]
        
        return Decision(
            decision_id="regularization",
            selected=selected,
            why=why,
            risks=[],
            tradeoffs=[],
            alternatives_considered=[],
            metadata={"rule_version": self.rule_version}
        )

    def _decide_quantization(
        self,
        fingerprint: DatasetFingerprint,
        intent: UserIntent,
        constraints: UserConstraints
    ) -> Decision:
        """Decide on quantization strategy"""
        
        if intent.deployment_target in (DeploymentTarget.MOBILE, DeploymentTarget.EMBEDDED):
            selected = "int8"
            why = ["Mobile/embedded deployment benefits from quantization"]
        else:
            selected = "none"
            why = ["Cloud deployment doesn't require quantization"]
        
        return Decision(
            decision_id="quantization",
            selected=selected,
            why=why,
            risks=[],
            tradeoffs=[],
            alternatives_considered=[],
            metadata={"rule_version": self.rule_version}
        )

    def _decide_preprocessing(
        self,
        fingerprint: DatasetFingerprint,
        intent: UserIntent,
        constraints: UserConstraints
    ) -> Decision:
        """Decide on preprocessing pipeline"""
        
        semantic = fingerprint.profiles['semantic']
        
        steps = ["resize", "normalize"]
        why = ["Standard preprocessing: resize and normalize"]
        
        if semantic.primary_domain == PrimaryDomain.MEDICAL:
            steps.insert(1, "histogram_equalization")
            why.append("Medical images benefit from histogram equalization")
        
        return Decision(
            decision_id="preprocessing",
            selected=steps,
            why=why,
            risks=[],
            tradeoffs=[],
            alternatives_considered=[],
            metadata={"rule_version": self.rule_version}
        )

    def _decide_loss_function(
        self,
        fingerprint: DatasetFingerprint,
        intent: UserIntent,
        constraints: UserConstraints
    ) -> Decision:
        """Decide on loss function"""
        
        distribution = fingerprint.profiles['distribution']
        
        selected = distribution.loss_recommendation.value
        why = [f"Distribution profile recommends {selected}"]
        
        if distribution.class_balance.type.value == "highly_imbalanced":
            why.append(f"Class imbalance ratio: {distribution.class_balance.imbalance_ratio:.2f}")
        
        return Decision(
            decision_id="loss_function",
            selected=selected,
            why=why,
            risks=[],
            tradeoffs=[],
            alternatives_considered=[],
            metadata={"rule_version": self.rule_version}
        )

    def _decide_sampling_strategy(
        self,
        fingerprint: DatasetFingerprint,
        intent: UserIntent,
        constraints: UserConstraints
    ) -> Decision:
        """Decide on sampling strategy"""
        
        distribution = fingerprint.profiles['distribution']
        
        selected = distribution.sampling_strategy.value
        why = [f"Distribution analysis indicates {selected} sampling"]
        
        return Decision(
            decision_id="sampling_strategy",
            selected=selected,
            why=why,
            risks=[],
            tradeoffs=[],
            alternatives_considered=[],
            metadata={"rule_version": self.rule_version}
        )
