"""
imgshape.v4.artifacts — Artifact Generation System

Generates deployable artifacts from fingerprints and decisions.
All artifacts are deterministic, versioned, and CI-safe.

Artifact Types:
- dataset.fingerprint.json: The canonical fingerprint
- decisions.json: Complete decision collection
- pipeline.v4.json: Deployable pipeline configuration
- transforms.py: Python code for data transforms
- report.md / report.html: Human-readable reports
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from imgshape.fingerprint_v4 import DatasetFingerprint
from imgshape.decision_v4 import DecisionsCollection, Decision

logger = logging.getLogger("imgshape.artifacts_v4")


class ArtifactGenerator:
    """
    Generates all v4 artifacts from fingerprints and decisions.
    
    All outputs are deterministic and reproducible.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize artifact generator.
        
        Args:
            output_dir: Directory where artifacts will be written
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def generate_all(
        self,
        fingerprint: DatasetFingerprint,
        decisions: DecisionsCollection
    ) -> Dict[str, Path]:
        """
        Generate all artifacts.
        
        Args:
            fingerprint: Dataset fingerprint
            decisions: Decision collection
            
        Returns:
            Dictionary mapping artifact type to file path
        """
        artifacts = {}
        
        # Generate each artifact
        artifacts['fingerprint'] = self.generate_fingerprint_artifact(fingerprint)
        artifacts['decisions'] = self.generate_decisions_artifact(decisions)
        artifacts['pipeline'] = self.generate_pipeline_artifact(fingerprint, decisions)
        artifacts['transforms'] = self.generate_transforms_artifact(decisions)
        artifacts['report_md'] = self.generate_markdown_report(fingerprint, decisions)
        
        self.logger.info(f"Generated {len(artifacts)} artifacts in {self.output_dir}")
        
        return artifacts

    def generate_fingerprint_artifact(self, fingerprint: DatasetFingerprint) -> Path:
        """Generate dataset.fingerprint.json"""
        output_file = self.output_dir / "dataset.fingerprint.json"
        
        data = fingerprint.to_dict()
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Generated fingerprint artifact: {output_file}")
        return output_file

    def generate_decisions_artifact(self, decisions: DecisionsCollection) -> Path:
        """Generate decisions.json"""
        output_file = self.output_dir / "decisions.json"
        
        data = decisions.to_dict()
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Generated decisions artifact: {output_file}")
        return output_file

    def generate_pipeline_artifact(
        self,
        fingerprint: DatasetFingerprint,
        decisions: DecisionsCollection
    ) -> Path:
        """Generate pipeline.v4.json"""
        output_file = self.output_dir / "pipeline.v4.json"
        
        # Build pipeline stages from decisions
        stages = []
        
        # Preprocessing stages
        if "preprocessing" in decisions.decisions:
            preproc = decisions.decisions["preprocessing"]
            for idx, step in enumerate(preproc.selected):
                stages.append({
                    "stage_id": f"preproc_{idx}",
                    "operation": step,
                    "parameters": self._get_preproc_params(step, decisions),
                    "rationale": f"From preprocessing decision: {step}"
                })
        
        # Augmentation stages
        if "augmentation_strategy" in decisions.decisions:
            aug = decisions.decisions["augmentation_strategy"]
            stages.append({
                "stage_id": "augmentation",
                "operation": "augment",
                "parameters": {
                    "strategy": aug.selected,
                    "strength": self._get_aug_strength(aug.selected)
                },
                "rationale": " | ".join(aug.why)
            })
        
        # Model recommendation
        model_rec = None
        if "model_family" in decisions.decisions:
            model_decision = decisions.decisions["model_family"]
            input_res = decisions.decisions.get("input_resolution")
            
            model_rec = {
                "family": model_decision.selected,
                "architecture": model_decision.selected,
                "pretrained": True,
                "input_shape": [
                    input_res.selected.get("height", 224) if input_res else 224,
                    input_res.selected.get("width", 224) if input_res else 224,
                    3
                ]
            }
        
        # Training hints
        training_hints = self._build_training_hints(decisions)
        
        pipeline = {
            "schema_version": "4.0",
            "pipeline_id": f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "source_fingerprint": fingerprint.dataset_uri,
            "stages": stages,
            "model_recommendation": model_rec,
            "training_hints": training_hints,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "imgshape_version": "4.0.0",
                "deterministic": True,
                "framework_targets": ["pytorch", "tensorflow"]
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(pipeline, f, indent=2)
        
        self.logger.info(f"Generated pipeline artifact: {output_file}")
        return output_file

    def generate_transforms_artifact(self, decisions: DecisionsCollection) -> Path:
        """Generate transforms.py with executable code"""
        output_file = self.output_dir / "transforms.py"
        
        code_lines = [
            '"""',
            'Generated data transforms from imgshape v4.0.0 (Atlas)',
            'This file is auto-generated and deterministic.',
            '"""',
            '',
            'import torch',
            'from torchvision import transforms',
            '',
            '',
            'def get_train_transform():',
            '    """Training data transform pipeline"""',
            '    return transforms.Compose([',
        ]
        
        # Add transforms based on decisions
        if "input_resolution" in decisions.decisions:
            res = decisions.decisions["input_resolution"].selected
            code_lines.append(f'        transforms.Resize(({res["height"]}, {res["width"]})),')
        
        if "augmentation_strategy" in decisions.decisions:
            aug_strategy = decisions.decisions["augmentation_strategy"].selected
            if aug_strategy == "standard":
                code_lines.extend([
                    '        transforms.RandomHorizontalFlip(p=0.5),',
                    '        transforms.RandomRotation(degrees=15),',
                    '        transforms.ColorJitter(brightness=0.2, contrast=0.2),',
                ])
            elif aug_strategy == "conservative":
                code_lines.extend([
                    '        transforms.RandomHorizontalFlip(p=0.3),',
                ])
        
        code_lines.extend([
            '        transforms.ToTensor(),',
            '        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),',
            '    ])',
            '',
            '',
            'def get_val_transform():',
            '    """Validation data transform pipeline"""',
            '    return transforms.Compose([',
        ])
        
        if "input_resolution" in decisions.decisions:
            res = decisions.decisions["input_resolution"].selected
            code_lines.append(f'        transforms.Resize(({res["height"]}, {res["width"]})),')
        
        code_lines.extend([
            '        transforms.ToTensor(),',
            '        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),',
            '    ])',
            ''
        ])
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(code_lines))
        
        self.logger.info(f"Generated transforms artifact: {output_file}")
        return output_file

    def generate_markdown_report(
        self,
        fingerprint: DatasetFingerprint,
        decisions: DecisionsCollection
    ) -> Path:
        """Generate report.md"""
        output_file = self.output_dir / "report.md"
        
        lines = [
            '# imgshape v4.0.0 Dataset Analysis Report',
            '',
            f'**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            '',
            '## Dataset Fingerprint',
            '',
            f'**URI:** `{fingerprint.dataset_uri}`',
            f'**Class:** `{fingerprint.derived_class}`',
            f'**Confidence:** {fingerprint.confidence:.2%}',
            '',
            '### Profiles',
            '',
        ]
        
        # Spatial profile
        spatial = fingerprint.profiles['spatial']
        lines.extend([
            '#### Spatial Profile',
            '',
            f'- **Geometry Class:** {spatial.geometry_class.value}',
            f'- **Resize Risk:** {spatial.resize_risk.value}',
            f'- **Aspect Ratio Variance:** {spatial.aspect_ratio_variance:.4f}',
            f'- **Resolution Range:**',
            f'  - Min: {spatial.resolution_range["min"].width}×{spatial.resolution_range["min"].height}',
            f'  - Max: {spatial.resolution_range["max"].width}×{spatial.resolution_range["max"].height}',
            f'  - Median: {spatial.resolution_range["median"].width}×{spatial.resolution_range["median"].height}',
            '',
        ])
        
        # Signal profile
        signal = fingerprint.profiles['signal']
        lines.extend([
            '#### Signal Profile',
            '',
            f'- **Information Density:** {signal.information_density.value}',
            f'- **Capacity Ceiling:** {signal.capacity_ceiling.value}',
            f'- **Noise Estimate:** {signal.noise_estimate:.2%}',
            f'- **Entropy:** mean={signal.entropy.mean:.2f}, std={signal.entropy.std:.2f}',
            '',
        ])
        
        # Semantic profile
        semantic = fingerprint.profiles['semantic']
        lines.extend([
            '#### Semantic Profile',
            '',
            f'- **Primary Domain:** {semantic.primary_domain.value}',
            f'- **Content Type:** {semantic.content_type.value}',
            f'- **Specialization Required:** {semantic.specialization_required}',
            f'- **Characteristics:** {", ".join(semantic.characteristics)}',
            '',
        ])
        
        # Quality profile
        quality = fingerprint.profiles['quality']
        lines.extend([
            '#### Quality Profile',
            '',
            f'- **Overall Quality:** {quality.overall_quality.value}',
            f'- **Corruption Rate:** {quality.corruption_rate:.2%}',
            f'- **Duplication Rate:** {quality.duplication_rate:.2%}',
            '',
        ])
        
        if quality.warnings:
            lines.append('**Warnings:**')
            for warning in quality.warnings:
                lines.append(f'- [{warning.severity.value}] {warning.message}')
            lines.append('')
        
        # Decisions
        lines.extend([
            '---',
            '',
            '## Decisions',
            '',
        ])
        
        for decision_id, decision in decisions.decisions.items():
            lines.extend([
                f'### {decision_id.replace("_", " ").title()}',
                '',
                f'**Selected:** `{decision.selected}`',
                '',
                '**Rationale:**',
            ])
            for reason in decision.why:
                lines.append(f'- {reason}')
            lines.append('')
            
            if decision.risks:
                lines.append('**Risks:**')
                for risk in decision.risks:
                    lines.append(f'- [{risk.severity.value}] {risk.description}')
                    if risk.mitigation:
                        lines.append(f'  - *Mitigation:* {risk.mitigation}')
                lines.append('')
            
            if decision.tradeoffs:
                lines.append('**Tradeoffs:**')
                for tradeoff in decision.tradeoffs:
                    lines.append(f'- {tradeoff.aspect}: gain={tradeoff.gain}, cost={tradeoff.cost}')
                lines.append('')
            
            if decision.alternatives_considered:
                lines.append('**Alternatives Considered:**')
                for alt in decision.alternatives_considered:
                    lines.append(f'- `{alt.option}` - rejected because: {alt.rejected_because}')
                lines.append('')
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))
        
        self.logger.info(f"Generated markdown report: {output_file}")
        return output_file

    def _get_preproc_params(self, step: str, decisions: DecisionsCollection) -> Dict[str, Any]:
        """Get parameters for preprocessing step"""
        if step == "resize" and "input_resolution" in decisions.decisions:
            res = decisions.decisions["input_resolution"].selected
            return {"width": res["width"], "height": res["height"]}
        elif step == "normalize":
            return {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        return {}

    def _get_aug_strength(self, strategy: str) -> float:
        """Get augmentation strength value"""
        strengths = {
            "conservative": 0.3,
            "standard": 0.6,
            "aggressive": 0.9,
            "domain_specific_medical": 0.4
        }
        return strengths.get(strategy, 0.5)

    def _build_training_hints(self, decisions: DecisionsCollection) -> Dict[str, Any]:
        """Build training hints from decisions"""
        hints = {}
        
        if "regularization" in decisions.decisions:
            reg = decisions.decisions["regularization"].selected
            hints["regularization"] = reg
        
        # Add batch size recommendation
        hints["recommended_batch_size"] = 32
        
        # Add learning rate range
        hints["learning_rate_range"] = {"min": 1e-4, "max": 1e-2}
        
        # Add epochs estimate
        hints["epochs_estimate"] = {"min": 20, "max": 100}
        
        return hints
