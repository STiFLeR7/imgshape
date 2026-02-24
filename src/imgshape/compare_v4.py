"""
imgshape.v4.compare — Dataset Comparison and Drift Analysis

This module implements the Dataset Comparison Module for imgshape v4.1.0.
It allows comparing two fingerprints to calculate drift scores, similarity indices,
and generate delta reports for auditability.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from pathlib import Path

from imgshape.fingerprint_v4 import DatasetFingerprint, SignalProfile, SpatialProfile

logger = logging.getLogger("imgshape.compare_v4")

@dataclass
class DriftScore:
    """Statistical drift between two datasets"""
    overall_drift: float  # 0.0 (none) to 1.0 (total)
    spatial_drift: float
    signal_drift: float
    is_significant: bool
    rationale: List[str]

@dataclass
class SimilarityIndex:
    """Geometric and distributional similarity between two datasets"""
    score: float  # 0.0 to 1.0
    shared_characteristics: List[str]
    divergent_characteristics: List[str]

class DatasetComparator:
    """
    Orchestrates comparison between two dataset fingerprints.
    Used for regression testing, active learning gating, and drift detection.
    """
    
    def compare(self, baseline: DatasetFingerprint, current: DatasetFingerprint) -> Dict[str, Any]:
        """
        Compare two fingerprints and return drift and similarity metrics.
        """
        logger.info(f"Comparing {baseline.dataset_uri} and {current.dataset_uri}")
        
        drift = self.calculate_drift_score(baseline, current)
        similarity = self.calculate_similarity_index(baseline, current)
        
        return {
            "drift": drift,
            "similarity": similarity,
            "baseline_uri": baseline.dataset_uri,
            "current_uri": current.dataset_uri
        }

    def calculate_drift_score(self, baseline: DatasetFingerprint, current: DatasetFingerprint) -> DriftScore:
        """Calculate statistical drift score using histogram intersection and metric deltas"""
        rationale = []
        
        # 1. Signal Drift (Entropy + Histograms)
        b_sig = baseline.profiles['signal']
        c_sig = current.profiles['signal']
        
        # Entropy delta
        entropy_delta = abs(b_sig.entropy.mean - c_sig.entropy.mean) / max(b_sig.entropy.mean, 1e-6)
        
        # Color Histogram Intersection
        sig_drift = 0.0
        if b_sig.color_histogram and c_sig.color_histogram:
            b_hist = np.array(b_sig.color_histogram.bins)
            c_hist = np.array(c_sig.color_histogram.bins)
            # Normalize
            b_hist = b_hist / (np.sum(b_hist) + 1e-6)
            c_hist = c_hist / (np.sum(c_hist) + 1e-6)
            # Intersection: 1.0 is identical, 0.0 is disjoint
            intersection = np.sum(np.minimum(b_hist, c_hist))
            sig_drift = 1.0 - intersection
            
        if sig_drift > 0.2:
            rationale.append(f"Significant signal drift detected ({sig_drift:.2f}) in color distribution")

        # 2. Spatial Drift (Resolution + Aspect Ratio)
        b_spa = baseline.profiles['spatial']
        c_spa = current.profiles['spatial']
        
        b_res = b_spa.resolution_range['median']
        c_res = c_spa.resolution_range['median']
        res_drift = abs(b_res.area() - c_res.area()) / max(b_res.area(), 1e-6)
        
        spa_drift = min(1.0, res_drift)
        if res_drift > 0.3:
            rationale.append(f"Median resolution shifted by {res_drift*100:.1f}%")

        # Overall weighted drift
        overall = (sig_drift * 0.6) + (spa_drift * 0.4)
        
        return DriftScore(
            overall_drift=float(overall),
            spatial_drift=float(spa_drift),
            signal_drift=float(sig_drift),
            is_significant=overall > 0.25,
            rationale=rationale
        )

    def calculate_similarity_index(self, baseline: DatasetFingerprint, current: DatasetFingerprint) -> SimilarityIndex:
        """Calculate high-level similarity index based on derived classes and profiles"""
        score = 0.0
        shared = []
        divergent = []
        
        # Class match
        if baseline.derived_class == current.derived_class:
            score += 0.4
            shared.append(f"Identical classification: {baseline.derived_class}")
        else:
            divergent.append(f"Classification mismatch: {baseline.derived_class} vs {current.derived_class}")
            
        # Domain match
        b_sem = baseline.profiles['semantic']
        c_sem = current.profiles['semantic']
        if b_sem.primary_domain == c_sem.primary_domain:
            score += 0.3
            shared.append(f"Shared domain: {b_sem.primary_domain.value}")
        else:
            divergent.append(f"Domain shift: {b_sem.primary_domain.value} -> {c_sem.primary_domain.value}")
            
        # Signal characteristics
        b_sig = baseline.profiles['signal']
        c_sig = current.profiles['signal']
        if b_sig.information_density == c_sig.information_density:
            score += 0.3
            shared.append("Compatible information density")
        else:
            divergent.append("Information density mismatch")
            
        return SimilarityIndex(
            score=float(score),
            shared_characteristics=shared,
            divergent_characteristics=divergent
        )

    def generate_delta_report(self, baseline: DatasetFingerprint, current: DatasetFingerprint) -> str:
        """Generate a Markdown delta report between two fingerprints"""
        drift = self.calculate_drift_score(baseline, current)
        sim = self.calculate_similarity_index(baseline, current)
        
        report = [
            f"# Dataset Delta Report: v4.1.0",
            f"**Baseline:** `{baseline.dataset_uri}`",
            f"**Current:** `{current.dataset_uri}`",
            "",
            "## 1. Executive Summary",
            f"- **Similarity Index:** `{sim.score:.2f}/1.00`",
            f"- **Drift Detection:** `{'SIGNIFICANT' if drift.is_significant else 'STABLE'}`",
            f"- **Overall Drift Score:** `{drift.overall_drift:.4f}`",
            "",
            "## 2. Statistical Analysis",
            "| Metric | Baseline | Current | Delta |",
            "| :--- | :--- | :--- | :--- |"
        ]
        
        # Add basic metric comparisons
        b_sig = baseline.profiles['signal']
        c_sig = current.profiles['signal']
        report.append(f"| Mean Entropy | {b_sig.entropy.mean:.2f} | {c_sig.entropy.mean:.2f} | {c_sig.entropy.mean - b_sig.entropy.mean:+.2f} |")
        
        b_res = baseline.profiles['spatial'].resolution_range['median']
        c_res = current.profiles['spatial'].resolution_range['median']
        report.append(f"| Median Res | {b_res.width}x{b_res.height} | {c_res.width}x{c_res.height} | - |")
        
        report.extend([
            "",
            "## 3. Drift Rationale",
        ])
        for r in drift.rationale:
            report.append(f"- {r}")
            
        if not drift.rationale:
            report.append("- No significant drift detected.")
            
        return "\n".join(report)
