from dataclasses import dataclass, asdict
from typing import List, Dict, Any


@dataclass
class DiffReport:
    overall_drift: float
    spatial_drift: float
    signal_drift: float
    semantic_drift: float
    is_significant: bool
    similarity_score: float
    rationale: List[str]
    baseline_uri: str
    current_uri: str

    def summary(self) -> str:
        lines = []
        lines.append("# Dataset Diff Report")
        lines.append(f"- **Baseline:** `{self.baseline_uri}`")
        lines.append(f"- **Current:** `{self.current_uri}`")
        lines.append("")
        lines.append("## Drift Metrics")
        lines.append(f"- **Overall Drift:** {self.overall_drift:.4f}")
        lines.append(f"- **Similarity Score:** {self.similarity_score:.4f}")
        lines.append(f"- **Spatial Drift:** {self.spatial_drift:.4f}")
        lines.append(f"- **Signal Drift:** {self.signal_drift:.4f}")
        lines.append(f"- **Semantic Drift:** {self.semantic_drift:.4f}")
        lines.append("")
        sig_str = (
            "SIGNIFICANT DRIFT DETECTED"
            if self.is_significant
            else "No significant drift detected"
        )
        lines.append(f"**Verdict:** {sig_str}")
        lines.append("")
        lines.append("## Rationale")
        for item in self.rationale:
            lines.append(f"- {item}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
