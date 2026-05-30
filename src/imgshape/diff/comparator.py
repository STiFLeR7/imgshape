from typing import Optional, Any

from imgshape.atlas.drift import (
    calculate_signal_drift,
    calculate_spatial_drift,
    calculate_semantic_drift,
)
from imgshape.atlas.spatial import (
    SpatialProfile,
    Resolution,
    AspectRatioCluster,
    ResizeRisk,
    GeometryClass,
)
from imgshape.atlas.signal import (
    SignalProfile,
    EntropyStats,
    EntropyStabilityIndex,
    ColorHistogram,
    CapacityCeiling,
    InformationDensity,
)
from imgshape.diff.report import DiffReport


def _get_val(obj: Any, attr_path: str, default: Any = None) -> Any:
    current = obj
    for part in attr_path.split("."):
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(part, default)
        else:
            current = getattr(current, part, default)
    return current


def dict_to_spatial_profile(d: Any) -> Any:
    if d is None:
        return SpatialProfile(
            resolution_range={},
            aspect_ratio_variance=0.0,
            resize_risk=ResizeRisk.LOW,
            geometry_class=GeometryClass.UNIFORM,
        )
    if not isinstance(d, dict):
        return d

    res_range = {}
    for k, v in d.get("resolution_range", {}).items():
        res_range[k] = Resolution(width=v.get("width", 0), height=v.get("height", 0))

    clusters = []
    for c in d.get("aspect_ratio_clusters", []):
        clusters.append(
            AspectRatioCluster(
                ratio=c.get("ratio", 0.0), frequency=c.get("frequency", 0.0)
            )
        )

    return SpatialProfile(
        resolution_range=res_range,
        aspect_ratio_variance=d.get("aspect_ratio_variance", 0.0),
        resize_risk=ResizeRisk(d.get("resize_risk", "low")),
        geometry_class=GeometryClass(d.get("geometry_class", "uniform")),
        aspect_ratio_clusters=clusters,
    )


def dict_to_signal_profile(d: Any) -> Any:
    if d is None:
        return SignalProfile(
            entropy=EntropyStats(0.0, 0.0, {}),
            noise_estimate=0.0,
            capacity_ceiling=CapacityCeiling.LIGHTWEIGHT,
            information_density=InformationDensity.SPARSE,
        )
    if not isinstance(d, dict):
        return d

    ent_d = d.get("entropy", {})
    ent = EntropyStats(
        mean=ent_d.get("mean", 0.0),
        std=ent_d.get("std", 0.0),
        range=ent_d.get("range", {"min": 0.0, "max": 0.0}),
    )

    ch_d = d.get("color_histogram")
    ch = None
    if ch_d:
        ch = ColorHistogram(
            bins=ch_d.get("bins", []), channels=ch_d.get("channels", [])
        )

    es_d = d.get("entropy_stability")
    es = None
    if es_d:
        es = EntropyStabilityIndex(
            score=es_d.get("score", 0.0), is_stable=es_d.get("is_stable", True)
        )

    return SignalProfile(
        entropy=ent,
        noise_estimate=d.get("noise_estimate", 0.0),
        capacity_ceiling=CapacityCeiling(d.get("capacity_ceiling", "lightweight")),
        information_density=InformationDensity(d.get("information_density", "sparse")),
        channel_variance=d.get("channel_variance", {}),
        color_histogram=ch,
        entropy_stability=es,
    )


class DatasetDiff:
    @staticmethod
    def compare(
        baseline: Any,
        current: Any,
        semantic: bool = False,
        threshold: Optional[float] = None,
    ) -> DiffReport:
        if threshold is None:
            threshold = 0.25

        baseline_uri = _get_val(baseline, "dataset_uri", "baseline")
        current_uri = _get_val(current, "dataset_uri", "current")

        b_spa_raw = _get_val(baseline, "profiles.spatial")
        c_spa_raw = _get_val(current, "profiles.spatial")

        b_sig_raw = _get_val(baseline, "profiles.signal")
        c_sig_raw = _get_val(current, "profiles.signal")

        b_spa = dict_to_spatial_profile(b_spa_raw)
        c_spa = dict_to_spatial_profile(c_spa_raw)

        b_sig = dict_to_signal_profile(b_sig_raw)
        c_sig = dict_to_signal_profile(c_sig_raw)

        spatial_drift = calculate_spatial_drift(b_spa, c_spa)
        signal_drift = calculate_signal_drift(b_sig, c_sig)

        b_emb = _get_val(baseline, "profiles.semantic.latent_embedding")
        c_emb = _get_val(current, "profiles.semantic.latent_embedding")

        semantic_drift = 0.0
        use_semantic_weight = False

        if semantic:
            if b_emb is not None and c_emb is not None:
                semantic_drift = calculate_semantic_drift(b_emb, c_emb)
                use_semantic_weight = True

        if use_semantic_weight:
            overall_drift = (
                0.5 * semantic_drift + 0.3 * signal_drift + 0.2 * spatial_drift
            )
            weights_str = "50% semantic, 30% signal, 20% spatial"
        else:
            overall_drift = 0.6 * signal_drift + 0.4 * spatial_drift
            weights_str = "60% signal, 40% spatial"

        is_significant = overall_drift > threshold
        similarity_score = 1.0 - overall_drift

        rationale = []
        if is_significant:
            rationale.append(
                f"Overall drift ({overall_drift:.4f}) exceeds threshold ({threshold:.4f}), "
                f"indicating significant change."
            )
        else:
            rationale.append(
                f"Overall drift ({overall_drift:.4f}) is within threshold ({threshold:.4f}), "
                f"indicating stable distributions."
            )

        rationale.append(f"Drift weight configuration: {weights_str}.")
        rationale.append(
            f"Spatial drift: {spatial_drift:.4f} (reflects median resolution area shifts)."
        )
        rationale.append(
            f"Signal drift: {signal_drift:.4f} (reflects color histogram intersection shifts)."
        )
        if semantic:
            if use_semantic_weight:
                rationale.append(
                    f"Semantic drift: {semantic_drift:.4f} (calculated from latent embeddings)."
                )
            else:
                rationale.append(
                    "Semantic drift: 0.0000 (requested, but latent embeddings were missing)."
                )
        else:
            rationale.append("Semantic drift: 0.0000 (not requested).")

        return DiffReport(
            overall_drift=overall_drift,
            spatial_drift=spatial_drift,
            signal_drift=signal_drift,
            semantic_drift=semantic_drift,
            is_significant=is_significant,
            similarity_score=similarity_score,
            rationale=rationale,
            baseline_uri=baseline_uri,
            current_uri=current_uri,
        )
