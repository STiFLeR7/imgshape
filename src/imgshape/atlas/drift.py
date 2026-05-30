import numpy as np
from typing import List, Union


def calculate_semantic_drift(
    baseline_emb: Union[np.ndarray, List[float], None],
    current_emb: Union[np.ndarray, List[float], None],
) -> float:
    """
    Calculate semantic drift using cosine distance (1 - cosine similarity).
    Returns 1.0 if either is missing, or if both are empty/zero.
    """
    if baseline_emb is None or current_emb is None:
        return 1.0

    b_arr = np.array(baseline_emb)
    c_arr = np.array(current_emb)

    norm_b = np.linalg.norm(b_arr)
    norm_c = np.linalg.norm(c_arr)
    if norm_b == 0 or norm_c == 0:
        return 1.0

    similarity = np.dot(b_arr, c_arr) / (norm_b * norm_c)
    return float(1.0 - max(0.0, similarity))


def calculate_signal_drift(b_sig, c_sig) -> float:
    """
    Calculate signal drift based on color histogram intersection.
    Returns 1.0 if histogram information is missing.
    """
    if not hasattr(b_sig, "color_histogram") or not hasattr(c_sig, "color_histogram"):
        return 1.0
    if b_sig.color_histogram is None or c_sig.color_histogram is None:
        return 1.0

    b_hist = np.array(b_sig.color_histogram.bins)
    c_hist = np.array(c_sig.color_histogram.bins)

    # Normalize histograms
    b_hist_norm = b_hist / (np.sum(b_hist) + 1e-6)
    c_hist_norm = c_hist / (np.sum(c_hist) + 1e-6)

    # Histogram intersection: 1.0 is identical, 0.0 is completely different
    intersection = np.sum(np.minimum(b_hist_norm, c_hist_norm))
    return float(1.0 - intersection)


def calculate_spatial_drift(b_spa, c_spa) -> float:
    """
    Calculate spatial drift based on median resolution area shift.
    """
    if not hasattr(b_spa, "resolution_range") or not hasattr(c_spa, "resolution_range"):
        return 1.0
    if "median" not in b_spa.resolution_range or "median" not in c_spa.resolution_range:
        return 1.0

    b_res = b_spa.resolution_range["median"]
    c_res = c_spa.resolution_range["median"]

    b_area = b_res.area()
    c_area = c_res.area()

    res_drift = abs(b_area - c_area) / max(b_area, 1e-6)
    return float(min(1.0, res_drift))
