import pytest
import json
import tempfile
import time
from pathlib import Path
import numpy as np
from PIL import Image

from imgshape.test import TestRunner, TestReport
from imgshape.test.assertions import (
    assert_corruption,
    assert_duplicates,
    assert_resolution_min,
    assert_entropy_min,
    assert_channel_consistency,
    assert_formats,
)
from imgshape.diff import DatasetDiff, DiffReport
from imgshape.audit import generate_provenance_id, FingerprintLock, AuditTrailWriter
from imgshape.atlas.orchestrator import DatasetFingerprint, Atlas
from imgshape.contract import Contract
from imgshape.atlas.distribution import DistributionProfiler
from imgshape.atlas.quality import QualityProfiler


# Create a mock DatasetFingerprint helper
def make_mock_fingerprint(
    corruption_rate=0.0,
    duplication_rate=0.0,
    min_w=64,
    min_h=64,
    entropy_mean=3.5,
    grayscale_consistency=True,
    grayscale_ratio=0.0,
    formats=None,
    latent_embedding=None,
    dataset_uri="mock_dataset",
):
    if formats is None:
        formats = ["png", "jpg"]

    profiles = {
        "spatial": {
            "resolution_range": {
                "min": {"width": min_w, "height": min_h},
                "median": {"width": min_w * 2, "height": min_h * 2},
                "max": {"width": min_w * 4, "height": min_h * 4},
            },
            "aspect_ratio_variance": 0.02,
            "resize_risk": "low",
            "geometry_class": "diverse",
            "aspect_ratio_clusters": [
                {"ratio": 1.0, "frequency": 0.8},
                {"ratio": 1.5, "frequency": 0.2},
            ],
        },
        "signal": {
            "entropy": {
                "mean": entropy_mean,
                "std": 0.5,
                "range": {"min": entropy_mean - 1.0, "max": entropy_mean + 1.0},
            },
            "noise_estimate": 0.1,
            "capacity_ceiling": "standard",
            "information_density": "moderate",
            "color_histogram": {
                "bins": [10, 20, 30, 40, 50],
                "channels": ["R", "G", "B"],
            },
        },
        "quality": {
            "overall_quality": "excellent",
            "warnings": [],
            "corruption_rate": corruption_rate,
            "duplication_rate": duplication_rate,
            "mean_blur": 150.0,
        },
        "semantic": {
            "primary_domain": "photographic",
            "content_type": "natural",
            "characteristics": [],
            "specialization_required": False,
            "latent_embedding": latent_embedding,
        },
    }

    return DatasetFingerprint(
        schema_version="5.0.0",
        dataset_uri=dataset_uri,
        profiles=profiles,
        derived_class="PhotographicClass",
        confidence=0.95,
        metadata={
            "sample_count": 10,
            "imgshape_version": "5.0.0",
            "acceleration": "cpu",
            "formats": formats,
            "channel_counts": 3,
            "grayscale_consistency": grayscale_consistency,
            "grayscale_ratio": grayscale_ratio,
        },
    )


# -----------------------------------------------------------------------------
# 1. Test Assertions & TestRunner
# -----------------------------------------------------------------------------


def test_assertions():
    fp = make_mock_fingerprint(
        corruption_rate=0.005,
        duplication_rate=0.03,
        min_w=64,
        min_h=64,
        entropy_mean=3.2,
        grayscale_consistency=True,
        formats=["png", "jpg"],
    )

    # Corruption Assertion
    status, actual, expected, msg = assert_corruption(fp, max_rate=0.01)
    assert status == "PASS"
    assert actual == 0.005
    assert expected == 0.01

    status, _, _, _ = assert_corruption(fp, max_rate=0.001)
    assert status == "FAIL"

    # Duplicates Assertion
    status, actual, expected, msg = assert_duplicates(fp, max_rate=0.05)
    assert status == "PASS"

    status, _, _, _ = assert_duplicates(fp, max_rate=0.01)
    assert status == "FAIL"

    # Resolution Assertion
    status, actual, expected, msg = assert_resolution_min(fp, min_w=32, min_h=32)
    assert status == "PASS"
    assert actual == (64, 64)
    assert expected == (32, 32)

    status, _, _, _ = assert_resolution_min(fp, min_w=128, min_h=128)
    assert status == "FAIL"

    # Entropy Assertion
    status, actual, expected, msg = assert_entropy_min(fp, min_val=2.5)
    assert status == "PASS"
    assert actual == 3.2

    status, _, _, _ = assert_entropy_min(fp, min_val=4.0)
    assert status == "FAIL"

    # Channel Consistency Assertion
    status, actual, expected, msg = assert_channel_consistency(fp, strict=True)
    assert status == "PASS"

    fp_mixed = make_mock_fingerprint(grayscale_consistency=False, grayscale_ratio=0.4)
    status, actual, expected, msg = assert_channel_consistency(fp_mixed, strict=True)
    assert status == "FAIL"
    status, actual, expected, msg = assert_channel_consistency(fp_mixed, strict=False)
    assert status == "WARN"

    # Formats Assertion
    status, actual, expected, msg = assert_formats(
        fp, allowed_formats=["png", "jpg", "gif"]
    )
    assert status == "PASS"

    status, actual, expected, msg = assert_formats(fp, allowed_formats=["png"])
    assert status == "FAIL"


def test_test_runner():
    fp = make_mock_fingerprint(
        corruption_rate=0.005,
        duplication_rate=0.03,
        min_w=64,
        min_h=64,
        entropy_mean=3.2,
        grayscale_consistency=True,
        formats=["png", "jpg"],
    )

    # Run all
    report = TestRunner.run(fp)
    assert isinstance(report, TestReport)
    assert report.passed is True
    assert len(report.results) == 6
    summary_str = report.summary()
    assert "Overall Result: PASS" in summary_str
    assert "Passed:      6" in summary_str

    report_dict = report.to_dict()
    assert report_dict["passed"] is True
    assert len(report_dict["results"]) == 6

    # Run Quality Suite only (corruption, duplicates, resolution_min, formats)
    report_quality = TestRunner.run(fp, only_suite="quality")
    assert report_quality.passed is True
    assert len(report_quality.results) == 4
    names = [r["name"] for r in report_quality.results]
    assert "assert_corruption" in names
    assert "assert_duplicates" in names
    assert "assert_resolution_min" in names
    assert "assert_formats" in names

    # Run Distribution Suite only (entropy, channel_consistency)
    report_dist = TestRunner.run(fp, only_suite="distribution")
    assert report_dist.passed is True
    assert len(report_dist.results) == 2
    names = [r["name"] for r in report_dist.results]
    assert "assert_entropy_min" in names
    assert "assert_channel_consistency" in names

    # Invalid suite
    with pytest.raises(ValueError):
        TestRunner.run(fp, only_suite="invalid_suite")


# -----------------------------------------------------------------------------
# 2. Test Dataset Diff Comparator
# -----------------------------------------------------------------------------


def test_dataset_diff():
    # Setup baseline and current fingerprints
    baseline = make_mock_fingerprint(
        min_w=100, min_h=100, entropy_mean=4.0, dataset_uri="baseline_ds"
    )
    # Different median resolution (to trigger spatial drift)
    # Different color histogram to trigger signal drift
    current = make_mock_fingerprint(
        min_w=200, min_h=200, entropy_mean=4.0, dataset_uri="current_ds"
    )
    # Manually edit the color histogram to simulate different distribution
    current.profiles["signal"]["color_histogram"]["bins"] = [50, 40, 30, 20, 10]

    # 1. Non-semantic comparison (weights 60% signal drift, 40% spatial drift)
    report = DatasetDiff.compare(baseline, current, semantic=False)
    assert isinstance(report, DiffReport)
    assert report.semantic_drift == 0.0
    assert report.spatial_drift > 0.0
    assert report.signal_drift > 0.0

    # Check overall drift formula: 0.6 * signal + 0.4 * spatial
    expected_overall = 0.6 * report.signal_drift + 0.4 * report.spatial_drift
    assert pytest.approx(report.overall_drift) == expected_overall
    assert pytest.approx(report.similarity_score) == 1.0 - expected_overall

    summary_str = report.summary()
    assert "Baseline:" in summary_str
    assert "Current:" in summary_str
    assert "Overall Drift:" in summary_str
    assert "Drift weight configuration: 60% signal, 40% spatial." in report.rationale[1]

    # 2. Semantic compare with missing embeddings (fallback to 60/40, semantic_drift=0.0)
    report_fallback = DatasetDiff.compare(baseline, current, semantic=True)
    assert report_fallback.semantic_drift == 0.0
    assert pytest.approx(report_fallback.overall_drift) == expected_overall
    assert any(
        "requested, but latent embeddings were missing" in r
        for r in report_fallback.rationale
    )

    # 3. Semantic comparison with embeddings present (weights 50% semantic, 30% signal, 20% spatial)
    baseline_emb = make_mock_fingerprint(latent_embedding=[1.0, 0.0, 0.0])
    current_emb = make_mock_fingerprint(
        latent_embedding=[0.0, 1.0, 0.0]
    )  # cosine similarity is 0.0, so cosine distance is 1.0

    report_semantic = DatasetDiff.compare(
        baseline_emb, current_emb, semantic=True, threshold=0.5
    )
    assert report_semantic.semantic_drift == 1.0
    # Expected overall: 0.5 * 1.0 + 0.3 * signal_drift + 0.2 * spatial_drift
    expected_sem_overall = (
        0.5 * 1.0
        + 0.3 * report_semantic.signal_drift
        + 0.2 * report_semantic.spatial_drift
    )
    assert pytest.approx(report_semantic.overall_drift) == expected_sem_overall
    assert report_semantic.is_significant is True  # overall drift > 0.5


# -----------------------------------------------------------------------------
# 3. Test Audit Trails and Lockfiles
# -----------------------------------------------------------------------------


def test_provenance_id():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Write some files
        (tmp_path / "file1.png").write_text("dummy image 1")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file2.jpg").write_text("dummy image 2")

        id1 = generate_provenance_id(tmp_path)
        # Deteministic: calling again on same directory gives same ID
        id2 = generate_provenance_id(tmp_path)
        assert id1 == id2

        # Change file content / size
        time.sleep(0.01)  # ensure modified timestamp might change
        (tmp_path / "file1.png").write_text("dummy image 1 changed content")

        id3 = generate_provenance_id(tmp_path)
        assert id1 != id3


def test_fingerprint_lock():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text("dataset:\n  allowed_formats: [jpg]\n")

        fp = make_mock_fingerprint()
        # Set a mock dataset_id / provenance_id in fingerprint metadata
        fp.metadata["provenance_id"] = "mock_provenance_sha256"

        lock_path = tmp_path / "custom_lock_name.json"

        # Write lock
        lock_data = FingerprintLock.write_lock(
            contract_path=contract_path,
            fingerprint=fp,
            report_passed=True,
            lock_path=lock_path,
        )

        assert lock_path.exists()
        assert lock_data["contract"] == "contract.yaml"
        assert lock_data["dataset_id"] == "mock_provenance_sha256"
        assert lock_data["result"] == "PASS"
        assert lock_data["imgshape_version"] == "5.0.0"

        # Read lock
        read_data = FingerprintLock.read_lock(lock_path)
        assert read_data == lock_data

        # Verify lock
        assert FingerprintLock.verify_lock(lock_path, fp) is True

        # Verify fails if current fingerprint dataset_id doesn't match
        fp_diff = make_mock_fingerprint()
        fp_diff.metadata["provenance_id"] = "different_provenance_sha256"
        assert FingerprintLock.verify_lock(lock_path, fp_diff) is False

        # Verify fails if result was FAIL
        FingerprintLock.write_lock(
            contract_path=contract_path,
            fingerprint=fp,
            report_passed=False,
            lock_path=lock_path,
        )
        assert FingerprintLock.verify_lock(lock_path, fp) is False


def test_audit_trail_writer():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        trail_file = tmp_path / "audit_trail.jsonl"

        writer = AuditTrailWriter(trail_file)
        writer.write_verdict(
            dataset_uri="mock_dataset_uri",
            verdict="PASS",
            details={"violation_count": 0},
        )

        writer.write_verdict(
            dataset_uri="mock_dataset_uri",
            verdict="FAIL",
            details={
                "violation_count": 2,
                "violations": ["assert_corruption", "assert_formats"],
            },
        )

        assert trail_file.exists()

        # Read the lines
        with open(trail_file, "r") as f:
            lines = f.readlines()

        assert len(lines) == 2
        entry1 = json.loads(lines[0])
        assert entry1["dataset_uri"] == "mock_dataset_uri"
        assert entry1["verdict"] == "PASS"
        assert entry1["details"]["violation_count"] == 0

        entry2 = json.loads(lines[1])
        assert entry2["verdict"] == "FAIL"
        assert entry2["details"]["violation_count"] == 2


# -----------------------------------------------------------------------------
# 4. New v5 Robustness & Audit Fixes Tests
# -----------------------------------------------------------------------------


def test_distribution_profiler_with_classes(tmp_path):
    # Create directory structure for a dataset with classes
    class_a_dir = tmp_path / "class_a"
    class_b_dir = tmp_path / "class_b"
    class_a_dir.mkdir()
    class_b_dir.mkdir()

    # We need 2 images in class_a and 1 in class_b to get 2.0 imbalance ratio
    img_data = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    # Save images
    Image.fromarray(img_data).save(class_a_dir / "img1.png")
    Image.fromarray(img_data).save(class_a_dir / "img2.png")
    Image.fromarray(img_data).save(class_b_dir / "img3.png")

    atlas = Atlas()
    images = atlas._collect_images(tmp_path)

    # Run DistributionProfiler
    profile = DistributionProfiler.profile(images)

    assert profile.class_balance.class_count == 2
    assert profile.class_balance.imbalance_ratio == 2.0
    assert profile.class_balance.type.value == "slightly_imbalanced"
    assert profile.sampling_strategy.value == "weighted"


def test_contradiction_and_null_contracts(tmp_path):
    from imgshape.contract import ContractSchemaError

    # 1. Null sections should default to empty dicts without crashing
    null_yaml = """
dataset: null
quality:
  blur_threshold: 10.0
distribution: null
"""
    contract_file = tmp_path / "null_contract.yaml"
    contract_file.write_text(null_yaml, encoding="utf-8")

    contract = Contract.load(contract_file)
    assert contract.clauses.dataset.resolution_min is None
    assert contract.clauses.quality.blur_threshold == 10.0
    assert contract.clauses.distribution.entropy_min is None

    # 2. Contradictory resolution min/max should raise ContractSchemaError
    contra_res_yaml = """
dataset:
  resolution_min: [256, 256]
  resolution_max: [128, 128]
"""
    contra_file = tmp_path / "contra_res.yaml"
    contra_file.write_text(contra_res_yaml, encoding="utf-8")
    with pytest.raises(ContractSchemaError) as excinfo:
        Contract.load(contra_file)
    assert "Contradictory resolution constraints" in str(excinfo.value)

    # 3. Contradictory entropy min/max should raise ContractSchemaError
    contra_ent_yaml = """
distribution:
  entropy_min: 6.0
  entropy_max: 4.0
"""
    contra_file2 = tmp_path / "contra_ent.yaml"
    contra_file2.write_text(contra_ent_yaml, encoding="utf-8")
    with pytest.raises(ContractSchemaError) as excinfo:
        Contract.load(contra_file2)
    assert "Contradictory entropy constraints" in str(excinfo.value)


def test_custom_severity_and_warnings(tmp_path):
    # Define contract with a warning-severity clause
    warn_yaml = """
quality:
  duplicate_max:
    value: 0.05
    severity: warning
"""
    contract_file = tmp_path / "warn_contract.yaml"
    contract_file.write_text(warn_yaml, encoding="utf-8")

    contract = Contract.load(contract_file)
    assert contract.clauses.severities["quality.duplicate_max"] == "warning"

    # Create fingerprint with duplicate violation
    fp = make_mock_fingerprint(duplication_rate=0.08)
    report = contract.validate(fp)

    assert report.passed is False
    assert len(report.violations) == 1
    assert report.violations[0].severity == "warning"


def test_quality_profiler_corruption_and_duplication(tmp_path):
    # 1. Corruption calculations
    img_data = {
        "size": (128, 128),
        "mode": "RGB",
        "is_grayscale": False,
        "entropy": 4.5,
        "mean": [120, 120, 120],
        "variance": [1000, 1000, 1000],
        "extrema": [(0, 255)],
        "histogram": [10] * 768,
        "blur_score": 150.0,
        "hash": "hash1",
    }

    profile = QualityProfiler.profile([img_data], failed_count=1)
    # 1 loaded, 1 failed -> total 2 attempted -> 50% corruption
    assert profile.corruption_rate == 0.5
    assert any(w.category.value == "corruption" for w in profile.warnings)

    # 2. Duplication calculations
    img_data_1 = dict(img_data)
    img_data_2 = dict(img_data)  # same hash 'hash1'

    profile_dup = QualityProfiler.profile([img_data_1, img_data_2])
    # 2 loaded, same hashes -> 1 duplicate -> 50% duplication
    assert profile_dup.duplication_rate == 0.5
    assert any(w.category.value == "duplication" for w in profile_dup.warnings)
