import numpy as np
from PIL import Image

from imgshape.atlas import (
    Atlas,
    DatasetFingerprint,
    GPUHandler,
    BatchedGPUHandler,
    SpatialProfiler,
    SignalProfiler,
    DistributionProfiler,
    QualityProfiler,
    SemanticProfiler,
    PrimaryDomain,
    ContentType,
    MedicalProfile,
    SatelliteProfile,
    OCRProfile,
    calculate_semantic_drift,
    calculate_signal_drift,
    calculate_spatial_drift,
)


def create_dummy_dataset(tmp_path, domain="photographic", count=5):
    dataset_dir = tmp_path / domain
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        if domain == "medical":
            img_data = np.random.randint(50, 100, (128, 128), dtype=np.uint8)
            img = Image.fromarray(img_data, mode="L")
        elif domain == "ocr":
            img_data = np.random.randint(0, 255, (64, 256), dtype=np.uint8)
            img = Image.fromarray(img_data, mode="L")
        elif domain == "satellite":
            img_data = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            img_data[0:50, 0:50, :] = 250
            img = Image.fromarray(img_data, mode="RGB")
        else:
            img_data = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            img = Image.fromarray(img_data, mode="RGB")

        img.save(dataset_dir / f"img_{i}.png")

    return dataset_dir


def test_gpu_handler():
    # Since torch might not be installed, we check if is_cuda_available returns a bool
    # without raising exception.
    res = GPUHandler.is_cuda_available()
    assert isinstance(res, bool)
    device = GPUHandler.get_device()
    assert device in ("cuda", "cpu")


def test_batched_gpu_handler_fallback():
    # Instantiating BatchedGPUHandler should work even without PyTorch
    handler = BatchedGPUHandler(batch_size=2)
    img1 = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

    res1 = handler.add_image(img1)
    assert res1 is None

    res2 = handler.add_image(img2)
    assert isinstance(res2, list)
    assert len(res2) == 2
    for r in res2:
        assert "entropy" in r
        assert "blur" in r
        assert isinstance(r["entropy"], float)
        assert isinstance(r["blur"], float)


def test_spatial_profiler(tmp_path):
    dataset_dir = create_dummy_dataset(tmp_path, "photographic", count=3)
    atlas = Atlas()
    images = atlas._collect_images(dataset_dir)

    profile = SpatialProfiler.profile(images)
    assert profile.aspect_ratio_variance >= 0.0
    assert len(profile.aspect_ratio_clusters) > 0
    assert profile.resolution_range["median"].width == 128
    assert profile.resolution_range["median"].height == 128


def test_signal_profiler(tmp_path):
    dataset_dir = create_dummy_dataset(tmp_path, "photographic", count=3)
    atlas = Atlas()
    images = atlas._collect_images(dataset_dir)

    profile = SignalProfiler.profile(images)
    assert profile.entropy.mean > 0.0
    assert profile.noise_estimate >= 0.0
    assert profile.entropy_stability is not None
    assert profile.color_histogram is not None
    assert len(profile.color_histogram.bins) == 48


def test_distribution_profiler(tmp_path):
    dataset_dir = create_dummy_dataset(tmp_path, "photographic", count=2)
    atlas = Atlas()
    images = atlas._collect_images(dataset_dir)

    profile = DistributionProfiler.profile(images)
    assert profile.class_balance.class_count == 0
    assert profile.sampling_strategy.value == "uniform"


def test_quality_profiler(tmp_path):
    dataset_dir = create_dummy_dataset(tmp_path, "photographic", count=2)
    atlas = Atlas()
    images = atlas._collect_images(dataset_dir)

    profile = QualityProfiler.profile(images)
    assert profile.corruption_rate == 0.0
    assert isinstance(profile.warnings, list)


def test_semantic_profiler_medical(tmp_path):
    dataset_dir = create_dummy_dataset(tmp_path, "medical", count=3)
    atlas = Atlas()
    images = atlas._collect_images(dataset_dir)

    spatial = SpatialProfiler.profile(images)
    signal = SignalProfiler.profile(images)

    profile = SemanticProfiler.profile(images, spatial, signal)
    assert profile.primary_domain == PrimaryDomain.MEDICAL
    assert profile.content_type == ContentType.LOW_SIGNAL

    med_profile = SemanticProfiler.profile_medical(images)
    assert isinstance(med_profile, MedicalProfile)
    assert med_profile.hu_range[0] >= 0.0


def test_semantic_profiler_satellite(tmp_path):
    dataset_dir = create_dummy_dataset(tmp_path, "satellite", count=3)
    atlas = Atlas()
    images = atlas._collect_images(dataset_dir)

    spatial = SpatialProfiler.profile(images)
    signal = SignalProfiler.profile(images)

    profile = SemanticProfiler.profile(images, spatial, signal)
    assert profile.primary_domain == PrimaryDomain.SATELLITE

    sat_profile = SemanticProfiler.profile_satellite(images)
    assert isinstance(sat_profile, SatelliteProfile)
    assert sat_profile.gsd_estimate > 0.0
    assert sat_profile.cloud_cover_estimate >= 0.0


def test_semantic_profiler_ocr(tmp_path):
    dataset_dir = create_dummy_dataset(tmp_path, "ocr", count=3)
    atlas = Atlas()
    images = atlas._collect_images(dataset_dir)

    spatial = SpatialProfiler.profile(images)
    signal = SignalProfiler.profile(images)

    profile = SemanticProfiler.profile(images, spatial, signal)
    assert profile.primary_domain == PrimaryDomain.OCR

    ocr_profile = SemanticProfiler.profile_ocr(images)
    assert isinstance(ocr_profile, OCRProfile)
    assert ocr_profile.text_density > 0.0


def test_drift_calculation():
    emb1 = [0.1, 0.2, 0.3, 0.4]
    emb2 = [0.1, 0.25, 0.3, 0.38]

    drift_sem = calculate_semantic_drift(emb1, emb2)
    assert 0.0 <= drift_sem <= 1.0

    # Check signal drift (mocking profile structures)
    class DummyColorHist:
        def __init__(self, bins):
            self.bins = bins

    class DummySignalProfile:
        def __init__(self, bins):
            self.color_histogram = DummyColorHist(bins)

    sig1 = DummySignalProfile([1] * 48)
    sig2 = DummySignalProfile([1] * 48)
    drift_sig = calculate_signal_drift(sig1, sig2)
    assert abs(drift_sig - 0.0) < 1e-5

    # Check spatial drift (mocking profile structures)
    class DummyResolution:
        def __init__(self, w, h):
            self.w = w
            self.h = h

        def area(self):
            return self.w * self.h

    class DummySpatialProfile:
        def __init__(self, w, h):
            self.resolution_range = {"median": DummyResolution(w, h)}

    spa1 = DummySpatialProfile(128, 128)
    spa2 = DummySpatialProfile(128, 128)
    drift_spa = calculate_spatial_drift(spa1, spa2)
    assert abs(drift_spa - 0.0) < 1e-5


def test_atlas_orchestrator(tmp_path):
    dataset_dir = create_dummy_dataset(tmp_path, "photographic", count=4)
    atlas = Atlas(sample_limit=3, batch_size=2)

    fingerprint = atlas.extract(dataset_dir)
    assert isinstance(fingerprint, DatasetFingerprint)
    assert fingerprint.schema_version == "5.0.0"
    assert fingerprint.metadata["sample_count"] == 3
    assert fingerprint.derived_class.startswith("vision.")

    # Test to_dict
    fingerprint_dict = fingerprint.to_dict()
    assert isinstance(fingerprint_dict, dict)
    assert fingerprint_dict["schema_version"] == "5.0.0"
    assert isinstance(fingerprint_dict["profiles"]["spatial"]["resize_risk"], str)
