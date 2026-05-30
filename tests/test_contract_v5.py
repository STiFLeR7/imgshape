import pytest
import numpy as np
from PIL import Image

from imgshape.atlas import Atlas
from imgshape.contract import Contract, ContractSchemaError


def test_yaml_loader_success(tmp_path):
    contract_yaml = """
dataset:
  expected_channels: 3
  allowed_formats: [png, jpg, jpeg]
  resolution_min: [128, 128]
  resolution_max: [1024, 1024]
quality:
  corruption_max: 0.05
  duplicate_max: 0.10
  blur_threshold: 40.0
distribution:
  entropy_min: 4.5
  entropy_max: 8.0
  channel_consistency: strict
"""
    contract_file = tmp_path / "valid_contract.yaml"
    contract_file.write_text(contract_yaml, encoding="utf-8")

    contract = Contract.load(contract_file)
    assert contract.clauses.dataset.expected_channels == 3
    assert contract.clauses.dataset.allowed_formats == ["png", "jpg", "jpeg"]
    assert contract.clauses.dataset.resolution_min == (128, 128)
    assert contract.clauses.dataset.resolution_max == (1024, 1024)
    assert contract.clauses.quality.corruption_max == 0.05
    assert contract.clauses.quality.duplicate_max == 0.10
    assert contract.clauses.quality.blur_threshold == 40.0
    assert contract.clauses.distribution.entropy_min == 4.5
    assert contract.clauses.distribution.entropy_max == 8.0
    assert contract.clauses.distribution.channel_consistency == "strict"


def test_yaml_loader_invalid_top_level(tmp_path):
    contract_yaml = """
invalid_section:
  some_key: some_value
"""
    contract_file = tmp_path / "invalid_top.yaml"
    contract_file.write_text(contract_yaml, encoding="utf-8")

    with pytest.raises(ContractSchemaError) as excinfo:
        Contract.load(contract_file)
    assert "Invalid top-level key" in str(excinfo.value)


def test_yaml_loader_invalid_types(tmp_path):
    contract_yaml = """
dataset:
  expected_channels: "not-an-int"
"""
    contract_file = tmp_path / "invalid_types.yaml"
    contract_file.write_text(contract_yaml, encoding="utf-8")

    with pytest.raises(ContractSchemaError) as excinfo:
        Contract.load(contract_file)
    assert "expected_channels must be an integer" in str(excinfo.value)


def test_yaml_loader_invalid_resolution(tmp_path):
    contract_yaml = """
dataset:
  resolution_min: [128]
"""
    contract_file = tmp_path / "invalid_resolution.yaml"
    contract_file.write_text(contract_yaml, encoding="utf-8")

    with pytest.raises(ContractSchemaError) as excinfo:
        Contract.load(contract_file)
    assert "resolution_min must contain exactly 2 integers" in str(excinfo.value)


def test_yaml_loader_syntax_error(tmp_path):
    contract_yaml = """
dataset:
  expected_channels: : : invalid yaml syntax
"""
    contract_file = tmp_path / "syntax_error.yaml"
    contract_file.write_text(contract_yaml, encoding="utf-8")

    with pytest.raises(ContractSchemaError) as excinfo:
        Contract.load(contract_file)
    assert "YAML syntax error" in str(excinfo.value)


def create_dummy_images(
    tmp_path,
    sizes=[(128, 128), (256, 256)],
    modes=["RGB", "RGB"],
    formats=["png", "jpg"],
):
    dataset_dir = tmp_path / "test_dataset"
    if dataset_dir.exists():
        import shutil

        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(exist_ok=True, parents=True)

    for i, (size, mode, fmt) in enumerate(zip(sizes, modes, formats)):
        if mode == "L":
            img_data = np.random.randint(0, 255, (size[1], size[0]), dtype=np.uint8)
        else:
            img_data = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        img = Image.fromarray(img_data, mode=mode)
        img_path = dataset_dir / f"img_{i}.{fmt}"
        img.save(img_path)

    return dataset_dir


def test_contract_validation_pass_scenario(tmp_path):
    # Create dataset with two 256x256 RGB images, saved as png
    dataset_dir = create_dummy_images(
        tmp_path,
        sizes=[(256, 256), (256, 256)],
        modes=["RGB", "RGB"],
        formats=["png", "png"],
    )

    # Extract fingerprint
    atlas = Atlas()
    fingerprint = atlas.extract(dataset_dir)

    # Define contract that should pass
    contract_yaml = """
dataset:
  expected_channels: 3
  allowed_formats: [png]
  resolution_min: [128, 128]
  resolution_max: [512, 512]
quality:
  corruption_max: 0.1
  duplicate_max: 0.2
  blur_threshold: 10.0
distribution:
  entropy_min: 2.0
  entropy_max: 8.0
  channel_consistency: strict
"""
    contract_file = tmp_path / "passing_contract.yaml"
    contract_file.write_text(contract_yaml, encoding="utf-8")

    contract = Contract.load(contract_file)
    report = contract.validate(fingerprint)

    # Should pass without violations
    assert report.passed is True
    assert len(report.violations) == 0
    assert report.fingerprint_id == fingerprint.dataset_uri
    assert report.timestamp.endswith("Z")


def test_contract_validation_fail_scenario(tmp_path):
    # Create dataset with one 64x64 grayscale and one 512x512 color image, saved as png & bmp
    dataset_dir = create_dummy_images(
        tmp_path,
        sizes=[(64, 64), (512, 512)],
        modes=["L", "RGB"],
        formats=["png", "bmp"],
    )

    atlas = Atlas()
    fingerprint = atlas.extract(dataset_dir)

    # Define contract that will fail on multiple clauses:
    # 1. expected_channels = 3 (mixed channels will trigger channel_consistency warning/fail)
    # 2. allowed_formats: [png, jpg] (bmp is present, so fails)
    # 3. resolution_min: [128, 128] (64x64 is below minimum, so fails)
    # 4. resolution_max: [256, 256] (512x512 is above maximum, so fails)
    # 5. channel_consistency: strict (fails because one image is grayscale and one is RGB)
    # 6. blur_threshold: 999999.0 (fails because mean_blur will be less than this huge number)
    contract_yaml = """
dataset:
  expected_channels: 3
  allowed_formats: [png, jpg]
  resolution_min: [128, 128]
  resolution_max: [256, 256]
quality:
  blur_threshold: 999999.0
distribution:
  channel_consistency: strict
"""
    contract_file = tmp_path / "failing_contract.yaml"
    contract_file.write_text(contract_yaml, encoding="utf-8")

    contract = Contract.load(contract_file)
    report = contract.validate(fingerprint)

    assert report.passed is False
    assert len(report.violations) > 0

    violated_clauses = {v.clause for v in report.violations}
    assert "dataset.allowed_formats" in violated_clauses
    assert "dataset.resolution_min" in violated_clauses
    assert "dataset.resolution_max" in violated_clauses
    assert "quality.blur_threshold" in violated_clauses
    assert "distribution.channel_consistency" in violated_clauses
