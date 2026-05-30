import json
import numpy as np
from pathlib import Path
from PIL import Image
from click.testing import CliRunner

from imgshape.cli.main import main
from imgshape.atlas import Atlas
from imgshape.audit import FingerprintLock


def create_dummy_images(
    tmp_path: Path,
    sizes=[(256, 256), (256, 256)],
    modes=["RGB", "RGB"],
    formats=["png", "png"],
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


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "validate" in result.output
    assert "test" in result.output
    assert "diff" in result.output
    assert "fingerprint" in result.output


def test_cli_validate_pass(tmp_path):
    dataset_dir = create_dummy_images(tmp_path)

    contract_yaml = """
dataset:
  expected_channels: 3
  allowed_formats: [png]
  resolution_min: [128, 128]
  resolution_max: [512, 512]
quality:
  blur_threshold: 1.0
"""
    contract_file = tmp_path / "pass_contract.yaml"
    contract_file.write_text(contract_yaml, encoding="utf-8")

    runner = CliRunner()

    # 1. Test validation success in text format
    result = runner.invoke(main, ["validate", str(dataset_dir), str(contract_file)])
    assert result.exit_code == 0
    assert "IMGSHAPE CONTRACT VALIDATION REPORT" in result.output
    assert "Overall Verdict: PASS" in result.output

    # 2. Test validation success in json format and save
    save_json = tmp_path / "report.json"
    result = runner.invoke(
        main,
        [
            "validate",
            str(dataset_dir),
            str(contract_file),
            "--output",
            "json",
            "--save",
            str(save_json),
            "--lock",
        ],
    )
    assert result.exit_code == 0
    assert save_json.exists()

    # Read saved JSON and verify structure
    with open(save_json, "r") as f:
        data = json.load(f)
    assert data["passed"] is True

    # Verify fingerprint lock file next to contract
    lock_file = contract_file.parent / ".fingerprint_lock"
    assert lock_file.exists()
    with open(lock_file, "r") as f:
        lock_data = json.load(f)
    assert lock_data["result"] == "PASS"


def test_cli_validate_fail(tmp_path):
    dataset_dir = create_dummy_images(tmp_path, sizes=[(64, 64)], formats=["bmp"])

    contract_yaml = """
dataset:
  expected_channels: 3
  allowed_formats: [png]
  resolution_min: [128, 128]
"""
    contract_file = tmp_path / "fail_contract.yaml"
    contract_file.write_text(contract_yaml, encoding="utf-8")

    runner = CliRunner()

    # Test validation failure in markdown format
    result = runner.invoke(
        main, ["validate", str(dataset_dir), str(contract_file), "--output", "markdown"]
    )
    assert result.exit_code == 1
    assert "# imgshape Contract Validation Report" in result.output
    assert "VALIDATION FAILED" in result.output
    assert "dataset.allowed_formats" in result.output


def test_cli_test_subcommand(tmp_path):
    dataset_dir = create_dummy_images(tmp_path)
    runner = CliRunner()

    # Test quality suite only
    result = runner.invoke(main, ["test", str(dataset_dir), "--only", "quality"])
    assert result.exit_code == 0
    assert "IMGSHAPE DATASET TEST REPORT" in result.output
    assert "Overall Verdict: PASS" in result.output

    # Test JSON output format with save path
    save_test_json = tmp_path / "test_report.json"
    result = runner.invoke(
        main,
        ["test", str(dataset_dir), "--output", "json", "--save", str(save_test_json)],
    )
    assert result.exit_code == 0
    assert save_test_json.exists()

    with open(save_test_json, "r") as f:
        data = json.load(f)
    assert data["passed"] is True
    assert len(data["results"]) == 6


def test_cli_fingerprint_subcommand(tmp_path):
    dataset_dir = create_dummy_images(tmp_path)
    runner = CliRunner()

    # Test text format
    result = runner.invoke(main, ["fingerprint", str(dataset_dir)])
    assert result.exit_code == 0
    assert "IMGSHAPE DATASET FINGERPRINT" in result.output

    # Test JSON format and lock/save options
    fp_save = tmp_path / "fp.json"
    result = runner.invoke(
        main,
        [
            "fingerprint",
            str(dataset_dir),
            "--output",
            "json",
            "--save",
            str(fp_save),
            "--lock",
        ],
    )
    assert result.exit_code == 0
    assert fp_save.exists()

    # Lock file should exist in dataset directory
    lock_file = dataset_dir / ".fingerprint_lock"
    assert lock_file.exists()


def test_cli_diff_subcommand(tmp_path):
    dataset1 = create_dummy_images(tmp_path / "ds1", sizes=[(256, 256)])
    dataset2 = create_dummy_images(tmp_path / "ds2", sizes=[(128, 128)])

    runner = CliRunner()

    # 1. Compare using directories (should exit 1 because of drift > 0.25)
    result = runner.invoke(main, ["diff", str(dataset1), str(dataset2)])
    assert result.exit_code == 1
    assert "IMGSHAPE DATASET DIFF REPORT" in result.output

    # Compare with high threshold override (should exit 0)
    result_high = runner.invoke(
        main, ["diff", str(dataset1), str(dataset2), "--threshold", "0.8"]
    )
    assert result_high.exit_code == 0

    # 2. Extract fingerprints and save as JSON
    fp1_path = tmp_path / "fp1.json"
    fp2_path = tmp_path / "fp2.json"

    runner.invoke(
        main,
        ["fingerprint", str(dataset1), "--output", "json", "--save", str(fp1_path)],
    )
    runner.invoke(
        main,
        ["fingerprint", str(dataset2), "--output", "json", "--save", str(fp2_path)],
    )

    # Compare using JSON fingerprint files (exits 1)
    result = runner.invoke(
        main, ["diff", str(fp1_path), str(fp2_path), "--output", "markdown"]
    )
    assert result.exit_code == 1
    assert "# imgshape Dataset Diff Report" in result.output

    # Compare using JSON fingerprint files with threshold (exits 0)
    result_ok = runner.invoke(
        main,
        [
            "diff",
            str(fp1_path),
            str(fp2_path),
            "--output",
            "markdown",
            "--threshold",
            "0.8",
        ],
    )
    assert result_ok.exit_code == 0

    # 3. Save lock file and compare using lock files
    lock_path = tmp_path / "ds1_lock.fingerprint_lock"
    # Write a fingerprint lock file
    atlas = Atlas()
    fp1 = atlas.extract(dataset1)
    FingerprintLock.write_lock(
        contract_path=tmp_path / "contract.yaml",
        fingerprint=fp1,
        report_passed=True,
        lock_path=lock_path,
    )

    result = runner.invoke(main, ["diff", str(lock_path), str(fp2_path)])
    assert result.exit_code == 1
    assert "IMGSHAPE DATASET DIFF REPORT" in result.output

    result_lock_ok = runner.invoke(
        main, ["diff", str(lock_path), str(fp2_path), "--threshold", "1.5"]
    )
    assert result_lock_ok.exit_code == 0
