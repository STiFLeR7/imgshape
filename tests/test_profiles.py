import pytest
import numpy as np
from PIL import Image
from pathlib import Path
from imgshape.fingerprint_v4 import FingerprintExtractor, PrimaryDomain, MedicalProfile, SatelliteProfile, OCRProfile

def create_dummy_dataset(tmp_path, domain="photographic", count=5):
    dataset_dir = tmp_path / domain
    dataset_dir.mkdir()
    
    for i in range(count):
        if domain == "medical":
            # Grayscale, low entropy
            img_data = np.random.randint(50, 100, (128, 128), dtype=np.uint8)
            img = Image.fromarray(img_data, mode='L')
        elif domain == "ocr":
            # Grayscale, high contrast/entropy (simulating text)
            img_data = np.random.randint(0, 255, (64, 256), dtype=np.uint8)
            img = Image.fromarray(img_data, mode='L')
        elif domain == "satellite":
            # Color, square, high entropy
            img_data = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            # Add some "clouds"
            img_data[0:50, 0:50, :] = 250
            img = Image.fromarray(img_data, mode='RGB')
        else:
            img_data = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            img = Image.fromarray(img_data, mode='RGB')
            
        img.save(dataset_dir / f"img_{i}.png")
    
    return dataset_dir

def test_medical_profile_detection(tmp_path):
    dataset_dir = create_dummy_dataset(tmp_path, "medical")
    extractor = FingerprintExtractor()
    fingerprint = extractor.extract(dataset_dir)
    
    assert fingerprint.profiles["semantic"].primary_domain == PrimaryDomain.MEDICAL
    assert "medical" in fingerprint.profiles
    assert isinstance(fingerprint.profiles["medical"], MedicalProfile)
    assert fingerprint.profiles["medical"].hu_range[0] >= 0
    assert fingerprint.profiles["medical"].slice_consistency > 0

def test_satellite_profile_detection(tmp_path):
    dataset_dir = create_dummy_dataset(tmp_path, "satellite")
    extractor = FingerprintExtractor()
    fingerprint = extractor.extract(dataset_dir)
    
    assert fingerprint.profiles["semantic"].primary_domain == PrimaryDomain.SATELLITE
    assert "satellite" in fingerprint.profiles
    assert isinstance(fingerprint.profiles["satellite"], SatelliteProfile)
    assert fingerprint.profiles["satellite"].gsd_estimate > 0
    assert fingerprint.profiles["satellite"].cloud_cover_estimate > 0

def test_ocr_profile_detection(tmp_path):
    dataset_dir = create_dummy_dataset(tmp_path, "ocr")
    extractor = FingerprintExtractor()
    fingerprint = extractor.extract(dataset_dir)
    
    assert fingerprint.profiles["semantic"].primary_domain == PrimaryDomain.OCR
    assert "ocr" in fingerprint.profiles
    assert isinstance(fingerprint.profiles["ocr"], OCRProfile)
    assert fingerprint.profiles["ocr"].text_density > 0
    assert fingerprint.profiles["ocr"].orientation_variance >= 0
