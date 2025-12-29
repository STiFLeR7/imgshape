"""
Generated data transforms from imgshape v4.0.0 (Atlas)
This file is auto-generated and deterministic.
"""

import torch
from torchvision import transforms


def get_train_transform():
    """Training data transform pipeline"""
    return transforms.Compose([
        transforms.Resize((2012, 2512)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform():
    """Validation data transform pipeline"""
    return transforms.Compose([
        transforms.Resize((2012, 2512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
