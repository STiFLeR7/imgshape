# 📊 imgshape Report
- Generated: 2025-09-22T06:25:12.887759Z
- Dataset: `assets/sample_images`

## Dataset Summary
```json
{
  "image_count": 1,
  "unique_shapes": {
    "4000x3000": 1
  },
  "most_common_shape": "4000x3000",
  "most_common_shape_count": 1,
  "avg_entropy": 7.341,
  "channels": [
    3
  ],
  "unreadable_count": 0,
  "sampled_paths_count": 1
}
```

## Representative Preprocessing
```json
{
  "user_prefs": [],
  "bias": "neutral",
  "augmentation_plan": {
    "order": [
      "RandomHorizontalFlip",
      "ColorJitter",
      "RandomResizedCrop"
    ],
    "augmentations": [
      {
        "name": "RandomHorizontalFlip",
        "params": {
          "p": 0.5
        },
        "reason": "Default safe flip for many datasets",
        "score": 0.7
      },
      {
        "name": "ColorJitter",
        "params": {
          "brightness": 0.2,
          "contrast": 0.2,
          "saturation": 0.2,
          "hue": 0.05
        },
        "reason": "Color variations",
        "score": 0.5
      },
      {
        "name": "RandomResizedCrop",
        "params": {
          "size": 224,
          "scale": [
            0.8,
            1.0
          ]
        },
        "reason": "Resize crop for big images",
        "score": 0.6
      }
    ]
  },
  "resize": {
    "size": [
      224,
      224
    ],
    "method": "bilinear",
    "suggested_model": "ResNet18 / MobileNetV2"
  },
  "normalize": {
    "mean": [
      0.485,
      0.456,
      0.406
    ],
    "std": [
      0.229,
      0.224,
      0.225
    ]
  },
  "mode": "RGB",
  "entropy": 7.341,
  "channels": 3,
  "image_count": 1
}
```

## Augmentation Plan
```json
{
  "augmentations": [
    {
      "name": "RandomHorizontalFlip",
      "params": {
        "p": 0.5
      },
      "reason": "Common orientation variance; usually safe for many datasets",
      "score": 0.7
    }
  ],
  "recommended_order": [
    "RandomHorizontalFlip"
  ],
  "seed": null
}
```