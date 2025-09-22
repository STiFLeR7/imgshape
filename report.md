# 📊 imgshape Report

- Generated: 2025-09-22T07:32:47.258916Z
- Dataset: `C:\Users\hilla\AppData\Local\Temp\pytest-of-hilla\pytest-6\test_generate_markdown0\report.md`

## Dataset Summary
| Field | Value |
|---|---|
| `image_count` | 0 |
| `unique_shapes` | {} |
| `notes` | "fallback" |

## Representative Preprocessing
<details><summary>Show JSON</summary>

```json
{
  "error": "fallback",
  "message": "Could not compute recommendation; returning safe defaults.",
  "user_prefs": [],
  "bias": "neutral",
  "augmentation_plan": {
    "order": [
      "RandomHorizontalFlip"
    ],
    "augmentations": [
      {
        "name": "RandomHorizontalFlip",
        "params": {
          "p": 0.5
        },
        "reason": "Default conservative augmentation",
        "score": 0.4
      }
    ]
  },
  "resize": {
    "size": [
      224,
      224
    ],
    "method": "bilinear",
    "suggested_model": "ResNet/MobileNet"
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
  "mode": "RGB"
}
```
</details>

## Augmentation Plan
<details><summary>Show augmentation plan</summary>

```json
{
  "order": [
    "RandomHorizontalFlip"
  ],
  "augmentations": [
    {
      "name": "RandomHorizontalFlip",
      "params": {
        "p": 0.5
      },
      "reason": "Default conservative augmentation",
      "score": 0.4
    }
  ]
}
```
</details>
