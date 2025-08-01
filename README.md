﻿# imgshape

Resize image folders fast — optimized for machine learning.(v 0.1.2)

## ✅ Features
- Resize all images in a folder
- Keep aspect ratio with padding
- Format conversion (jpg/png)
- Preserve folder structure
- Keep original images (optional)

## 🚀 Installation
```bash
pip install imgshape
```

## 💻 CLI Usage
```bash
imgshape ./images --size 224 --format jpg --save-dir ./resized --keep-structure --keep-original
```

## 🧠 Python Usage
```python
from imgshape import batch_resize

batch_resize("./images", size="224x224", fmt="png", save_dir="./out", keep_structure=True, keep_original=True)
```

## 📜 License
MIT
