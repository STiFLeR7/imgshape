try:
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    models = None
    transforms = None
    TORCH_AVAILABLE = False

import numpy as np
from PIL import Image

class SemanticExtractor:
    def __init__(self, model_name="squeezenet1_1"):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch and torchvision are required for SemanticExtractor. "
                "Install with 'pip install \"imgshape[torch]\"'."
            )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1 is the modern way
        self.model = getattr(models, model_name)(weights="DEFAULT").to(self.device).eval()
        
        if "squeezenet" in model_name:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d((1, 1)), 
                torch.nn.Flatten()
            )
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, img_array: np.ndarray) -> np.ndarray:
        if not TORCH_AVAILABLE:
            raise ImportError("SemanticExtractor requires torch and torchvision")
            
        img = Image.fromarray(img_array)
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(img_t)
        return features.squeeze().cpu().numpy()
