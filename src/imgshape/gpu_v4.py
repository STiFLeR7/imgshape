import torch
import numpy as np
from typing import List, Optional, Dict

class BatchedGPUHandler:
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.queue = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def add_image(self, img: np.ndarray) -> Optional[List[Dict]]:
        self.queue.append(img)
        if len(self.queue) >= self.batch_size:
            return self.flush()
        return None

    def flush(self) -> List[Dict]:
        if not self.queue:
            return []
        
        # 1. Stack into batch tensor [B, C, H, W]
        batch_t = torch.stack([
            torch.from_numpy(img).permute(2, 0, 1).float().to(self.device) 
            for img in self.queue
        ])
        
        # 2. Batched Entropy & Blur
        results = []
        for i in range(len(self.queue)):
            img_t = batch_t[i]
            # Shannon entropy on GPU
            hist = torch.histc(img_t, bins=256, min=0, max=255)
            probs = hist / hist.sum()
            probs = probs[probs > 0]
            entropy = -torch.sum(probs * torch.log2(probs)).item()
            
            # Laplacian Blur
            laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=self.device).float().view(1, 1, 3, 3)
            # Grayscale conversion
            gray = 0.2989 * img_t[0] + 0.5870 * img_t[1] + 0.1140 * img_t[2]
            gray = gray.unsqueeze(0).unsqueeze(0)
            edge_map = torch.nn.functional.conv2d(gray, laplacian_kernel)
            blur_score = torch.var(edge_map).item()
            
            results.append({
                "entropy": float(entropy),
                "blur": float(blur_score)
            })
            
        self.queue = []
        return results
