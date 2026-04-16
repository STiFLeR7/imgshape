import torch
import numpy as np
from typing import List, Optional, Dict
class BatchedGPUHandler:
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.queue = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.laplacian_kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=self.device
        ).float().view(1, 1, 3, 3)

    def add_image(self, img: np.ndarray) -> Optional[List[Dict]]:
        self.queue.append(img)
        if len(self.queue) >= self.batch_size:
            return self.flush()
        return None

    def flush(self) -> List[Dict]:
        if not self.queue:
            return []

        # 1. Stack on CPU first, then move to GPU for better throughput
        # [B, C, H, W]
        try:
            batch_cpu = torch.stack([
                torch.from_numpy(img).permute(2, 0, 1).float() 
                for img in self.queue
            ])
            batch_t = batch_cpu.to(self.device)

            # 2. Vectorized Grayscale Conversion [B, 1, H, W]
            # Weights: 0.2989, 0.5870, 0.1140
            weights = torch.tensor([0.2989, 0.5870, 0.1140], device=self.device).view(1, 3, 1, 1)
            batch_gray = (batch_t * weights).sum(dim=1, keepdim=True)

            # 3. Vectorized Laplacian Blur [B, 1, H-2, W-2]
            edge_maps = torch.nn.functional.conv2d(batch_gray, self.laplacian_kernel)
            # Variance per image in batch
            # Reshape to [B, -1] then compute var across dim 1
            blur_scores = torch.var(edge_maps.view(len(self.queue), -1), dim=1)

            # 4. Entropy (torch.histc isn't fully vectorized for batches easily, loop for now)
            results = []
            for i in range(len(self.queue)):
                img_t = batch_t[i]
                hist = torch.histc(img_t, bins=256, min=0, max=255)
                probs = hist / hist.sum()
                probs = probs[probs > 0]
                entropy = -torch.sum(probs * torch.log2(probs)).item()

                results.append({
                    "entropy": float(entropy),
                    "blur": float(blur_scores[i].item())
                })
        finally:
            self.queue = []

        return results
