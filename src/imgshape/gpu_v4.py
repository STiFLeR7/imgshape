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
        
        # Simple implementation for now, will add kernels in next task
        batch_t = [torch.from_numpy(img).permute(2, 0, 1).float().to(self.device) for img in self.queue]
        # Placeholder for actual kernel results
        results = [{"batch_idx": i} for i in range(len(self.queue))]
        self.queue = []
        return results
