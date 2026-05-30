import logging
import numpy as np

logger = logging.getLogger("imgshape.atlas.gpu")

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


class GPUHandler:
    """
    Utility to handle GPU availability and detection.
    """

    @staticmethod
    def is_cuda_available() -> bool:
        if not TORCH_AVAILABLE:
            return False
        return torch.cuda.is_available()

    @staticmethod
    def get_device() -> str:
        return "cuda" if GPUHandler.is_cuda_available() else "cpu"


class BatchedGPUHandler:
    """
    Vectorized batch processor using PyTorch, with clean fallback if torch is missing.
    """

    def __init__(self, batch_size: int = 32, use_gpu: bool = True):
        self.batch_size = batch_size
        self.queue = []
        self.use_gpu = use_gpu and TORCH_AVAILABLE

        if TORCH_AVAILABLE:
            self.device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
            self.laplacian_kernel = (
                torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=self.device)
                .float()
                .view(1, 1, 3, 3)
            )
        else:
            self.device = "cpu"
            self.laplacian_kernel = None

    def add_image(self, img: np.ndarray) -> list or None:
        self.queue.append(img)
        if len(self.queue) >= self.batch_size:
            return self.flush()
        return None

    def flush(self) -> list:
        if not self.queue:
            return []

        try:
            if TORCH_AVAILABLE:
                # Check if all images in the queue have the same shape
                shapes = [img.shape for img in self.queue]
                uniform_shape = len(set(shapes)) == 1

                if uniform_shape:
                    # Stack on CPU first, then move to GPU/device for better throughput
                    batch_cpu = torch.stack(
                        [
                            torch.from_numpy(img).permute(2, 0, 1).float()
                            for img in self.queue
                        ]
                    )
                    batch_t = batch_cpu.to(self.device)

                    # Vectorized Grayscale Conversion [B, 1, H, W]
                    weights = torch.tensor(
                        [0.2989, 0.5870, 0.1140], device=self.device
                    ).view(1, 3, 1, 1)
                    batch_gray = (batch_t * weights).sum(dim=1, keepdim=True)

                    # Vectorized Laplacian Blur [B, 1, H, W]
                    edge_maps = torch.nn.functional.conv2d(
                        batch_gray, self.laplacian_kernel, padding=1
                    )
                    blur_scores = torch.var(edge_maps.view(len(self.queue), -1), dim=1)

                    results = []
                    for i in range(len(self.queue)):
                        img_t = batch_t[i]
                        # Entropy calculation
                        hist = torch.histc(img_t, bins=256, min=0, max=255)
                        probs = hist / hist.sum()
                        probs = probs[probs > 0]
                        entropy = -torch.sum(probs * torch.log2(probs)).item()

                        results.append(
                            {
                                "entropy": float(entropy),
                                "blur": float(blur_scores[i].item()),
                            }
                        )
                else:
                    # Heterogeneous batch: process sequentially on target device
                    results = []
                    weights = torch.tensor(
                        [0.2989, 0.5870, 0.1140], device=self.device
                    ).view(3, 1, 1)

                    for img in self.queue:
                        img_t = (
                            torch.from_numpy(img)
                            .permute(2, 0, 1)
                            .float()
                            .to(self.device)
                        )

                        # Grayscale
                        gray = (img_t * weights).sum(dim=0, keepdim=True).unsqueeze(0)

                        # Blur
                        edge_map = torch.nn.functional.conv2d(
                            gray, self.laplacian_kernel, padding=1
                        )
                        blur_score = torch.var(edge_map).item()

                        # Entropy
                        hist = torch.histc(img_t, bins=256, min=0, max=255)
                        probs = hist / hist.sum()
                        probs = probs[probs > 0]
                        entropy = -torch.sum(probs * torch.log2(probs)).item()

                        results.append(
                            {"entropy": float(entropy), "blur": float(blur_score)}
                        )
            else:
                # NumPy / CPU Fallback path if PyTorch is not available
                results = []
                for img in self.queue:
                    # Entropy calculation
                    _, counts = np.unique(img, return_counts=True)
                    probs = counts / counts.sum()
                    entropy = -np.sum(probs * np.log2(probs))

                    # Blur calculation
                    if len(img.shape) == 3:
                        weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
                        gray = np.dot(img[..., :3], weights).astype(np.float32)
                    else:
                        gray = img.astype(np.float32)

                    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
                    from scipy.signal import convolve2d

                    laplacian = convolve2d(gray, kernel, mode="same")
                    blur_score = np.var(laplacian)

                    results.append(
                        {"entropy": float(entropy), "blur": float(blur_score)}
                    )
        finally:
            self.queue = []

        return results
