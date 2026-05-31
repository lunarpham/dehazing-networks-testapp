"""
Image I/O utilities for standalone dehazing inference.
"""

import torch
import numpy as np
from PIL import Image


def load_image(filepath):
    """
    Load an image from filepath and return as a normalized NumPy array.
    Outputs an RGB array of shape (H, W, 3) in range [0, 1].
    """
    img = Image.open(filepath).convert('RGB')
    return np.array(img, dtype=np.float32) / 255.0


def save_image(filepath, img_tensor):
    """
    Save a PyTorch tensor shape (C, H, W) or (1, C, H, W) to file.
    Assumes values are in [0, 1].
    """
    if len(img_tensor.shape) == 4:
        img_tensor = img_tensor.squeeze(0)

    img_np = img_tensor.detach().cpu().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC

    # Clip values and convert to uint8
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

    img = Image.fromarray(img_np)
    img.save(filepath)


def to_tensor(img_np):
    """Convert HWC numpy array to CHW PyTorch tensor with batch dimension."""
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
    return img_tensor.unsqueeze(0)  # Add batch dimension
