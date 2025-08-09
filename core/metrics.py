"""
This module provides functions for computing image similarity metrics:
    1. LPIPS (Learned Perceptual Image Patch Similarity)
    2. SSIM (Structural Similarity Index)
    3. MS-SSIM (Multi-Scale Structural Similarity Index)
    4. A combined metric that merges LPIPS->similarity, SSIM, and MS-SSIM.

All metric functions internally resize images to 256x256 to handle
inputs of varying dimensions and prevent dimension mismatch errors.
"""

import logging
import numpy as np
import torch
import lpips
from PIL import Image
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim_metric
from pytorch_msssim import ms_ssim

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(name)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Global variable to hold the LPIPS model, so it's only loaded once
_lpips_model = None

def get_lpips_model(net: str = 'alex', device: str = 'cpu'):
    """
    Lazily loads and returns a global LPIPS model instance
    This prevents reloading the model from disk on every function call
    """
    global _lpips_model
    if _lpips_model is None:
        logger.debug(f"Loading LPIPS model with backbone '{net}' on device '{device}'")
        _lpips_model = lpips.LPIPS(net=net)
        _lpips_model.to(device)
    return _lpips_model


def pil_to_tensor(image: Image.Image, device: str = 'cpu') -> torch.Tensor:
    """
    Converts a PIL Image to a PyTorch Tensor for metric calculation

    This function performs several key steps:
    1.  Ensures the image is in RGB format
    2.  Resizes the image to a fixed 256x256 dimension to prevent size mismatch errors
    3.  Converts the image to a Tensor with values in the [0, 1] range
    4.  Normalizes the tensor values to the [-1, 1] range, which is expected by the LPIPS model
    """
    if image.mode != 'RGB':
        image = image.convert("RGB")

    # Define the transformation pipeline
    transform_pipeline = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to a fixed size
        transforms.ToTensor()          # Convert to a Tensor (values 0-1)
    ])
    
    # Apply the pipeline and add a batch dimension
    tensor = transform_pipeline(image).unsqueeze(0)
    
    # Normalize tensor from [0, 1] to [-1, 1] for LPIPS
    tensor = tensor * 2 - 1
    return tensor.to(device)


def compute_lpips(img1: Image.Image, img2: Image.Image,
                  net: str = 'alex', device: str = 'cpu') -> float:
    """
    Computes the LPIPS distance between two images
    Images are resized to 256x256 internally via the pil_to_tensor helper
    """
    logger.debug("Computing LPIPS metric.")
    model = get_lpips_model(net=net, device=device)
    
    # Convert images to resized, normalized tensors.
    tensor1 = pil_to_tensor(img1, device)
    tensor2 = pil_to_tensor(img2, device)

    # Calculate LPIPS distance.
    with torch.no_grad():
        lpips_val = model.forward(tensor1, tensor2).item()
        
    return float(lpips_val)


def compute_ssim(img1: Image.Image, img2: Image.Image) -> float:
    """
    Computes the Structural Similarity Index (SSIM)
    Images are resized to 256x256 and converted to grayscale internally
    """
    logger.debug("Computing SSIM metric.")
    
    # Define the target size for consistency with other metrics
    size = (256, 256)

    # Resize images and then convert to grayscale NumPy arrays
    gray1 = np.array(img1.resize(size).convert("L"))
    gray2 = np.array(img2.resize(size).convert("L"))

    # The data range for an 8-bit grayscale image is 0-255
    data_range = 255.0
    
    # Calculate SSIM score
    ssim_score = ssim_metric(gray1, gray2, data_range=data_range)
    return float(ssim_score)


def compute_ms_ssim(img1: Image.Image, img2: Image.Image, device: str = 'cpu') -> float:
    """
    Computes the Multi-Scale Structural Similarity Index (MS-SSIM)
    Images are resized to 256x256 internally
    """
    logger.debug("Computing MS-SSIM metric.")
    
    # Use the main tensor helper to get resized tensors in LPIPS format [-1, 1]
    tensor1_lpips = pil_to_tensor(img1, device)
    tensor2_lpips = pil_to_tensor(img2, device)

    # MS-SSIM expects tensors in the [0, 1] range, so we rescale them
    tensor1 = (tensor1_lpips + 1) / 2
    tensor2 = (tensor2_lpips + 1) / 2
    
    # Calculate MS-SSIM score
    with torch.no_grad():
        msssim_val = ms_ssim(tensor1, tensor2, data_range=1.0, size_average=True).item()
        
    return float(msssim_val)


def compute_combined_metric(
    img1: Image.Image,
    img2: Image.Image,
    alpha: float = 1/3,
    beta: float = 1/3,
    gamma: float = 1/3,
    net: str = 'alex',
    device: str = 'cpu'
) -> float:
    """
    Computes a weighted combination of LPIPS, SSIM, and MS-SSIM
    All underlying metrics handle resizing internally
    """
    logger.debug(f"Computing combined metric with alpha={alpha:.3f}, beta={beta:.3f}, gamma={gamma:.3f}")

    # 1. LPIPS distance -> similarity (lower LPIPS is better)
    lpips_score = compute_lpips(img1, img2, net=net, device=device)
    sim_lpips = 1.0 / (1.0 + lpips_score)

    # 2. SSIM similarity
    ssim_score = compute_ssim(img1, img2)

    # 3. MS-SSIM similarity
    ms_ssim_score = compute_ms_ssim(img1, img2, device=device)

    # 4. Compute the final weighted average
    combined_score = alpha * sim_lpips + beta * ssim_score + gamma * ms_ssim_score
    
    return float(combined_score)
