"""
This module provides functions for computing image similarity metrics:
    1. LPIPS (Learned Perceptual Image Patch Similarity)
    2. SSIM (Structural Similarity Index)
    3. MS-SSIM (Multi-Scale Structural Similarity Index)
    4. A combined metric that merges LPIPS->similarity, SSIM, MS-SSIM,
       weighted by alpha,beta,gamma in [0,1]. By default, alpha=beta=gamma=1/3,
       and alpha+beta+gamma should equal 1.

Functions:
    - compute_lpips(img1, img2, net='alex', device='cpu')
    - compute_ssim(img1, img2)
    - compute_ms_ssim(img1, img2, device='cpu')
    - compute_combined_metric(img1, img2, alpha, beta, gamma, net='alex', device='cpu')

The code also includes logic to handle smaller images for MS-SSIM
by adapting the window size and number of scales automatically
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

_lpips_model = None

def get_lpips_model(net: str = 'alex', device: str = 'cpu'):
    """
    Lazily load and return a global LPIPS model instance

    Parameters:
        net (str): The backbone network for LPIPS ('alex' or 'vgg'). Default 'alex'
        device (str): 'cpu' or 'cuda'
    """
    global _lpips_model
    if _lpips_model is None:
        logger.debug(f"Loading LPIPS model with backbone '{net}' on device '{device}'")
        _lpips_model = lpips.LPIPS(net=net)
        _lpips_model.to(device)
    return _lpips_model


def pil_to_tensor(image: Image.Image, device: str = 'cpu') -> torch.Tensor:
    """
    Convert a PIL Image to a torch.Tensor in the range [-1, 1] for LPIPS or other PyTorch-based metrics
    If image is not RGB, convert to the image to RGB

    Returns a tensor of shape [1, 3, H, W]
    """
    if image.mode != 'RGB':
        logger.debug("Converting image to RGB")
        image = image.convert("RGB")

    tensor = transforms.ToTensor()(image).unsqueeze(0) #[1,3,H,W] in [0,1]
    
    #For LPIPS ,we scale to [-1,1]
    tensor = tensor * 2 - 1
    return tensor.to(device)


def compute_lpips(img1: Image.Image, img2: Image.Image,
                  net: str = 'alex', device: str = 'cpu') -> float:
    """
    Compute LPIPS distance using a specified backbone (alex or vgg)
    Returns a distance in [0, ∞), where 0 indicates identical images
    """
    logger.debug("Computing LPIPS metric between two images.")
    model = get_lpips_model(net=net, device=device)

    tensor1 = pil_to_tensor(img1, device)
    tensor2 = pil_to_tensor(img2, device)

    with torch.no_grad():
        lpips_val = model.forward(tensor1, tensor2).item()
    logger.debug(f"LPIPS score computed: {lpips_val:.4f}")
    return float(lpips_val)


def compute_ssim(img1: Image.Image, img2: Image.Image) -> float:
    """
    Compute the Structural Similarity Index (SSIM) in [0, 1]
    For constant images, returns 1 if they're identical, 0 otherwise
    """
    logger.debug("Computing SSIM metric between two images")
    gray1 = np.array(img1.convert("L"))
    gray2 = np.array(img2.convert("L"))

    data_range = gray1.max() - gray1.min()
    if data_range == 0:
        #constant image
        if np.array_equal(gray1, gray2):
            ssim_score = 1.0
        else:
            ssim_score = 0.0
        logger.debug(f"Images are constant; returning SSIM={ssim_score}")
        return ssim_score

    ssim_score = ssim_metric(gray1, gray2, data_range=data_range)
    logger.debug(f"SSIM score computed: {ssim_score:.4f}")
    return float(ssim_score)


def compute_ms_ssim(img1: Image.Image, img2: Image.Image, device: str = 'cpu') -> float:
    """
    Compute MS-SSIM using pytorch_msssim, adapting the window size and
    number of scales for smaller images to avoid assertion errors

    Tiers:
      1. Very small images (min_dim < 80): 2-scale, win_size=3
      2. Small images (80 <= min_dim < 160): 3-scale, win_size=5
      3. Large images (min_dim >= 160): default ms-ssim

    Returns [0,1]

    Parameters:
        img1 (PIL.Image.Image): The first image
        img2 (PIL.Image.Image): The second image
        device (str): 'cpu' or 'cuda'

    Returns:
        float: The MS-SSIM similarity score in [0,1]
    """
    logger.debug("Computing MS-SSIM metric with a multi-tier approach for image size.")
    
    if img1.mode != 'RGB':
        img1 = img1.convert("RGB")
    if img2.mode != 'RGB':
        img2 = img2.convert("RGB")

    tform = transforms.ToTensor()
    tensor1 = tform(img1).unsqueeze(0).to(device)  #[1,3,H,W] in [0,1]
    tensor2 = tform(img2).unsqueeze(0).to(device)
    
    _, _, H, W = tensor1.shape
    min_dim = min(H, W)

    #Category 1: Very small images
    if min_dim < 80:
        #For example, win_size=3, 2 scales
        #E.g., weights=[0.5, 0.5]
        logger.debug(f"min_dim={min_dim} < 80 => 2-scale MS-SSIM, win_size=3")
        with torch.no_grad():
            msssim_val = ms_ssim(tensor1, tensor2,
                                 data_range=1.0,
                                 size_average=True,
                                 win_size=3,
                                 weights=[0.5, 0.5]).item()
            
    #Category 2: Medium Images        
    elif min_dim < 160:
        #For example, win_size=5, 3 scales
        logger.debug(f"min_dim={min_dim} < 160 => 3-scale MS-SSIM, win_size=5")
        with torch.no_grad():
            msssim_val = ms_ssim(tensor1, tensor2,
                                 data_range=1.0,
                                 size_average=True,
                                 win_size=5,
                                 weights=[0.3, 0.3, 0.4]).item()
            
    else:
        #Category 3: Large => default
        logger.debug(f"min_dim={min_dim} >= 160 => default 5-scale MS-SSIM")
        with torch.no_grad():
            msssim_val = ms_ssim(tensor1, tensor2, data_range=1.0).item()

    logger.debug(f"MS-SSIM score computed: {msssim_val:.4f}")
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
    Computes a weighted combined similarity metric based on LPIPS->similarity, SSIM, and MS-SSIM

    combined_score = alpha*(1/(1+lpips_score)) + beta*(ssim_score) + gamma*(ms_ssim_score)

    By default alpha=beta=gamma=1/3, but the user can override them in the pipeline

    Returns a final similarity in [0,1]
    """
    logger.debug(f"Computing combined metric with alpha={alpha:.3f}, beta={beta:.3f}, gamma={gamma:.3f}")

    #1.LPIPS distance -> similarity
    lpips_score = compute_lpips(img1, img2, net=net, device=device)
    sim_lpips = 1.0 / (1.0 + lpips_score)

    #2. SSIM
    ssim_score = compute_ssim(img1, img2)

    #3. MS-SSIM
    ms_ssim_score = compute_ms_ssim(img1, img2, device=device)

    #Weighted sum
    combined_score = alpha * sim_lpips + beta * ssim_score + gamma * ms_ssim_score

    logger.debug(f"Weighted combined => LPIPS sim={sim_lpips:.4f}, SSIM={ssim_score:.4f}, "
                 f"MS-SSIM={ms_ssim_score:.4f}, final={combined_score:.4f}")
    return float(combined_score)