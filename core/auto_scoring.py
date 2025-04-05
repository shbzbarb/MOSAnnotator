"""
This module provides functionality to automatically assign MOS (Mean Opinion Score)
to unannotated images based on similarity to a manually annotated subset using different
similarity metrics. It supports four modes:
    - 'lpips': Only the LPIPS metric is used
    - 'ssim': Only the SSIM metric is used
    - 'msssim': MS-SSIM metric (multi-scale SSIM)
    - 'combined': A weighted combination of LPIPS->similarity, SSIM, and MS-SSIM,
      using user-defined alpha, beta, and gamma (each in [0,1], sum ~ 1).

For each unannotated image, the function computes similarity scores with each annotated image,
then uses these similarities as weights to compute a weighted average of the annotated scores.
"""

import logging
from typing import List, Dict, Tuple

from PIL import Image
from core.metrics import (
    compute_lpips,
    compute_ssim,
    compute_ms_ssim,
    compute_combined_metric
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(name)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def auto_score_images(
    annotated_images: List[Tuple[str, Image.Image, float]],
    unannotated_images: List[Tuple[str, Image.Image]],
    mode: str = 'combined',
    net: str = 'alex',
    device: str = 'cpu',
    alpha: float = 1/3,
    beta: float = 1/3,
    gamma: float = 1/3
) -> Dict[str, float]:
    """
    Auto-score unannotated images using a chosen similarity metric.

    Parameters:
        annotated_images (List[Tuple[str, Image.Image, float]]):
            A list of tuples, each containing:
                - image name (str)
                - PIL Image object (manually annotated image)
                - manual MOS score (float in [0, 1])

        unannotated_images (List[Tuple[str, Image.Image]]):
            A list of tuples, each containing:
                - image name (str)
                - PIL Image object (image to be auto-scored)

        mode (str, optional):
            The similarity mode to use:
                'lpips'   -> LPIPS metric (distance -> similarity)
                'ssim'    -> SSIM metric
                'msssim'  -> MS-SSIM metric
                'combined'-> Weighted triple combination of (LPIPS->sim, SSIM, MS-SSIM)
            Defaults to 'combined'.

        net (str, optional):
            The backbone network for LPIPS (if applicable). Defaults to 'alex'.

        device (str, optional):
            The device for running metric computations ('cpu' or 'cuda'). Defaults to 'cpu'.

        alpha (float, optional):
            The weight for LPIPS->similarity in 'combined' mode. Default 1/3.
        beta (float, optional):
            The weight for SSIM in 'combined' mode. Default 1/3.
        gamma (float, optional):
            The weight for MS-SSIM in 'combined' mode. Default 1/3.

    Returns:
        Dict[str, float]:
            A dictionary mapping each unannotated image name to its predicted MOS
            (a float in [0,1]) based on the chosen metric.

    Raises:
        ValueError: If an unsupported mode is provided or if alpha+beta+gamma != 1 in combined mode.
    """
    logger.info("Starting automatic scoring with mode=%s, alpha=%.2f, beta=%.2f, gamma=%.2f",
                mode, alpha, beta, gamma)

    mode = mode.lower()
    if mode not in ('lpips', 'ssim', 'msssim', 'combined'):
        logger.error("Unsupported mode provided: %s", mode)
        raise ValueError("Unsupported mode. Choose 'lpips','ssim','msssim','combined'.")

    predictions = {}

    for unannotated_name, unannotated_img in unannotated_images:
        weighted_score_sum = 0.0
        similarity_sum = 0.0

        logger.debug("Scoring unannotated image: %s", unannotated_name)
        for annotated_name, annotated_img, annotated_score in annotated_images:
            if mode == 'lpips':
                lpips_distance = compute_lpips(unannotated_img, annotated_img, net=net, device=device)
                similarity = 1 / (1 + lpips_distance)
                logger.debug("LPIPS similarity '%s' vs '%s': %.4f",
                             unannotated_name, annotated_name, similarity)

            elif mode == 'ssim':
                similarity = compute_ssim(unannotated_img, annotated_img)
                logger.debug("SSIM similarity '%s' vs '%s': %.4f",
                             unannotated_name, annotated_name, similarity)

            elif mode == 'msssim':
                similarity = compute_ms_ssim(unannotated_img, annotated_img, device=device)
                logger.debug("MS-SSIM similarity '%s' vs '%s': %.4f",
                             unannotated_name, annotated_name, similarity)

            elif mode == 'combined':
                similarity = compute_combined_metric(
                    unannotated_img, annotated_img,
                    alpha=alpha, beta=beta, gamma=gamma,
                    net=net, device=device
                )
                logger.debug("Combined similarity '%s' vs '%s': %.4f",
                             unannotated_name, annotated_name, similarity)

            weighted_score_sum += similarity * annotated_score
            similarity_sum += similarity

        if similarity_sum == 0:
            logger.warning("Total similarity=0 for '%s'. Using average annotated score.", unannotated_name)
            predicted_score = sum(item[2] for item in annotated_images) / len(annotated_images)
        else:
            predicted_score = weighted_score_sum / similarity_sum

        logger.info("Predicted MOS for '%s': %.4f", unannotated_name, predicted_score)
        predictions[unannotated_name] = predicted_score

    logger.info("Automatic scoring completed for %d image(s).", len(unannotated_images))
    return predictions