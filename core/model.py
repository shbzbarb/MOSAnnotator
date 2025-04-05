"""
This module handles CNN-based scoring using a pretrained ResNet50 model for feature extraction
and implements a KNN-based MOS predictor with optional distance weighting. It also provides
helper functions to pick representative images either via k-means or via DBSCAN clustering.

Classes:
    1. ResNetFeatureExtractor:
       - Loads a pretrained ResNet50 model (with the final classification layer removed)
         and sets up an image preprocessing pipeline
       - Provides a method to extract a feature vector from a given image
    2. KNNMOSPredictor:
       - Implements a simple k‑Nearest Neighbors regression in the feature space
       - Given feature vectors and MOS scores from annotated images, it predicts the MOS
         for a new image by averaging or distance-weighting the MOS scores of its k nearest neighbors
    3. pick_representative_images (k-means):
       - Uses k-means clustering on the ResNet feature vectors to select a subset of images
         that best represent the dataset (one representative per cluster)
    4. pick_representative_images_dbscan (DBSCAN):
       - Uses DBSCAN clustering on the ResNet feature vectors to select a subset of images
         that best represent the dataset, ignoring outliers (label = -1).
"""

import logging
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(name)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

from core.image_loader import load_image


class ResNetFeatureExtractor:
    """
    Loads a pretrained ResNet50 model and provides methods for image feature extraction

    Attributes:
        device (str): The device on which to run the model ('cpu' or 'cuda').
        model (nn.Module): The ResNet50 model with its final classification layer removed
        transform (transforms.Compose): Preprocessing pipeline for input images
    """
    def __init__(self, device: str = 'cpu'):
        """
        Initialize the feature extractor by loading ResNet50 and setting up preprocessing
        
        Parameters:
            device (str): The target device for model inference ('cpu' or 'cuda')
        """
        self.device = device
        
        logger.debug("Loading pretrained ResNet50 model.")
        resnet = models.resnet50(pretrained=True)
        
        #removing the final fully connected layer
        self.model = nn.Sequential(*(list(resnet.children())[:-1]))
        self.model.to(self.device)
        self.model.eval()
        logger.info("ResNet50 model loaded and set to evaluation mode.")

        #Transformation pipeline for consistent input
        self.transform = transforms.Compose([
            transforms.Resize(256),            # Resize the shortest side to 256
            transforms.CenterCrop(224),        # Center crop to 224x224
            transforms.ToTensor(),             # Convert the image to a PyTorch tensor
            transforms.Normalize(              # Normalize using ImageNet's mean and std
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_features(self, image: Image.Image) -> np.ndarray:
        """
        Extract a feature vector from an image. The image is preprocessed using the
        defined transformation pipeline, then passed through the model. The resulting
        feature tensor is flattened into a 1D array
        
        Parameters:
            image (PIL.Image.Image): The input image (RGB or other; will be converted to RGB)

        Returns:
            np.ndarray: A 1D numpy array containing the extracted features
        """
        logger.debug("Extracting features from an image using ResNet50")
        if image.mode != 'RGB':
            logger.debug("Converting image to RGB.")
            image = image.convert("RGB")

        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(input_tensor)  # shape [1,2048,1,1]

        #flatten the input to 1D
        feature_vector = features.view(features.size(0), -1).cpu().numpy().flatten()
        logger.debug(f"Feature extraction complete. Shape={feature_vector.shape}")
        return feature_vector


class KNNMOSPredictor:
    """
    kNN regressor in feature space for predicting MOS

    Attributes:
        annotated_features (List[np.ndarray]): Feature vectors for annotated images
        annotated_scores (List[float]): Corresponding MOS scores in [0,1]
        k (int): The number of nearest neighbors to consider
        weighting (str): 'uniform' or 'distance' weighting
    """
    def __init__(self,
                 annotated_features: List[np.ndarray],
                 annotated_scores: List[float],
                 k: int = 3,
                 weighting: str = 'uniform'):
        """
        Initialize the KNNMOSPredictor with annotated data

        Parameters:
            annotated_features (List[np.ndarray]): List of feature vectors
            annotated_scores (List[float]): List of MOS scores for each vector
            k (int): Number of nearest neighbors (default=3)
            weighting (str): 'uniform' or 'distance' weighting mode
        
        Raises:
            ValueError: If len(annotated_features) != len(annotated_scores)
        """
        if len(annotated_features) != len(annotated_scores):
            logger.error("Mismatch in lengths: features=%d, scores=%d",
                         len(annotated_features), len(annotated_scores))
            raise ValueError("annotated_features and annotated_scores must have the same length")

        self.annotated_features = annotated_features
        self.annotated_scores = annotated_scores
        self.k = k
        self.weighting = weighting.lower()
        if self.weighting not in ('uniform','distance'):
            logger.warning(f"Unknown weighting '{self.weighting}', defaulting to 'uniform'")
            self.weighting = 'uniform'

        logger.info(f"KNNMOSPredictor: {len(annotated_features)} annotated samples, k={k}, weighting='{self.weighting}'.")

    def predict(self, feature: np.ndarray) -> float:
        """
        Predict the MOS for a new feature vector using kNN in feature space

        Steps:
          1) compute Euclidean distance to each annotated feature
          2) find the k nearest neighbors
          3) if weighting='distance', do distance-based weighting; else uniform
          4) return the average/weighted MOS score

        Returns a float in [0,1]
        """
        logger.debug("Predicting MOS with kNN in feature space")
        if not self.annotated_features:
            logger.error("No annotated features available for prediction")
            raise ValueError("No annotated features available")

        feats = np.array(self.annotated_features)
        dists = np.linalg.norm(feats - feature, axis=1)  #Euclidean distances

        #sorting to find the k-nearest
        k = min(self.k, len(dists))
        nearest_indices = dists.argsort()[:k]
        nearest_scores = np.array([self.annotated_scores[i] for i in nearest_indices])
        nearest_dists = dists[nearest_indices]

        if self.weighting == 'distance':
            logger.debug("Using distance-based weighting for kNN.")
            inv_dist = 1.0 / (nearest_dists + 1e-8)
            weights = inv_dist / inv_dist.sum()
            mos = float((weights * nearest_scores).sum())
        else:
            # uniform weighting
            logger.debug("Using uniform weighting for kNN.")
            mos = float(nearest_scores.mean())

        logger.debug(f"Predicted MOS={mos:.4f} using k={k} neighbors.")
        return mos

#K-Means
def pick_representative_images(image_paths: List[Path], n_clusters: int=5, device='cpu') -> List[Path]:
    """
    Perform k-means clustering on ResNet50 feature vectors to pick one 'representative' image
    per cluster. This helps identify a subset of images that best represent the overall dataset

    Parameters:
        image_paths (List[Path]): List of image file paths
        n_clusters (int): The number of clusters to form (Default=5)
        device (str): 'cpu' or 'cuda' for the feature extractor

    Returns:
        List[Path]: A subset of paths, one per cluster, closest to each cluster centroid
    """
    logger.info(f"Picking representative images from {len(image_paths)} images, n_clusters={n_clusters}.")
    if not image_paths:
        return []

    extractor = ResNetFeatureExtractor(device=device)

    feats = []
    for p in image_paths:
        img = load_image(str(p))
        vec = extractor.extract_features(img)
        feats.append(vec)
    feats = np.array(feats)

    if len(feats) < n_clusters:
        logger.warning(f"Number of images < n_clusters={n_clusters}. Returning all as 'representative'.")
        return image_paths

    logger.debug("Clustering feature vectors with KMeans")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(feats)
    centroids = kmeans.cluster_centers_

    representatives = []
    for c in range(n_clusters):
        indices = np.where(kmeans.labels_ == c)[0]
        if len(indices) == 0:
            continue
        best_idx = indices[0]
        best_dist = np.linalg.norm(feats[best_idx] - centroids[c])
        for i in indices[1:]:
            dist = np.linalg.norm(feats[i] - centroids[c])
            if dist < best_dist:
                best_idx = i
                best_dist = dist
        representatives.append(image_paths[best_idx])

    logger.info(f"Found {len(representatives)} representative images (k-means)")
    return representatives

#DBSCAN
def pick_representative_images_dbscan(
    image_paths: List[Path],
    eps: float = 0.5,
    min_samples: int = 5,
    device: str = 'cpu'
) -> List[Path]:
    """
    Use DBSCAN to cluster ResNet50 feature vectors, picking one representative per cluster
    and ignoring outliers (label=-1).

    Steps:
      1) Extract features (ResNet) for each image path.
      2) Run DBSCAN(eps=..., min_samples=...).
      3) For each cluster label >= 0, compute a centroid and pick the single closest image to it.
      4) Return these images' paths as the "representatives".

    Parameters:
        image_paths (List[Path]): The list of image file paths.
        eps (float): DBSCAN 'eps' parameter. Smaller => more clusters or outliers.
        min_samples (int): DBSCAN 'min_samples' parameter.
        device (str): 'cpu' or 'cuda' for feature extraction.

    Returns:
        List[Path]: One representative image path per cluster. Outliers are skipped.
    """
    logger.info(f"Picking DBSCAN representative images from {len(image_paths)} images. eps={eps}, min_samples={min_samples}")
    if not image_paths:
        logger.warning("No image paths provided to DBSCAN rep. selection.")
        return []

    extractor = ResNetFeatureExtractor(device=device)

    feats = []
    for p in image_paths:
        img = load_image(str(p))
        vec = extractor.extract_features(img)
        feats.append(vec)
    feats = np.array(feats)

    if len(feats) == 0:
        logger.warning("No features extracted => returning empty")
        return []

    logger.debug("Clustering feature vectors with DBSCAN")
    
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(feats)
    labels = db.labels_

    unique_labels = set(labels) - {-1}
    logger.info(f"Found {len(unique_labels)} cluster(s); ignoring outliers label=-1")

    representatives = []
    for cluster_label in unique_labels:
        cinds = np.where(labels == cluster_label)[0]
        if len(cinds)==0:
            continue
        
        #computing the "centroid" as mean
        cluster_feats = feats[cinds]
        centroid = np.mean(cluster_feats, axis=0)

        #finding the image whose feature is closest to centroid
        best_idx = cinds[0]
        best_dist = np.linalg.norm(feats[best_idx] - centroid)
        for i in cinds[1:]:
            dist = np.linalg.norm(feats[i] - centroid)
            if dist < best_dist:
                best_idx = i
                best_dist = dist
        representatives.append(image_paths[best_idx])

    logger.info(f"Returning {len(representatives)} representative images (DBSCAN)")
    return representatives