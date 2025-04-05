"""
This module provides functionality to load image file paths from a specified folder and 
to load individual images into memory using Pillow (PIL). It is designed for use in the 
core of the MOS Annotator app

Functions:
    - load_images(folder_path: str, allowed_extensions: set = None) -> List[Path]:
        Load and return a sorted list of image file paths from a folder
    - load_image(image_path: str) -> Image.Image:
        Open an image file and return a PIL Image object
"""

import logging
from pathlib import Path
from typing import List, Set

from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(name)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def is_image_file(file_path: Path, allowed_extensions: Set[str]) -> bool:
    """
    Check if the file at file_path has an allowed image file extension
    
    Parameters:
        file_path (Path): The path of the file to check.
        allowed_extensions (Set[str]): Set of allowed file extensions (e.g., {'.jpg', '.png'})
    
    Returns:
        bool: True if file has an allowed extension, False otherwise
    """
    result = file_path.suffix.lower() in allowed_extensions
    logger.debug(f"Checking file {file_path.name}: {'Accepted' if result else 'Rejected'}")
    return result


def load_images(folder_path: str, allowed_extensions: Set[str] = None) -> List[Path]:
    """
    Load image file paths from the specified folder
    
    The function scans the given folder, filters files based on allowed image file extensions,
    sorts them by file name, and returns a list of Path objects
    
    Parameters:
        folder_path (str): Path to the folder containing images
        allowed_extensions (Set[str], optional): Set of allowed image file extensions
            Defaults to common image extensions if not provided
    
    Returns:
        List[Path]: Sorted list of image file paths
    
    Raises:
        FileNotFoundError: If the specified folder does not exist
        ValueError: If no image files are found in the folder
    """
    if allowed_extensions is None:
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}

    folder = Path(folder_path)
    logger.debug(f"Attempting to load images from folder: {folder.resolve()}")

    if not folder.exists():
        logger.error(f"Folder does not exist: {folder}")
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist")

    #filter the files with mentioned file extensions
    image_files = [
        file for file in folder.iterdir()
        if file.is_file() and is_image_file(file, allowed_extensions)
    ]

    if not image_files:
        logger.error(f"No image files found in {folder} with extensions {allowed_extensions}")
        raise ValueError(f"No image files found in folder '{folder_path}' with allowed extensions: {allowed_extensions}")

    #sorting the images by file name for consistency
    image_files.sort(key=lambda file: file.name)
    logger.info(f"Found {len(image_files)} image(s) in folder '{folder_path}'")
    return image_files


def load_image(image_path: str) -> Image.Image:
    """
    Load an image from a file path using Pillow (PIL)
    
    Parameters:
        image_path (str): The path to the image file
    
    Returns:
        Image.Image: The loaded image object
    
    Raises:
        IOError: If the image cannot be opened or is invalid
    """
    logger.debug(f"Loading image: {image_path}")
    try:
        img = Image.open(image_path)
        img.load()
        logger.info(f"Successfully loaded image: {image_path}")
        return img
    except Exception as e:
        logger.exception(f"Unable to load image '{image_path}'.")
        raise IOError(f"Unable to load image '{image_path}'. Error: {e}") from e