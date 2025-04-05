"""
This module defines the data structures used to manage 
image annotations in the MOS Annotator app.

Classes:
    - ImageRecord: Represents an image with its file name, path, associated scores,
      and a flag indicating if it's a 'representative' image that must be annotated.
    - AnnotationSession: Manages a collection of ImageRecord objects, including
      navigation, score updates, and CSV export.
      
Usage:
    1. Create ImageRecord objects for each image.
    2. Initialize an AnnotationSession with those records.
    3. Use session methods to navigate between records, update scores,
       and optionally export results to CSV.
"""

import csv
import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(name)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


@dataclass
class ImageRecord:
    """
    Data class representing an image and its associated scores.

    Attributes:
        file_name (str): The name of the image file.
        file_path (Path): The file path to the image.
        manual_score (Optional[float]): The manually annotated score, in [0, 1].
        auto_score (Optional[float]): The automatically computed score, in [0, 1].
        is_required (bool): Whether this image is 'representative' and must be annotated.
    """
    file_name: str
    file_path: Path
    manual_score: Optional[float] = None
    auto_score: Optional[float] = None
    is_required: bool = False

    def get_final_score(self) -> Optional[float]:
        """
        Determine the final score (MOS) for the image.

        Returns:
            Optional[float]: The final score using the manual score if available,
                             otherwise the auto score. If neither is available,
                             returns None.
        """
        if self.manual_score is not None:
            return self.manual_score
        return self.auto_score


class AnnotationSession:
    """
    Manages a collection of ImageRecord objects and provides helper methods for
    navigation, updating scores, and exporting the results.

    Attributes:
        image_records (List[ImageRecord]): A list of ImageRecord objects.
        current_index (int): The index of the currently active image record.
    """

    def __init__(self, image_records: Optional[List[ImageRecord]] = None):
        """
        Initialize the AnnotationSession.

        Parameters:
            image_records (Optional[List[ImageRecord]]): A preloaded list of image records.
                                                         If None, initializes with an empty list.
        """
        self.image_records: List[ImageRecord] = image_records if image_records else []
        self.current_index: int = 0
        logger.debug(f"AnnotationSession initialized with {len(self.image_records)} image record(s).")

    def get_current_record(self) -> Optional[ImageRecord]:
        """
        Retrieve the currently selected image record.

        Returns:
            Optional[ImageRecord]: The current ImageRecord, or None if the session is empty.
        """
        if not self.image_records:
            logger.warning("No image records available in the session.")
            return None
        return self.image_records[self.current_index]

    def next_record(self) -> Optional[ImageRecord]:
        """
        Move to the next image record if available.

        Returns:
            Optional[ImageRecord]: The next ImageRecord, or None if already at the last record.
        """
        if self.current_index < len(self.image_records) - 1:
            self.current_index += 1
            logger.debug(f"Moved to next record: index {self.current_index}.")
            return self.get_current_record()
        else:
            logger.info("Already at the last image record.")
            return None

    def previous_record(self) -> Optional[ImageRecord]:
        """
        Move to the previous image record if available.

        Returns:
            Optional[ImageRecord]: The previous ImageRecord, or None if already at the first record.
        """
        if self.current_index > 0:
            self.current_index -= 1
            logger.debug(f"Moved to previous record: index {self.current_index}.")
            return self.get_current_record()
        else:
            logger.info("Already at the first image record.")
            return None

    def update_manual_score(self, index: int, score: float) -> None:
        """
        Update the manual score for a specific image record.

        Parameters:
            index (int): The index of the image record to update.
            score (float): The score to assign (expected between 0 and 1).

        Raises:
            IndexError: If the provided index is out of range.
            ValueError: If the score is not within the valid range [0, 1].
        """
        if not (0 <= score <= 1):
            logger.error(f"Invalid score value: {score}. Score must be between 0 and 1.")
            raise ValueError("Score must be between 0 and 1.")

        try:
            record = self.image_records[index]
            record.manual_score = score
            logger.info(f"Updated manual score for '{record.file_name}' to {score:.4f}")
        except IndexError as e:
            logger.exception(f"IndexError: Provided index {index} is out of range.")
            raise IndexError("The provided index is out of range.") from e

    def update_auto_score(self, index: int, score: float) -> None:
        """
        Update the automatic score for a specific image record.

        Parameters:
            index (int): The index of the image record to update.
            score (float): The automatic score to assign (expected between 0 and 1).

        Raises:
            IndexError: If the provided index is out of range.
            ValueError: If the score is not within the valid range [0, 1].
        """
        if not (0 <= score <= 1):
            logger.error(f"Invalid score value: {score}. Score must be between 0 and 1.")
            raise ValueError("Score must be between 0 and 1.")

        try:
            record = self.image_records[index]
            record.auto_score = score
            logger.info(f"Updated auto score for '{record.file_name}' to {score:.4f}")
        except IndexError as e:
            logger.exception(f"IndexError: Provided index {index} is out of range.")
            raise IndexError("The provided index is out of range.") from e

    def export_to_csv(self, csv_file: str) -> None:
        """
        Export the final scores (MOS) for all image records to a CSV file, sorted numerically
        by the leading digits of the file name (e.g., '14_xxx.png' < '149_xxx.png').
        
        The CSV file will contain two columns:
            - Image_Name
            - MOS (formatted to four decimals, empty if no score)
        """
    
        logger.info(f"Exporting annotation results to CSV file: {csv_file}")
    
        def numeric_key(record):

            match = re.match(r'^(\d+)_', record.file_name)
            return int(match.group(1)) if match else 999999
    
        #sort by the integer prefix of the file_name
        sorted_records = sorted(self.image_records, key=numeric_key)
    
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Image_Name", "MOS"])
            for record in sorted_records:
                final_score = record.get_final_score()
                mos_value = f"{final_score:.4f}" if final_score is not None else ""
                writer.writerow([record.file_name, mos_value])
                logger.debug(f"Wrote {record.file_name} -> {mos_value}")
    
        logger.info("Export completed successfully.")