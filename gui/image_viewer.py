"""
image_viewer.py

This module defines the ImageViewer class, a custom PyQt5 widget for displaying images
at their original resolution within a fixed viewport size of 1280×720. The image is not resized;
if the image is larger than the viewport, scroll bars will allow access to the entire image
"""

import logging
from PyQt5.QtWidgets import QScrollArea, QLabel, QSizePolicy
from PyQt5.QtGui import QPixmap, QPalette
from PyQt5.QtCore import Qt


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(name)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class ImageViewer(QScrollArea):
    """
    ImageViewer displays images at their original resolution within a fixed viewport
    of 1280×720. The image is displayed without scaling; if the image is larger than the
    viewport, scroll bars will appear to allow viewing the entire image.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        #fixed viewport size to 1280×720
        self.setFixedSize(1280, 720)
        self._init_ui()

    def _init_ui(self):
        """
        Set up the UI with a QLabel inside a QScrollArea. The QLabel will display the QPixmap
        at its original size.
        """
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setBackgroundRole(QPalette.Base)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        
        # NOT ENABLING SCALING OF CONTENTS
        # self.image_label.setScaledContents(False)  # This is the default behavior.
        self.setWidget(self.image_label)
        self.setWidgetResizable(True)
        logger.debug("ImageViewer UI initialized with fixed viewport 1280x720.")

    def setImage(self, pixmap: QPixmap):
        """
        Display the provided QPixmap in the viewer at its original resolution.
        
        Parameters:
            pixmap (QPixmap): The image to display.
        """
        if pixmap.isNull():
            logger.error("Attempted to set a null pixmap.")
            return
        #setting up the pixmap without scaling
        self.image_label.setPixmap(pixmap)
        
        #adjusting the label's size to match the image's original dimensions
        self.image_label.resize(pixmap.size())
        logger.info("Image set in viewer at original resolution.")

    def clearImage(self):
        """
        Clear the currently displayed image.
        """
        self.image_label.clear()
        logger.info("ImageViewer cleared.")
