"""
This module defines the MainWindow class, the primary interface for the MOS Annotator application
It integrates:
  - An ImageViewer widget to display images.
  - An AnnotationsPanel widget for manual scoring
  - Navigation (Previous/Next).
  - Controls to load a folder, pick the representative method (k-means or DBSCAN),
    pick auto scoring mode (LPIPS, SSIM, MS-SSIM, Combined), set alpha/beta/gamma,
    or do ResNet50+KNN with user-defined k and weighting
  - CSV export

Representative images are forced to be annotated first, so they are sorted to the front
"""

import logging
from pathlib import Path

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QComboBox, QLabel, QMessageBox,
    QCheckBox, QProgressBar, QSpinBox, QDoubleSpinBox, QRadioButton, QButtonGroup
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PIL import Image

from gui.image_viewer import ImageViewer
from gui.annotations_panel import AnnotationsPanel

from core.image_loader import load_images, load_image
from core.data_model import AnnotationSession, ImageRecord
from core.auto_scoring import auto_score_images
from core.model import (
    ResNetFeatureExtractor,
    KNNMOSPredictor,
    pick_representative_images,
    pick_representative_images_dbscan
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(name)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class MainWindow(QMainWindow):
    """
    The primary GUI window for the MOS Annotator application.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MOS Annotator")
        self.setGeometry(100, 100, 1280, 720)

        self.session = None
        self.image_files = []

        self.initUI()

    def initUI(self):
        """
        Set up the main window's user interface.
        """
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()

        # TOP PANEL: Folder, auto score, save CSV, metric mode, CNN toggle, KNN config, alpha/beta/gamma        
        top_panel = QHBoxLayout()
        self.load_folder_btn = QPushButton("Load Folder")
        self.auto_score_btn = QPushButton("Auto Score")
        self.save_csv_btn = QPushButton("Save CSV")

        #Scoring modes (LPIPS, SSIM, MS-SSIM, Combined)
        mode_label = QLabel("Auto Scoring Mode:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["LPIPS", "SSIM", "MS-SSIM", "Combined"])

        #CNN-based auto scoring
        self.use_cnn_checkbox = QCheckBox("Use ResNet50 + KNN")

        #K spin
        k_label = QLabel("k:")
        self.k_spin = QSpinBox()
        self.k_spin.setRange(1, 50)
        self.k_spin.setValue(3)

        #Weighting radio (uniform/distance)
        self.weighting_group = QButtonGroup()
        self.uniform_radio = QRadioButton("Uniform")
        self.distance_radio = QRadioButton("Distance")
        self.uniform_radio.setChecked(True)
        self.weighting_group.addButton(self.uniform_radio)
        self.weighting_group.addButton(self.distance_radio)

        #Weighted combined (alpha,beta,gamma)
        alpha_label = QLabel("α:")
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setValue(0.33)

        beta_label = QLabel("β:")
        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setRange(0.0, 1.0)
        self.beta_spin.setSingleStep(0.05)
        self.beta_spin.setValue(0.33)

        gamma_label = QLabel("γ:")
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.0, 1.0)
        self.gamma_spin.setSingleStep(0.05)
        self.gamma_spin.setValue(0.34)

        #representative method as a combo box (for selecting k-means or DBSCAN)
        rep_method_label = QLabel("Rep. Method:")
        self.rep_method_combo = QComboBox()
        self.rep_method_combo.addItems(["k-means","DBSCAN"])  #default is k-means

        #Add to the top panel
        top_panel.addWidget(self.load_folder_btn)
        top_panel.addWidget(self.auto_score_btn)
        top_panel.addWidget(self.save_csv_btn)
        top_panel.addStretch()

        top_panel.addWidget(mode_label)
        top_panel.addWidget(self.mode_combo)
        top_panel.addWidget(self.use_cnn_checkbox)

        top_panel.addWidget(rep_method_label)
        top_panel.addWidget(self.rep_method_combo)

        top_panel.addWidget(k_label)
        top_panel.addWidget(self.k_spin)
        top_panel.addWidget(self.uniform_radio)
        top_panel.addWidget(self.distance_radio)

        top_panel.addWidget(alpha_label)
        top_panel.addWidget(self.alpha_spin)
        top_panel.addWidget(beta_label)
        top_panel.addWidget(self.beta_spin)
        top_panel.addWidget(gamma_label)
        top_panel.addWidget(self.gamma_spin)

        main_layout.addLayout(top_panel)

        
        # MIDDLE PANEL: Image Viewer
        self.image_viewer = ImageViewer()
        main_layout.addWidget(self.image_viewer, stretch=1)


        # BOTTOM PANEL: Navigation, Annotation, and Progress Bar
        bottom_panel = QHBoxLayout()
        self.prev_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")
        self.annotations_panel = AnnotationsPanel()

        bottom_panel.addWidget(self.prev_btn)
        bottom_panel.addWidget(self.next_btn)
        bottom_panel.addStretch()
        bottom_panel.addWidget(self.annotations_panel)
        main_layout.addLayout(bottom_panel)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        central_widget.setLayout(main_layout)

        #Connect signals
        self.load_folder_btn.clicked.connect(self.loadFolder)
        self.prev_btn.clicked.connect(self.showPreviousImage)
        self.next_btn.clicked.connect(self.showNextImage)
        self.annotations_panel.scoreChanged.connect(self.updateCurrentImageScore)
        self.auto_score_btn.clicked.connect(self.runAutoScoring)
        self.save_csv_btn.clicked.connect(self.saveCSV)

    def loadFolder(self):
        """
        Load images from a folder, pick representative images using either k-means
        or DBSCAN, and build an AnnotationSession. Required reps are sorted first
        """
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder", "")
        if not folder:
            return
        try:
            self.image_files = load_images(folder)
            if not self.image_files:
                QMessageBox.warning(self, "No Images", "No valid images found")
                return

            #1. selecting which approach for picking reps 'k-means' or 'DBSCAN'
            rep_method = self.rep_method_combo.currentText()
            if rep_method == "DBSCAN":
                
                eps_val = 0.5
                min_samples_val = 5
                reps = pick_representative_images_dbscan(
                    self.image_files,
                    eps=eps_val,
                    min_samples=min_samples_val,
                    device='cpu'
                )
                logger.info(f"DBSCAN reps picked with eps={eps_val}, min_samples={min_samples_val}")
            else:
                
                # default is 'k-means'
                n_clusters = 5
                reps = pick_representative_images(
                    self.image_files,
                    n_clusters=n_clusters,
                    device='cpu'
                )
                logger.info(f"k-means reps picked with n_clusters={n_clusters}")

            #2. building records, marking reps as required
            image_records = []
            for p in self.image_files:
                rec = ImageRecord(file_name=p.name, file_path=p)
                if p in reps:
                    rec.is_required = True
                image_records.append(rec)

            #3. Sort the images so that is_required images appear first
            image_records.sort(key=lambda r: not r.is_required)

            self.session = AnnotationSession(image_records)
            self.showCurrentImage()
            logger.info(f"Loaded {len(self.image_files)} images from {folder}")

        except Exception as e:
            logger.exception("Failed to load images from folder.")
            QMessageBox.critical(self, "Error", f"Failed to load images: {str(e)}")

    def showCurrentImage(self):
        """
        Display the current image from the AnnotationSession in the ImageViewer
        """
        if not self.session:
            return
        rec = self.session.get_current_record()
        if not rec:
            return
        try:
            img = load_image(str(rec.file_path))
            if img.mode != 'RGB':
                img = img.convert('RGB')

            data = img.tobytes("raw", "RGB")
            qimg = QImage(data, img.width, img.height, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.image_viewer.setImage(pixmap)

            current_score = rec.manual_score if rec.manual_score is not None else 0.5
            self.annotations_panel.setScore(current_score)
            logger.info(f"Displayed {rec.file_name}, score={current_score:.4f}, required={rec.is_required}")
        except Exception as e:
            logger.exception("Failed to display image.")
            QMessageBox.critical(self, "Error", str(e))

    def showNextImage(self):
        """
        Navigate to the next image in the session and display it.
        """
        if not self.session:
            return
        next_rec = self.session.next_record()
        if next_rec:
            self.showCurrentImage()
        else:
            QMessageBox.information(self, "End", "Last image reached")

    def showPreviousImage(self):
        """
        Navigate to the previous image in the session and display it
        """
        if not self.session:
            return
        prev_rec = self.session.previous_record()
        if prev_rec:
            self.showCurrentImage()
        else:
            QMessageBox.information(self, "Start", "First image")

    def updateCurrentImageScore(self, score: float):
        """
        Update the manual score for the current image record when the slider value changes
        """
        if not self.session:
            return
        idx = self.session.current_index
        try:
            self.session.update_manual_score(idx, score)
            logger.info(f"Set manual_score={score:.4f} for idx={idx}")
        except Exception as e:
            logger.exception("Failed to set manual score.")
            QMessageBox.critical(self, "Error", str(e))

    def runAutoScoring(self):
        """
        Trigger automatic scoring for unannotated images.
        A progress bar is displayed as each image is processed.
        """
        if not self.session:
            QMessageBox.warning(self, "No Data", "Please load images before running auto scoring.")
            return

        #ensuring all required images are annotated
        required_unannotated = [r for r in self.session.image_records if r.is_required and r.manual_score is None]
        if required_unannotated:
            names = ", ".join(r.file_name for r in required_unannotated)
            QMessageBox.warning(self, "Required Images Not Annotated",
                f"The following representative images must be annotated first:\n{names}")
            return

        unannotated = [r for r in self.session.image_records if r.manual_score is None]
        total_u = len(unannotated)
        if total_u == 0:
            QMessageBox.information(self, "Auto Scoring", "All images are annotated.")
            return
        
        self.progress_bar.setMaximum(total_u)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        #CNN-based (ResNet50 + KNN) or metric-based approach
        if self.use_cnn_checkbox.isChecked():
            logger.info("CNN-based auto scoring (ResNet50 + KNN).")
            try:
                from core.image_loader import load_image
                annotated_feats = []
                annotated_scores = []
                for rec in self.session.image_records:
                    if rec.manual_score is not None:
                        img = load_image(str(rec.file_path))
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        extractor = ResNetFeatureExtractor(device='cpu')
                        feat_vec = extractor.extract_features(img)
                        annotated_feats.append(feat_vec)
                        annotated_scores.append(rec.manual_score)

                if not annotated_feats:
                    QMessageBox.warning(self, "No Annotated Data",
                                        "At least one manually annotated image is required for KNN-based scoring")
                    self.progress_bar.setVisible(False)
                    return

                k_val = self.k_spin.value()
                weighting_mode = 'distance' if self.distance_radio.isChecked() else 'uniform'
                predictor = KNNMOSPredictor(annotated_feats, annotated_scores,
                                            k=k_val, weighting=weighting_mode)

                processed = 0
                predictions = {}
                for rec in self.session.image_records:
                    if rec.manual_score is None:
                        img = load_image(str(rec.file_path))
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        extractor = ResNetFeatureExtractor(device='cpu')
                        feat_vec = extractor.extract_features(img)
                        pred_mos = predictor.predict(feat_vec)
                        predictions[rec.file_name] = pred_mos
                        processed += 1
                        self.progress_bar.setValue(processed)

                for i, rec in enumerate(self.session.image_records):
                    if rec.manual_score is None and rec.file_name in predictions:
                        self.session.update_auto_score(i, predictions[rec.file_name])

                QMessageBox.information(self, "Auto Scoring Completed", "CNN-based KNN scoring finished")
            except Exception as e:
                logger.exception("CNN-based auto scoring failed.")
                QMessageBox.critical(self, "Error", str(e))
        else:
            logger.info("Metric-based auto scoring (LPIPS, SSIM, MS-SSIM, or Weighted Combined)")
            try:
                from core.image_loader import load_image
                annotated_data = []
                unannotated_data = []
                for rec in self.session.image_records:
                    img = load_image(str(rec.file_path))
                    if img.mode!='RGB':
                        img = img.convert('RGB')
                    if rec.manual_score is not None:
                        annotated_data.append((rec.file_name, img, rec.manual_score))
                    else:
                        unannotated_data.append((rec.file_name, img))

                if not annotated_data:
                    QMessageBox.warning(self, "Insufficient Data",
                                        "No manually annotated images available for metric-based scoring")
                    self.progress_bar.setVisible(False)
                    return

                mode = self.mode_combo.currentText().lower()
                alpha=1/3; beta=1/3; gamma=1/3

                if mode=='combined':
                    alpha = float(self.alpha_spin.value())
                    beta  = float(self.beta_spin.value())
                    gamma = float(self.gamma_spin.value())
                    s = alpha+beta+gamma
                    if abs(s-1.0)>0.001:
                        QMessageBox.warning(self, "Invalid Weights",
                            "Please ensure alpha+beta+gamma=1.0 for Combined mode.")
                        self.progress_bar.setVisible(False)
                        return

                all_preds = auto_score_images(
                    annotated_data,
                    unannotated_data,
                    mode=mode,
                    alpha=alpha, beta=beta, gamma=gamma
                )

                processed=0
                for i, rec in enumerate(self.session.image_records):
                    if rec.manual_score is None and rec.file_name in all_preds:
                        self.session.update_auto_score(i, all_preds[rec.file_name])
                        processed+=1
                        self.progress_bar.setValue(processed)

                QMessageBox.information(self, "Auto Scoring Completed", "Metric-based scoring finished.")
            except Exception as e:
                logger.exception("Metric-based auto scoring failed.")
                QMessageBox.critical(self, "Error", str(e))

        self.progress_bar.setVisible(False)

    def saveCSV(self):
        """
        Save the final annotations (MOS scores) to a CSV file.
        """
        if not self.session:
            QMessageBox.warning(self, "No Data", "No annotations to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save MOS CSV", "", "CSV Files (*.csv)")
        if path:
            try:
                self.session.export_to_csv(path)
                QMessageBox.information(self, "Saved", f"CSV saved to {path}")
            except Exception as e:
                logger.exception("Failed to save CSV.")
                QMessageBox.critical(self, "Error", str(e))