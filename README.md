# MOS ANNOTATOR APP
Assigning Mean Opinion Scores (MOS) manually to large-scale image datasets is a highly labor-intensive, tedious, and time-consuming task. MOSAnnotator addresses these challenges by effectively integrating manual annotation with automated methods, significantly reducing the human effort involved while ensuring accuracy and consistency. MOSAnnotator is an image annotation tool designed specifically for efficient and precise MOS assignment. It operates by allowing users to manually annotate only a carefully selected subset of representative images. This subset selection leverages advanced clustering methods, including K-means and DBSCAN, to identify the most representative images within a dataset. Once manually annotated, the remaining images can be automatically scored with high accuracy using sophisticated image similarity metrics and powerful machine learning models. The primary motivations driving MOSAnnotator include:
- Minimizing manual annotation through reliable automated predictions
- Incorporating advanced similarity metrics to enhance scoring accuracy
- Leveraging deep learning methodologies for robust feature extraction and automated scoring
- Streamlining and simplifying the annotation workflow, enabling users to focus predominantly on representative image annotations


## Key Features

- **Selective Manual Annotation**: Users manually annotate a representative subset of images, dramatically minimizing manual workload

- **Automated Scoring with Advanced Metrics**: The tool integrates advanced similarity metrics such as LPIPS, SSIM, MS-SSIM, or a user-defined weighted combination to generate precise and consistent automated scores

- **Robust CNN-Based Scoring**: It incorporates deep learning techniques using a ResNet50-based CNN model for feature extraction, combined with regression methods such as k-nearest neighbors (KNN) for robust and accurate automated MOS prediction

- **Advanced Clustering Techniques**: Efficient clustering algorithms K-means and DBSCAN ensure representative images are intelligently selected, streamlining the annotation process

- **Accelerated Annotation Workflows**: By automating significant portions of the scoring task, MOSAnnotator accelerates the annotation workflow, substantially reducing manual effort and ensuring consistency across large-scale MOS annotation projects


## Project Structure
```
└── image_annotator/
    ├── core/
    │   ├── __init__.py
    │   ├── image_loader.py    # Functions to load images from a folder
    │   ├── data_model.py      # Data structures/classes for image records and session management
    │   ├── metrics.py         # Implementation of SSIM, LPIPS, and combined metrics
    │   ├── auto_scoring.py    # Module to automatically assign scores to remaining images
    │   └── model.py           # Code for loading and using a CNN (e.g., ResNet50) for scoring
    ├── gui/
    │   ├── __init__.py
    │   ├── main_window.py          # Main window containing the overall GUI layout
    │   ├── image_viewer.py         # Custom widget for displaying images
    │   └── annotations_panel.py    # Widget for the scoring slider and score display
    ├── main.py                # Entry point to launch the PyQt5 application
    ├── output/
    │   └── MOS.csv            # Output file for the final image scores (auto generated)
    ├── README.md              # Project overview, installation, and usage instructions
    └── requirements.txt       # List of packages and libraries
```


## Key Components and Modules
### Core (core/)
**```image_loader.py```**
- load_images(directory): Scans and loads images from a specified directory, returning their paths
- load_image(path): Loads a single image and converts it into a PIL Image object, handling color conversions as needed (grayscale or RGB)

**```data_model.py```**
- ImageRecord: Data class holding individual image metadata, paths, manual MOS scores, automatic MOS scores, and whether the image is required (representatives)
- AnnotationSession: Manages the list of ImageRecord instances, provides functionality for navigating between records (next_record, previous_record), updating MOS scores, and exporting results into a numerically sorted CSV file

**```metrics.py```**
- Implements advanced image similarity metrics used for MOS prediction
- compute_lpips: Computes Learned Perceptual Image Patch Similarity (LPIPS)
- compute_ssim: Computes Structural Similarity Index Measure (SSIM)
- compute_ms_ssim: Computes Multi-scale SSIM (MS-SSIM)
- compute_combined_metric: Computes a weighted combination (alpha, beta, gamma) of LPIPS→similarity, SSIM, and MS-SSIM for comprehensive image comparison

**```auto_scoring.py```**
- Assigns automatic MOS scores to unannotated images based on similarity with manually annotated reference images
- Supports scoring modes such as LPIPS, SSIM, MS-SSIM, and Combined mode with adjustable weighting (alpha, beta, gamma)

**```model.py```**
- ResNetFeatureExtractor: Utilizes a pretrained ResNet50 CNN for robust feature extraction from images
- KNNMOSPredictor: Implements a K-Nearest Neighbors regression in the feature space for predicting MOS scores from ResNet50-extracted features. Supports both uniform and distance-based weighting
- pick_representative_images: Selects representative images using either k-means clustering or DBSCAN clustering on ResNet50-extracted image features. These images are prioritized for manual annotation

### GUI (gui/)
**```image_viewer.py```**
- Implements the GUI widget for displaying images within the application
- Handles automatic resizing and scaling of images to fit the application's viewport while maintaining aspect ratio

**```annotations_panel.py```**
- Provides a user interface element (slider) allowing manual assignment of MOS scores to images
- Emits a signal (scoreChanged) each time the slider value changes, ensuring immediate synchronization between the GUI and backend data model


**```main_window.py```**
- Core GUI integrating all functionalities, providing controls
- Loading image directories and selecting representative images using clustering methods (k-means or DBSCAN)
- Selecting auto-scoring methods (CNN-based KNN, LPIPS, SSIM, MS-SSIM, Combined metric)
- Adjusting parameters such as k for KNN, weighting modes (uniform or distance), and combined metric weights (alpha, beta, gamma)
- Displaying a progress bar during auto-scoring processes to visually track progress
- Exporting final MOS results to a numerically sorted CSV file


## Complete Workflow
### Step 1: Launch Application
- User starts the application using:
```sh
python main.py
```

### Step 2: Load Image Folder
- Images are loaded using ```core.image_loader.py```
- User selects a representative image selection method (**k-means** or **DBSCAN**)
- Representatives are marked as required (```is_required=True```) and appear first in the annotation session

### Step 3: Manual Annotation
- Navigate images (Next/Previous)
- Assign MOS scores via annotation slider (GUI)
- Required images must be annotated before proceeding

### Step 4: Automatic Scoring
Select scoring method:
- ResNet50 + KNN (CNN-based approach)
    - Parameters: number of neighbors (k) and weighting mode (uniform/distance)
    - Requires at least one manually annotated image

- Metric-based Scoring:
    - Modes: LPIPS, SSIM, MS-SSIM, Combined
    - Optional weighted metric (alpha/beta/gamma)

- Automatic MOS scoring applied to unannotated images

- Progress bar provides real-time feedback


## Step 5: Export Results
- Final scores can be saved at any location as MOS.csv (or otherwise named by the user)
- The CSV file consists of data with two columns ```Image_Name```, ```MOS```
- The ```Image_Name``` is numerically sorted


## Conclusion and Future Work
### Conclusion
The MOSAnnotator application significantly reduces the manual effort associated with assigning Mean Opinion Scores (MOS) to large-scale image datasets. By combining strategic manual annotations focused on representative images selected through robust clustering methods with advanced automated scoring techniques, MOSAnnotator ensures high-quality and reliable MOS assignment. The integration of sophisticated metrics like LPIPS, SSIM, and MS-SSIM, along with powerful CNN-based feature extraction (ResNet50) combined with KNN and DBSCAN clustering, empowers users with flexibility and accuracy in their annotation workflows

### Future Work
- Integration of Additional CNN Architectures: Evaluate and integrate alternative feature extraction models (e.g., EfficientNet, Vision Transformers) to potentially boost scoring accuracy
- Advanced Clustering and Scoring Algorithms: Explore semi-supervised and unsupervised deep learning models for improved MOS predictions without significant manual annotation
- Real-time Interactive Annotation: Implement dynamic feedback systems allowing users to receive immediate model-based MOS suggestions during the manual annotation phase, streamlining the annotation experience

## References
### **CNN Feature Extraction (ResNet50)**
- **ResNet50 - PyTorch Official Documentation**: [Link](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)
- **ResNet Explained by Papers with Code**: [Link](https://paperswithcode.com/method/resnet)

### **Image Quality Metrics**
- **LPIPS (Learned Perceptual Image Patch Similarity)**:
  - Official GitHub repository: [Link](https://github.com/richzhang/PerceptualSimilarity)
  - PyPI Package: [Link](https://pypi.org/project/lpips/)

- **SSIM (Structural Similarity Index Measure)**: scikit-image Documentation: [Link](https://scikit-image.org/docs/stable/api/skimage.metrics.html#structural-similarity)

- **MS-SSIM (Multi-scale Structural Similarity)**: PyTorch implementation (pytorch-msssim): [Link](https://github.com/VainF/pytorch-msssim)

### **Clustering Techniques**
- **K-Means Clustering (scikit-learn)**: scikit-learn official documentation [Link](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

- **DBSCAN (Density-Based Spatial Clustering)**: scikit-learn official documentation [Link](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
