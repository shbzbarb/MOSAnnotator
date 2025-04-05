import logging
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSlider, QLabel
from PyQt5.QtCore import Qt, pyqtSignal

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    from PyQt5.QtWidgets import QStyle
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(name)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class AnnotationsPanel(QWidget):
    """
    A custom widget for annotating images with a score using a slider
    
    The widget includes:
        - A QSlider (horizontal) with a range from 0 to 10000, representing scores 0.0000 to 1.0000
        - A QLabel that displays the current score formatted to four decimal places
    
    It emits a 'scoreChanged' signal (float) whenever the slider's value is updated
    """
    scoreChanged = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        """
        Set up the user interface elements: a slider and a label, arranged vertically.
        """
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(10000)  # 0 to 1 in increments of 0.0001
        self.slider.setValue(5000)     # Default score of 0.5000
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1000)  # Tick every 0.1000 change

        self.scoreLabel = QLabel("Score: 0.5000", self)
        self.scoreLabel.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.scoreLabel)
        layout.addWidget(self.slider)
        self.setLayout(layout)

        self.slider.valueChanged.connect(self.updateScoreLabel)
        logger.debug("AnnotationsPanel UI initialized with slider range 0-10000.")

    def updateScoreLabel(self, value: int):
        """
        Slot that updates the score label and emits the scoreChanged signal when the slider changes.
        """
        #converting the slider value to a float between 0.0000 and 1.0000
        score = value / 10000.0
        self.scoreLabel.setText(f"Score: {score:.4f}")
        self.scoreChanged.emit(score)
        logger.debug(f"Score updated to {score:.4f}")

    def getScore(self) -> float:
        """
        Retrieve the current score from the slider.
        """
        return self.slider.value() / 10000.0

    def setScore(self, score: float):
        """
        Set the slider to a specific score and update the label accordingly.
        
        Parameters:
            score (float): The score to set (must be between 0 and 1).
        """
        if not (0.0 <= score <= 1.0):
            logger.error("Attempted to set invalid score: %f", score)
            raise ValueError("Score must be between 0 and 1.")
        slider_value = int(score * 10000)
        self.slider.setValue(slider_value)
        logger.debug("Score set to %f (slider value: %d)", score, slider_value)