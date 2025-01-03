from PySide6.QtWidgets import (QLabel)
from PySide6.QtGui import QImage, QPixmap
import cv2

class CameraView(QLabel):

    def __init__(self):
        super().__init__()
        
        self.width = 300
        self.height = 300

        # set the size of each camera
        self.setStyleSheet("border: 1px solid dark brown")
        self.setFixedSize(self.width, self.height)

    def setImage(self, image):
        height, width, color = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # fit to 300 pixels
        image = cv2.resize(image, (self.width, int(height * self.width/width)))
        
        image = QImage(image, image.shape[1], image.shape[0], QImage.Format.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(image))


