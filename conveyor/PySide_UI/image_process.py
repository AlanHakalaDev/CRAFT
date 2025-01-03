from PySide6.QtWidgets import (
    QWidget, QGridLayout, QPushButton, QHBoxLayout, QCheckBox)
from PySide6.QtCore import Qt, Signal
from cameraView import CameraView
from camera_thread import CameraThread
from capturedImageView import CapturedImageView
from settings import chosen_settings

class ImageProcess(QWidget):
    go_to_settings = Signal()

    def __init__(self):
        super().__init__()
        grid = QGridLayout()

        # set up camera view and connect signal
        grid.setColumnStretch(0, 1)
        self.camera_label = CameraView()
        grid.addWidget(self.camera_label, 0, 1)
        grid.setColumnStretch(2, 1)
        self.cameraThread = CameraThread()
        self.cameraThread.video_view_signal.connect(self.camera_label.setImage)

        # display captured image
        self.captured_image = CapturedImageView()
        grid.addWidget(self.captured_image, 0, 3)
        grid.setColumnStretch(4, 1)
        self.cameraThread.image_captured_signal.connect(self.captured_image.setImage)
        
        # create start and stop buttons
        buttonWidget = QWidget()
        button_box = QHBoxLayout()
        buttonWidget.setLayout(button_box)
        self.startButton = QPushButton("Start")
        self.stopButton = QPushButton("Stop")
        button_box.addWidget(self.startButton)
        button_box.addWidget(self.stopButton)
        grid.addWidget(buttonWidget, 1, 1, 1, 3)

        # create settings button
        self.settingsButton = QPushButton("Settings")
        grid.addWidget(self.settingsButton, 2, 2)
        self.settingsButton.clicked.connect(self.handle_settings)

        # connect video functions to buttons
        self.startButton.clicked.connect(self.cameraThread.start_thread)
        self.stopButton.clicked.connect(self.stop_thread)
        
        # set the layout
        self.setLayout(grid)

    def stop_thread(self):
        self.cameraThread.stop = True

    def handle_settings(self):
        self.go_to_settings.emit()