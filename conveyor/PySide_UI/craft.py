import sys
from PySide6.QtWidgets import ( QMainWindow, QApplication,
QLabel, QStackedLayout, QWidget, QVBoxLayout)
from PySide6.QtCore import Qt
from image_process import ImageProcess
from settings import Settings

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # window setup
        self.setWindowTitle('Sherd Conveyor System')
        self.setFixedSize(900, 500)
        layout = QVBoxLayout()
        self.pages = QStackedLayout()

        # connect pages
        image_process = ImageProcess()
        image_process.go_to_settings.connect(self.show_settings_page)  
        settings = Settings()  
        settings.save_settings_signal.connect(self.show_process_page)

        # add pages
        self.pages.addWidget(image_process)
        self.pages.addWidget(settings)

        # craft banner
        title_label = QLabel("CRAFT")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font: 40px; font-weight: bold; color: dark brown")  
        layout.addWidget(title_label) 

        # setting the layout
        stackedLayoutWidget = QWidget()
        stackedLayoutWidget.setLayout(self.pages)
        layout.addWidget(stackedLayoutWidget)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def show_process_page(self):
        self.pages.setCurrentIndex(0)

    def show_settings_page(self):
        self.pages.setCurrentIndex(1)
        

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()