from PySide6.QtWidgets import (QLabel, QWidget, QGridLayout, QLineEdit, QPushButton, QCheckBox, QFileDialog)
from PySide6.QtCore import Qt, Signal

chosen_settings = {"site_id": "",
                   "feature_id": "",
                   "level_id": "",
                   "directory_name": "",
                   "csv_file_name": "",
                   "classify_bool": False,
                   "training_data_dir": ""}
#training_data_dir is a bad name, its really just the dir you want to append your info to alongside the one we create/are already using

class Settings(QWidget):
    # Signal to pass data to other files
    save_settings_signal = Signal()

    def __init__(self):
        super().__init__()

        grid = QGridLayout()
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        grid.setColumnStretch(3, 1)
        grid.setColumnStretch(4, 1)

        # Option 1: Site ID
        site_id_message = QLabel("Enter Site ID:")
        self.site_id_input = QLineEdit()

        grid.addWidget(site_id_message, 1, 0, 1, 5)
        grid.addWidget(self.site_id_input, 2, 0, 1, 4)

        # Option 2: Feature ID
        feature_id_message = QLabel("Enter Feature ID (optional):")
        self.feature_id_input = QLineEdit()

        grid.addWidget(feature_id_message, 3, 0, 1, 5)
        grid.addWidget(self.feature_id_input, 4, 0, 1, 4)

        # Option 3: Level ID
        level_id_message = QLabel("Enter Level ID (optional):")
        self.level_id_input = QLineEdit()

        grid.addWidget(level_id_message, 5, 0, 1, 5)
        grid.addWidget(self.level_id_input, 6, 0, 1, 4)

        # Option 4: Classify images?
        classify_checkbox_message = QLabel("Classify images with deep learning model:")
        self.classify_checkbox = QCheckBox()

        grid.addWidget(classify_checkbox_message, 7, 0, 1, 2)
        grid.addWidget(self.classify_checkbox, 7, 2)

        # Option 6: Add to training data?
        training_data_message = QLabel("Select an existing CSV file of training data to add images to:")
        self.training_data_input = QLineEdit()
        self.training_data_input.setReadOnly(True)
        training_data_button = QPushButton("Search Files")
        training_data_button.clicked.connect(self.training_csv_dialog)

        grid.addWidget(training_data_message, 9, 0, 1, 5)
        grid.addWidget(self.training_data_input, 10, 0, 1, 4)
        grid.addWidget(training_data_button, 10, 4)

        # Submit button
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_settings)
        grid.addWidget(save_button, 11, 2, 1, 1)

        # Set the layout
        grid.setRowStretch(0, 1)
        grid.setRowStretch(12, 1)
        self.setLayout(grid)

    def save_settings(self):
        self.save_settings_signal.emit()

        # Save settings data
        chosen_settings["site_id"] = self.site_id_input.text()
        chosen_settings["feature_id"] = self.feature_id_input.text()
        chosen_settings["level_id"] = self.level_id_input.text()

        # Create directory and CSV file name
        site_id = chosen_settings["site_id"]
        feature_id = chosen_settings["feature_id"]
        level_id = chosen_settings["level_id"]

        if feature_id and level_id:
            dir_csv_name = f"{site_id}_{feature_id}_{level_id}"
        elif feature_id:
            dir_csv_name = f"{site_id}_{feature_id}"
        else:
            dir_csv_name = site_id

        chosen_settings["directory_name"] = dir_csv_name
        chosen_settings["csv_file_name"] = f"{dir_csv_name}.csv"

        chosen_settings["classify_bool"] = self.classify_checkbox.isChecked()
        chosen_settings["training_data_dir"] = self.training_data_input.text()
        print(dir_csv_name)

    def training_csv_dialog(self):
        file = QFileDialog.getOpenFileName(self, "Open CSV", "/home", "CSV Files (*.csv)")
        self.training_data_input.setText(file[0])
