import cv2
from PySide6.QtCore import QThread, Signal
import numpy as np
import imutils
import time
import csv
from pathlib import Path
from image_utils import MovingAverageQ
from settings import chosen_settings
import numpy
import os

def is_centered(x, y, w, h, center_x, center_y):
    """ Check if the object is near the center of the screen """
    obj_center_x = x + w // 2
    obj_center_y = y + h // 2
    # Define a region around the center for detection
    tolerance = 50
    return (center_x - tolerance < obj_center_x < center_x + tolerance) and \
           (center_y - tolerance < obj_center_y < center_y + tolerance)

class CameraThread(QThread):
    video_view_signal = Signal(numpy.ndarray)
    image_captured_signal = Signal(numpy.ndarray)

    def __init__(self):
        super().__init__()
        self.stop = False
        self.image_saved = False

    def start_thread(self):
        self.stop = False
        self.xQ = MovingAverageQ(3)
        self.yQ = MovingAverageQ(3)
        super().start()

    def run(self):
        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        self.frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

        ret, background = cap.read()
        original_frame = background.copy()

        # Convert the background to grayscale
        background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to the background to reduce noise
        background_blur = cv2.GaussianBlur(background_gray, (21, 21), 0)

        # Screen center coordinates
        height, width = background.shape[:2]
        center_x, center_y = width // 2, height // 2

        # Flag to prevent multiple screenshots and a counter for image naming
        screenshot_taken = False
        image_counter = 0

        directory_path = chosen_settings["directory_name"]
        csv_fileName = directory_path + '.csv'

        # Ensure the directory exists
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)  # Create the directory

        csv_path = os.path.join(directory_path, csv_fileName)

        # Open CSV file in append mode ('a') if it exists, otherwise create it
        if os.path.exists(csv_path):
            print(f"CSV file '{csv_path}' exists. Data will be appended.")
            csv_file = open(csv_path, 'a', newline='')
        else:
            print(f"CSV file '{csv_path}' does not exist. Creating a new CSV file.")
            csv_file = open(csv_path, 'w', newline='')  # 'w' to create a new file
            csv_writer = csv.writer(csv_file)
            # Add only the "Image Name" header to the new CSV file
            csv_writer.writerow(["Image Name"])

        csv_writer = csv.writer(csv_file)
        while not self.stop:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert the current frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to the frame to reduce noise
            gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)
            
            # Compute the absolute difference between the background and the current frame
            diff = cv2.absdiff(background_blur, gray_blur)
            
            # Apply a threshold to get the foreground mask
            _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            
            # Find contours to detect the moving object
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create a blank mask for the largest object
            largest_object_mask = np.zeros_like(mask)
            
            # Find the largest contour
            largest_contour = None
            largest_area = 500  # Minimum area threshold
            largest_bbox = None
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > largest_area:
                    largest_area = area
                    largest_contour = contour
                    largest_bbox = cv2.boundingRect(contour)

            # If we found a large enough contour, draw only that one on the mask
            if largest_contour is not None:
                cv2.drawContours(largest_object_mask, [largest_contour], -1, 255, -1)
                
            # Invert the mask to get the background mask
            mask_inv = cv2.bitwise_not(largest_object_mask)
            
            # Create a white background of the same size as the frame
            white_background = np.full_like(frame, 255)
            
            # Use the mask to extract the moving object from the current frame
            foreground = cv2.bitwise_and(frame, frame, mask=largest_object_mask)
            
            # Use the inverse mask to extract the background from the white background
            white_bg = cv2.bitwise_and(white_background, white_background, mask=mask_inv)
            
            # Combine the foreground (moving object) with the white background
            result = cv2.add(foreground, white_bg)
            
            # Flag to check if any object is currently centered
            object_centered = False

            # Process the largest object if it exists
            if largest_bbox is not None:
                x, y, w, h = largest_bbox
                
                # Check if the object is passing through the center
                if is_centered(x, y, w, h, center_x, center_y):
                    object_centered = True
                    if not screenshot_taken:
                        screenshot_taken = True

                        # Get the center of the bounding box
                        obj_center_x = x + w // 2
                        obj_center_y = y + h // 2
                        
                        # Calculate the crop area for 400x400 image centered on the bounding box center
                        crop_x = max(0, obj_center_x - 200)
                        crop_y = max(0, obj_center_y - 200)
                        crop_x2 = min(width, crop_x + 400)
                        crop_y2 = min(height, crop_y + 400)
                        
                        # Ensure the crop is exactly 400x400
                        if (crop_x2 - crop_x) < 400:
                            crop_x = max(0, crop_x2 - 400)
                        if (crop_y2 - crop_y) < 400:
                            crop_y = max(0, crop_y2 - 400)
                        
                        # Crop the image around the center of the bounding box
                        cropped_result = result[crop_y:crop_y2, crop_x:crop_x2]
                        
                        # Send processed image to GUI
                        self.image_captured_signal.emit(cropped_result)

                        # Save edited image     
                        if not os.path.exists(directory_path):
                            os.makedirs(directory_path)
                            print(f"Created New Directory: '{directory_path}'")

                        fileName = directory_path + "_" + str(image_counter) + '.png'
                        fileName = os.path.join(directory_path, fileName)
                        cv2.imwrite(fileName, cropped_result)

                        print(f"Screenshot taken [Img Edit]: {fileName}")
                        csv_writer.writerow([fileName])      

                        # Ensure the 'archive' subdirectory exists under the main directory_path
                        archive_path = Path('.') / "Archive"
                        os.makedirs(archive_path, exist_ok=True)

                        # Save the original frame using the existing naming convention in the 'archive' directory
                        screenshot = "Sherd_" + str(image_counter) + '.png'
                        screenshot = os.path.join(archive_path, screenshot)
                        #cv2.imwrite(screenshot, frame)
                        print(f"Screenshot taken [Img Original]: {screenshot}")

                        image_counter += 1
                        self.image_saved = True

            # Reset screenshot_taken flag when no object is centered
            if not object_centered:
                screenshot_taken = False

            # Send video feed to UI
            self.video_view_signal.emit(frame)

        # Check if user would like to have sherd be classified
        if (chosen_settings["classify_bool"]):
            import modelClassify
            #close csv so it can be written to
            csv_file.close()

            # Classify sherds
            modelClassify.classifySherds(directory_path, chosen_settings["training_data_dir"])

    def get_crop_area(self, x, y, w, h, image_width, image_height):
        x1 = max(x - 20, 0)
        y1 = max(y - 20, 0)
        w1 = w + 40 if w + 40 + x <= image_width else image_width - x
        h1 = h + 40 if h + 40 + y <= image_height else image_height - y
        return x1, y1, w1, h1