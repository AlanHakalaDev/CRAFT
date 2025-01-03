import cv2
import os
from PySide6.QtCore import QThread, Signal
import numpy as np
import imutils
import time
import csv
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import shutil
import pyrebase

def is_centered(x, y, w, h, center_x, center_y):
        """ Check if the object is near the center of the screen """
        obj_center_x = x + w // 2
        obj_center_y = y + h // 2
        # Define a region around the center for detection
        tolerance = 50
        return (center_x - tolerance < obj_center_x < center_x + tolerance) and \
               (center_y - tolerance < obj_center_y < center_y + tolerance)


# Method to handle lat/long EXIF formatting
def convert_to_degrees(value):
    degrees = int(value)
    minutes = int((value - degrees) * 60)
    seconds = (value - degrees - minutes / 60) * 3600
    return (degrees, minutes, seconds)

# Method adds GPS metadata to image
def add_gps_data(image_path, lat, lon):
    img = Image.open(image_path)
    exif_data = img.getexif()
    gps_info = {}

    # Convert latitude and longitude to EXIF format
    lat_deg = convert_to_degrees(abs(lat))
    lon_deg = convert_to_degrees(abs(lon))

    gps_info[1] = 'N' if lat >= 0 else 'S'  # Latitude Ref
    gps_info[2] = lat_deg  # Latitude
    gps_info[3] = 'E' if lon >= 0 else 'W'  # Longitude Ref
    gps_info[4] = lon_deg  # Longitude

    # Write the GPS data back into the image
    exif_data[0x8825] = gps_info  # GPSInfo tag
    img.save(image_path, exif=exif_data)
    print(f"Geolocation data added to {image_path}")

class CameraThread(QThread):
    video_view_signal = Signal(np.ndarray)
    image_captured_signal = Signal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.stop = False
        self.image_saved = False

    def start_thread(self):
        # start the thread
        self.stop = False
        #will probably remove this
        #initiates run()
        super().start()

    def run(self):
        # Prompt user for latitude, longitude, directory, and CSV file input
        latitude = float(input("Enter latitude: "))
        longitude = float(input("Enter longitude: "))

        # Prompt for directory, check if it exists or create a new one
        directory = input("Enter the directory to save images (default is ~/images): ") or "~/images"
        directory = os.path.expanduser(directory)
        
        if os.path.exists(directory):
            print(f"Directory '{directory}' exists. Images will be saved there.")
        else:
            os.makedirs(directory)
            print(f"Directory '{directory}' created.")

        # Archive directory for webcam footage
        archive_directory = os.path.join(directory, "Archive")
        if not os.path.exists(archive_directory):
            os.makedirs(archive_directory)
            print(f"Archive directory '{archive_directory}' created for saving raw webcam footage.")

        # Prompt for CSV file, check if it exists or create a new one
        csv_filename = input("Enter the CSV file to log saved images: ")
        csv_path = os.path.join(directory, csv_filename)

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

        # Initialize the webcam
        cap = cv2.VideoCapture(1)

        # Read the first frame as the background
        ret, background = cap.read()

        # Convert the background to grayscale
        background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to the background to reduce noise
        background_blur = cv2.GaussianBlur(background_gray, (21, 21), 0)

        # Screen center coordinates
        height, width = background.shape[:2]
        center_x, center_y = width // 2, height // 2

        # Flag to prevent multiple screenshots and a counter for image naming
        screenshot_taken = False
        image_counter = 1  # Start counting images from 1


        while cap.isOpened and not self.stop:
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
            
            # Invert the mask to get the background mask
            mask_inv = cv2.bitwise_not(mask)
            
            # Create a white background of the same size as the frame
            white_background = np.full_like(frame, 255)
            
            # Use the mask to extract the moving object from the current frame
            foreground = cv2.bitwise_and(frame, frame, mask=mask)
            
            # Use the inverse mask to extract the background from the white background
            white_bg = cv2.bitwise_and(white_background, white_background, mask=mask_inv)
            
            # Combine the foreground (moving object) with the white background
            result = cv2.add(foreground, white_bg)
            
            # Find contours to detect the moving object
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Flag to check if any object is currently centered
            object_centered = False

            # Iterate over the contours to draw bounding boxes
            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Filter small objects
                    x, y, w, h = cv2.boundingRect(contour)
                    # Draw a bounding box around the object
                    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
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
                            
                            # Generate image path with a consistent counter for unique names
                            image_name = f'detected_object_{image_counter}.jpeg'  # Changed extension to .jpeg
                            image_counter += 1  # Increment the counter
                            image_path = os.path.join(directory, image_name)
                            cv2.imwrite(image_path, cropped_result)  # Save as .jpeg without quality adjustment, default is 100
                            print(f"Cropped screenshot saved: {image_name}")       

                            # Save raw webcam footage to Archive directory
                            archive_image_name = f'webcam_footage_{image_counter}.jpeg'
                            archive_image_path = os.path.join(archive_directory, archive_image_name)
                            cv2.imwrite(archive_image_path, frame)  # Save the raw webcam footage
                            print(f"Raw webcam footage saved: {archive_image_name}")

                            # Add geolocation metadata to the image
                            add_gps_data(image_path, latitude, longitude)

                            # Log the image name in the CSV file (no latitude or longitude)
                            csv_writer.writerow([image_name])

                # Reset screenshot_taken flag when no object is centered
                if not object_centered:
                    screenshot_taken = False

                # Show the result
                self.video_view_signal.emit(result)
                
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Close the CSV file
        csv_file.close()

        # Ask for a directory to append the CSV content
        directory_to_append = input("Enter the name of the directory to append the CSV file: ")

        # Look for any CSV file in the given directory
        csv_files_in_dir = [f for f in os.listdir(directory_to_append) if f.endswith('.csv')]

        if csv_files_in_dir:
            csv_to_append = os.path.join(directory_to_append, csv_files_in_dir[0])  # Use the first CSV file found
            print(f"Appending to '{csv_to_append}'.")

            # Open the CSV file we just created, and the target CSV file to append to
            with open(csv_path, 'r') as source_csv, open(csv_to_append, 'a', newline='') as target_csv:
                target_writer = csv.writer(target_csv)
                source_reader = csv.reader(source_csv)
                next(source_reader)  # Skip the header row if needed
                for row in source_reader:
                    target_writer.writerow(row)
            print(f"Appended the CSV content to {csv_to_append}")
        else:
            print(f"No CSV file found in the directory '{directory_to_append}'.")
        # Ask the user if they want to save images to the cloud
        save_to_cloud = input("Would you like to save the images to the cloud? (yes/no): ").lower()

        if save_to_cloud == 'yes':

            # Firebase configuration
            config = {
                "apiKey": "AIzaSyAo-4kZ6do3q5rJoWaifse6MgCvCkPspcc",
                "authDomain": "craftconveyortesting.firebaseapp.com",
                "databaseURL": "https://craftconveyortesting-default-rtdb.firebaseio.com",
                "projectId": "craftconveyortesting",
                "storageBucket": "craftconveyortesting.appspot.com",
                "messagingSenderId": "1054739246697",
                "appId": "1:1054739246697:web:4b8bd7512a4a48c0d13cf8",
                "measurementId": "G-PERCW5TSQJ"
            }

            firebase = pyrebase.initialize_app(config)
            storage = firebase.storage()

            # Save all the images saved during the program to the cloud
            for image_name in os.listdir(directory):
                if image_name.endswith('.jpeg'):  # Assuming the saved images are in .jpeg format
                    path_local = os.path.join(directory, image_name)
                    path_on_cloud = f"images/{image_name}"
                    storage.child(path_on_cloud).put(path_local)
                    print(f"Uploaded {image_name} to the cloud.")

        # Release the webcam and close the windows
        cap.release()
        cv2.destroyAllWindows()
        # Release the webcam and close the windows
        cap.release()
        cv2.destroyAllWindows()
    

    
    


