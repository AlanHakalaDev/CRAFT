import tensorflow as tf
import numpy as np
from pathlib import Path
import csv
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Define constants
IMAGE_DIM = 384
MODEL_PATH = Path('.') / 'convNext.tflite'
CLASS_NAMES = ['Kanaa', 'Black_Mesa', 'Sosi', 'Dogoszhi', 'Flagstaff', 'Tusayan', 'Kayenta']

# Get images from the specified directory
def load_dataset(directory, target_size):
    # Filter only .png files
    image_paths = [str(p) for p in Path(directory).glob('**/*.png') if p.is_file()]
    images = []
    for image_path in image_paths:
        # Load and preprocess image
        img = load_img(image_path, target_size=(target_size, target_size))
        # Normalize pixel values
        img_array = img_to_array(img)
        # Save image with its path
        images.append((img_array, image_path))  
    return images

# Start up TFLite
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

# Use our TFLite model to get results
def predict_with_tflite_model(interpreter, input_details, output_details, images):
    results = []
    for img_array, img_path in images:
        # Add batch dimension
        input_data = np.expand_dims(img_array, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        # Get the prediction
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        predicted_class = np.argmax(predictions)
        results.append((img_path, predicted_class, predictions))
    return results

def write_results_to_csv(results, class_names, output_csv):
    with open(output_csv, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write header (filename only)
        csv_writer.writerow(['Filename'])
        # Write data rows (filename + predicted class)
        for img_path, predicted_class, _ in results:
            csv_writer.writerow([Path(img_path).stem + ".png", class_names[predicted_class]])  # Use stem to get filenames only
    print(f"Results written to {output_csv}")
    
# Display a few results
def display_results(results, class_names, num_to_display=5):
    for i, (img_path, predicted_class, predictions) in enumerate(results[:num_to_display]):
        print(f"File: {img_path}")
        print(f"Predicted Class: {class_names[predicted_class]}")
        print(f"Prediction Probabilities: {predictions}")
        print("-" * 50)  # Separator for better readability

def classifySherds(dataPath, copyCSV=None):
    from shutil import copyfile

    # Find directory and prepare output CSV file
    DATASET_PATH = Path('.') / dataPath
    csv_name = dataPath + ".csv"
    OUTPUT_CSV = Path(DATASET_PATH) / csv_name

    # Load the TFLite model
    interpreter, input_details, output_details = load_tflite_model(MODEL_PATH)
    print("TFLite model loaded successfully.")

    # Load dataset
    dataset = load_dataset(DATASET_PATH, IMAGE_DIM)
    print(f"Loaded {len(dataset)} images from {DATASET_PATH}")

    # Process images with the TFLite model
    predictions = predict_with_tflite_model(interpreter, input_details, output_details, dataset)
    print("Inference complete.")

    # Write results to the local output CSV
    write_results_to_csv(predictions, CLASS_NAMES, OUTPUT_CSV)

    # Append results to the specified CSV if copyCSV is provided
    if copyCSV:
        try:
            copyCSV_path = Path(copyCSV)

            # Ensure the file exists
            if not copyCSV_path.exists():
                raise FileNotFoundError(f"The specified file '{copyCSV}' does not exist.")

            # Append results to the existing CSV
            with open(copyCSV_path, 'a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                for img_path, predicted_class, _ in predictions:
                    csv_writer.writerow([Path(img_path), CLASS_NAMES[predicted_class]])

            print(f"Results successfully appended to {copyCSV}")
        except Exception as e:
            print(f"Failed to append results to {copyCSV}: {e}")

    # Display a few results
    display_results(predictions, CLASS_NAMES, num_to_display=5)