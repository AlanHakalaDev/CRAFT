import argparse
import keras
from keras.api import layers
import keras.api
import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator

from convNextModel import ConvNextModel
try:
  sys.path.append(str(Path('.').resolve().parent))
  import datasets.tusayan_whiteware as tusayan_whiteware
except Exception as e:
  print(e)


# Get file path for data output
parser = argparse.ArgumentParser()
parser.add_argument("output", help="file path to output location")
args = parser.parse_args()
if args.output:
    output_dir = Path(args.output)
    if not output_dir.exists():
        output_dir.mkdir()
    model_path = output_dir / 'trained_models/'
    if not model_path.exists():
        model_path.mkdir()
    results_path = output_dir / 'results/'
    if not results_path.exists():
       results_path.mkdir()

image_dim = 384
num_classes = 7
batch_size = 16
epochs = 100
epochs_head = 15
l2_constant = 0.02
model_path = model_path / ("convNext" + ".keras")

data_augmentation = keras.Sequential([
    layers.RandomRotation(factor=0.5,
                          fill_mode="constant", 
                          fill_value=1.0),
    layers.RandomZoom(height_factor=(-0.2, 0.2),
                      width_factor=(-0.2, 0.2),
                      fill_mode="constant", 
                      fill_value=1.0),
])

#Parameters for ImageDataGenerator
shift=0.0
zoom=0.3
# construct the image generator for data augmentation
#fill_mode is value put into empty spaces created by rotation or zooming; cval=1.0 means white
aug = ImageDataGenerator(rotation_range=180,
	horizontal_flip=False, vertical_flip=False, width_shift_range=shift, 
    height_shift_range=shift, zoom_range=zoom, fill_mode="constant",cval=1.0)

def augment_images(image, label):
  image = aug.random_transform(image)
  return image, label

(train_dataset, test_dataset) = tusayan_whiteware.load_data(
   image_dimension=(image_dim, image_dim),
   training_split=(0.8,0.0,0.2),
   batch_size=batch_size,
   regenerate=False
   )

train_generator = aug.flow_from_directory(
   "../../image_data/tusayan_whiteware/Training",
   target_size=(image_dim, image_dim),
   batch_size=batch_size,
   class_mode="categorical"
)

train_dataset = tf.data.Dataset.from_generator(
   lambda: train_generator,
   output_types=(tf.float32, tf.float32),
   output_shapes=((None, image_dim, image_dim, 3), (None, num_classes))
)

#train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y),
#                                  num_parallel_calls=tf.data.AUTOTUNE)

x_train = []
y_train = []

#for images, labels in train_dataset.take(-1):
#   numpy_image = images.numpy()
   
#images, labels = tuple(zip(*train_dataset))
#images = np.array(images)
#labels = np.array(labels)

#train_dataset = train_dataset.map(lambda x, y: (augment_images(x.numpy(), y)), 
#                                  num_parallel_calls=tf.data.AUTOTUNE)

model = ConvNextModel(image_dim=image_dim,
                      batch_size=batch_size,
                      num_classes=num_classes,
                      epochs=epochs,
                      epochs_head=epochs_head,
                      l2_constant=l2_constant,
                      model_path=model_path,
                      output_path=results_path)

model.train(train_dataset=train_dataset,
            val_dataset=test_dataset)

model.load_model()
model.evaluate(test_dataset=test_dataset)