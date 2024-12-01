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

from tusNetModel import TusNetModel
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
batch_size = 32
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

(train_dataset, test_dataset) = tusayan_whiteware.load_data(
   image_dimension=(image_dim, image_dim),
   training_split=(0.8,0.0,0.2),
   batch_size=batch_size,
   regenerate=True
   )

train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y),
                                  num_parallel_calls=tf.data.AUTOTUNE)

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