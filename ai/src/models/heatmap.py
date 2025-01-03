import tensorflow as tf
import keras
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2
from scipy.ndimage import zoom
from keras.api.applications import resnet_v2

from convNextModel import ConvNextModel
try:
  sys.path.append(str(Path('.').resolve().parent))
  import datasets.tusayan_whiteware as tusayan_whiteware
except Exception as e:
  print(e)


image_dim = 224
num_classes = 7
batch_size = 32
epochs = 55
epochs_head = 15
l2_constant = 0.02
model_path = Path('.').resolve() / 'trained_models' / ("ResNet152V2" + ".keras")

"""
(train_dataset, test_dataset) = tusayan_whiteware.load_data(
   image_dimension=(image_dim, image_dim),
   training_split=(0.8,0.0,0.2),
   batch_size=batch_size
   )
"""
model = keras.saving.load_model(str(model_path))

#class_names = train_dataset.class_names

#model.summary()

conv_output = model.get_layer("post_relu").output
pred_output = model.get_layer("predictions").output
heatmap_model = keras.models.Model(model.input, outputs=[conv_output, pred_output])

#heatmap_model.summary()
image = cv2.imread('../../image_data/images/CD_Black_Mesa_044.jpg')
image = cv2.resize(image, (image_dim, image_dim))
#image = keras.utils.img_to_array(image)
#image = np.expand_dims(image, axis=0).astype(np.float32)
image = resnet_v2.preprocess_input(image)

predict_image = np.expand_dims(image, axis=0).astype(np.float32)
conv, pred = heatmap_model.predict(predict_image)

target = np.argmax(pred, axis=1).squeeze()
w, b = model.get_layer("predictions").weights
weights = w[:, target].numpy()
heatmap = conv.squeeze() @ weights

scale = 224 / 7
plt.figure(figsize=(12, 12))
plt.imshow(image)
plt.imshow(zoom(heatmap, zoom=(scale, scale)), cmap="jet", alpha=0.5)
plt.savefig('heatmap.png', bbox_inches='tight')
plt.show()