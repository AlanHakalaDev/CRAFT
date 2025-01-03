import tensorflow as tf
import keras
from keras.api.applications import ResNet152V2, resnet_v2
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from csv import reader
import matplotlib.pyplot as plt
import math
import os
from random import shuffle
from pathlib import Path
import sys

try:
  sys.path.append(str(Path('.').resolve().parent))
  import datasets.tusayan_whiteware as tusayan_whiteware
except Exception as e:
  print(e)


def preprocess(images, labels):
    return resnet_v2.preprocess_input(images), labels


def print_statistics(model, test_ds, hist):
    print("[INFO] evaluating test data for base model...")
    y_true = []
    y_pred = []
    for image_batch, label_batch in test_ds:
        y_true.append(label_batch)
        preds = model.predict(image_batch)
        y_pred.append(np.argmax(preds, axis=-1))
    correct_labels = tf.concat([item for item in y_true], axis = 0)
    predicted_labels = tf.concat([item for item in y_pred], axis = 0)
    class_report=classification_report(np.argmax(correct_labels, axis=1),
                                    predicted_labels,
                                    target_names=[label.name for label in tusayan_whiteware.SherdType])
    print(class_report)
    print("Confusion matrix")
    print([label.name for label in tusayan_whiteware.SherdType])
    con_mat=confusion_matrix(np.argmax(correct_labels, axis=1), predicted_labels)
    print(con_mat)

    # plot the accuracy
    plt.style.use("ggplot")
    plt.figure(figsize=(11, 8))
    plt.plot(np.arange(0, epochs), hist.history["accuracy"], label="Train accuracy")
    plt.plot(np.arange(0, epochs), hist.history["val_accuracy"], label="Test accuracy")
    plt.title("model" + " Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend()
    #plt.savefig(full_model_path + "_accuracy_plot.png")
    plt.show()

    # plot the training loss
    plt.style.use("ggplot")
    plt.figure(figsize=(11, 8))
    plt.plot(np.arange(0, epochs), hist.history["loss"], label="Train loss")
    plt.plot(np.arange(0, epochs), hist.history["val_loss"], label="Test loss")
    plt.title("model" + " Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend()
    #plt.savefig(full_model_path  + "_loss_plot.png")
    plt.show()


def step_decay(epoch):
    init_lr = 0.005
    drop = 0.5
    epochs_drop = 10
    lr = init_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lr


def train_model(model, train_dataset, val_dataset, transfer=True):   
    if transfer:
        # Prevent base model from training
        for layer in baseModel.layers:
            layer.trainable = False

        opt = keras.optimizers.RMSprop(learning_rate=0.005)

        model.compile(loss="categorical_crossentropy", 
                    optimizer=opt,
                    metrics=["accuracy"])

        # Add Learning rate scheduler to model callbacks
        #callbacks=[keras.callbacks.LearningRateScheduler(step_decay)]

        print("[INFO] training head...")
        model.fit(train_dataset.prefetch(tf.data.AUTOTUNE), 
                batch_size=batch_size,
                epochs=epochs_head,
                validation_split=0.1,
                validation_data=val_dataset.prefetch(tf.data.AUTOTUNE),)
                #callbacks=callbacks)

        # Allow base model to train along with head model for full training
        for layer in baseModel.layers[0:]:
            layer.trainable = True

        print("Re-compiling model...")

    # Have to rebuild optimizer for model recompile
    #opt = keras.optimizers.SGD(learning_rate=0.005)
    opt = keras.optimizers.RMSprop(learning_rate=0.005)
    model.compile(loss="categorical_crossentropy", 
                  optimizer=opt,
                  metrics=["accuracy"])
    print("Model Compiled")

    full_model_path = Path('.').resolve() / 'trained_models' / ("ResNet152V2" + ".keras")
    log_path = Path('.').resolve() / 'logs'
    if not log_path.exists():
        log_path.mkdir()
    tensorboard = keras.callbacks.TensorBoard(log_dir=str(log_path))
    callbacks=[keras.callbacks.LearningRateScheduler(step_decay),
               keras.callbacks.ModelCheckpoint(full_model_path, 
                                               monitor="val_accuracy",
                                               verbose=1, 
                                               save_best_only=True,
                                               save_weights_only=False, 
                                               mode="max",
                                               save_freq="epoch"),
                #keras.callbacks.EarlyStopping(patience=5),
                #tensorboard,
                ]

    print("Fine-Tuning Final Model...")
    hist = model.fit(train_dataset.prefetch(tf.data.AUTOTUNE), 
                        batch_size=batch_size, 
                        epochs=epochs,  
                        validation_data=val_dataset.prefetch(tf.data.AUTOTUNE),
                        callbacks=callbacks)

    model = keras.models.load_model(full_model_path)
    return hist


image_dir = Path('.').resolve().parents[1] / 'image_data'
num_classes = 7
image_dimension = 224
batch_size = 32
epochs_head = 10
epochs = 60
l2_constant = 0.02

data_augmentation = keras.Sequential([
    keras.layers.RandomRotation(factor=0.5, 
                                fill_mode="constant", 
                                fill_value=1.0),
    keras.layers.RandomZoom(height_factor=(-0.2, 0.2), 
                            width_factor=(-0.2, 0.2),
                            fill_mode="constant", 
                            fill_value=1.0),
])

(train_ds, test_ds) = tusayan_whiteware.load_data(
   image_dimension=(image_dimension, image_dimension),
   training_split=(0.8,0.0,0.2),
   batch_size=batch_size
   )

# Preprocess datasets for resnet
train_ds = train_ds.map(preprocess,
                        num_parallel_calls=tf.data.AUTOTUNE)
#val_ds = val_ds.map(preprocess)
test_ds = test_ds.map(preprocess,
                      num_parallel_calls=tf.data.AUTOTUNE)
# Augment training dataset
#train_ds = train_ds.prefetch(tf.data.AUTOTUNE).map(lambda x, y: (data_augmentation(x), y),
#                                                   num_parallel_calls=tf.data.AUTOTUNE)

input = keras.Input(shape=(image_dimension, image_dimension, 3))
# Create base model of ResNet
baseModel = ResNet152V2(weights="imagenet", include_top=False, 
                        input_tensor=input)
random_base_model = ResNet152V2(weights=None, include_top=True, 
                        input_tensor=input, classes=num_classes)
# Create head model from ResNet base model for transfer learning
headModel = baseModel.output
headModel = keras.layers.GlobalAveragePooling2D()(headModel)
headModel = keras.layers.Flatten()(headModel)
headModel = keras.layers.Dense(512, activation="relu", 
                               kernel_regularizer=keras.regularizers.L2(l2_constant))(headModel)
headModel = keras.layers.Dense(num_classes, activation="softmax")(headModel)

model = keras.models.Model(inputs=baseModel.input, outputs=headModel)

# Testing if ImageDataGenerator against augmentation layers
#Parameters for ImageDataGenerator
shift=0.0
zoom=0.3
# construct the image generator for data augmentation
#fill_mode is value put into empty spaces created by rotation or zooming; cval=1.0 means white
aug = ImageDataGenerator(rotation_range=180,
	horizontal_flip=False, vertical_flip=False, width_shift_range=shift, 
    height_shift_range=shift, zoom_range=zoom, fill_mode="constant",cval=1.0)

best_gen_augmented_model = train_model(model, 
                              train_dataset=aug.flow(train_ds.as_numpy_iterator()), 
                              val_dataset=test_ds)
"""
hist = train_model(model,
                    train_dataset=train_ds, 
                    val_dataset=test_ds)

print_statistics(model=model,
                 test_ds=test_ds,
                 hist=hist)
"""
hist = train_model(random_base_model,
                    train_dataset=train_ds, 
                    val_dataset=test_ds,
                    transfer=False)

print_statistics(model=random_base_model,
                 test_ds=test_ds,
                 hist=hist)