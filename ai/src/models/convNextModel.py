import keras
from keras.api import layers
from keras.api.applications import ConvNeXtBase, ConvNeXtXLarge
import keras.api
import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import sys
import tensorflow as tf

try:
  sys.path.append(str(Path('.').resolve().parent))
  import datasets.tusayan_whiteware as tusayan_whiteware
except Exception as e:
  print(e)


class ConvNextModel:
    def __init__(
          self,
          image_dim,
          batch_size,
          num_classes,
          epochs,
          epochs_head,
          l2_constant,
          model_path,
          output_path=None,
          use_step_decay=True
    ):
        self.image_dim = image_dim
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.epochs = epochs
        self.epochs_head = epochs_head
        self.l2_constant = l2_constant
        self.model_path = model_path
        self.output_path = output_path
        self.use_step_decay = use_step_decay
        self.history = None
        self.model = self.build()
    
    def build(self):
       input_layer = keras.Input(shape=(self.image_dim, self.image_dim, 3))
       self.base_model = ConvNeXtBase(include_top=False,
                                 input_tensor=input_layer)
       head_model = self.base_model.output
       head_model = layers.GlobalAveragePooling2D()(head_model)
       head_model = layers.Flatten()(head_model)
       head_model = layers.Dropout(0.5)(head_model)
       head_model = layers.Dense(512, 
                                 activation="relu",
                                 kernel_regularizer=keras.regularizers.L2(self.l2_constant))(head_model)
       head_model = layers.Dense(self.num_classes,
                                 activation="softmax")(head_model)
       return keras.models.Model(inputs=self.base_model.input, outputs=head_model)
    
    def evaluate(self, test_dataset):
        print("Evaluating Model...")
        y_true = []
        y_pred = []
        for image_batch, label_batch in test_dataset:
            y_true.append(label_batch)
            preds = self.model.predict(image_batch)
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

        """
        # plot the accuracy
        plt.style.use("ggplot")
        plt.figure(figsize=(11, 8))
        plt.plot(np.arange(0, self.epochs), self.history.history["accuracy"], label="Train accuracy")
        plt.plot(np.arange(0, self.epochs), self.history.history["val_accuracy"], label="Test accuracy")
        plt.title("model" + " Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Accuracy")
        plt.legend()
        if self.output_path:
            acc_path = self.output_path / 'accuracy_plot.png'
            plt.savefig(acc_path)

        # plot the training loss
        plt.style.use("ggplot")
        plt.figure(figsize=(11, 8))
        plt.plot(np.arange(0, self.epochs), self.history.history["loss"], label="Train loss")
        plt.plot(np.arange(0, self.epochs), self.history.history["val_loss"], label="Test loss")
        plt.title("model" + " Training Loss")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        #plt.ylim(0, 2.0)
        plt.legend()
        if self.output_path:
            loss_path = self.output_path / 'loss_plot.png'
            plt.savefig(loss_path)
        else:
            plt.show()
        """

    def load_model(self):
        self.model = keras.models.load_model(self.model_path)

    def step_decay(self, epoch):
        initAlpha=.0020
        if epoch <= 20:
            alpha=initAlpha
        elif epoch <= 40:
            alpha=initAlpha*0.75
        elif epoch <= 60:
            alpha=initAlpha*0.50
        elif epoch <= 80:
            alpha=initAlpha*0.25
        elif epoch <=100:
            alpha=initAlpha*0.175
        elif epoch <= 125:
            alpha=initAlpha*0.10
        else:
            alpha=initAlpha*0.05
        return float(alpha)

    def train(self, train_dataset, val_dataset):
        for layer in self.base_model.layers:
            layer.trainable = False
        
        opt = keras.optimizers.RMSprop(learning_rate=0.01)
        self.model.compile(loss="categorical_crossentropy",
                           optimizer=opt,
                           metrics=["accuracy"])
        
        if self.use_step_decay:
            callbacks = [keras.callbacks.LearningRateScheduler(self.step_decay)]
        
        print("[INFO] Training Head...")

        self.model.fit(train_dataset,
                       batch_size=self.batch_size,
                       epochs=self.epochs_head,
                       validation_data=val_dataset,
                       callbacks=callbacks)
        
        print("[INFO] Recompiling Model...")

        for layer in self.model.layers:
            layer.trainable = True
        
        opt = keras.optimizers.SGD(learning_rate=0.005)
        self.model.compile(loss="categorical_crossentropy",
                           optimizer=opt,
                           metrics=["accuracy"])
        
        print("[INFO] Model Recompiled...")

        callbacks = [
            keras.callbacks.ModelCheckpoint(filepath=self.model_path,
                                            monitor="val_accuracy",
                                            save_best_only=True)
        ]
        if self.use_step_decay:
            callbacks.append(keras.callbacks.LearningRateScheduler(self.step_decay))
        
        print("[INFO] Fine-Tuning Model...")

        self.history = self.model.fit(train_dataset,
                                      batch_size=self.batch_size,
                                      epochs=self.epochs,
                                      validation_data=val_dataset,
                                      callbacks=callbacks)