import tensorflow as tf
import keras
import os;os.environ["TF_USE_LEGACY_KERAS"]="1"
import shutil
import keras.applications as modelList
from keras import layers


class LayerScale(layers.Layer):
    """Layer scale module.

    References:
      - https://arxiv.org/abs/2103.17239

    Args:
      init_values (float): Initial value for layer scale. Should be within
        [0, 1].
      projection_dim (int): Projection dimensionality.

    Returns:
      Tensor multiplied to the scale.
    """

    def __init__(self, init_values, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, input_shape):
        self.gamma = tf.Variable(
            self.init_values * tf.ones((self.projection_dim,))
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "init_values": self.init_values,
                "projection_dim": self.projection_dim,
            }
        )
        return config

model_path = '../../image_data/Set_1/models/ConvNetBase_1_3.keras'


keras_model = tf.keras.models.load_model(
    model_path,
    custom_objects={"LayerScale": LayerScale},
)
#keras_model.export('trained_models/convNext')
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()

with open('trained_models/downgraded_convNext.tflite', 'wb') as f:
    f.write(tflite_model)

#shutil.rmtree('trained_models/convNext')