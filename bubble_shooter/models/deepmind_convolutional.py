import tensorflow as tf
from bubble_shooter.utils import huber_loss_mean

class DeepmindConvolutional:
    def __init__(self, input_shape, output_shape, learning_rate):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.name = 'deepmind_convolutional_c32x8_c64x4_c128x4_d512'

    def build(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(32, (5, 5), strides=(4, 4), activation='relu', input_shape=self.input_shape, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(tf.keras.layers.Conv2D(64, (4, 4), strides=(3, 3), activation='relu', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=512, activation='relu', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(tf.keras.layers.Dense(units=self.output_shape, activation='linear', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))

        model.compile(
            loss=huber_loss_mean,
            optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate),
            metrics=['accuracy'])

        return model
