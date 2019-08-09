import tensorflow as tf
from bubble_shooter.utils import huber_loss_mean

class SimpleDense:
    def __init__(self, input_shape, output_shape, learning_rate):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.name = 'simpledense_d500_d500'

    def build(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=self.input_shape))
        model.add(tf.keras.layers.Dense(500, activation='relu'))
        model.add(tf.keras.layers.Dense(500, activation='relu'))
        model.add(tf.keras.layers.Dense(self.output_shape, activation='linear'))
        model.compile(
                loss=huber_loss_mean,
                optimizer=tf.keras.optimizers.RMSprop(lr=self.learning_rate))
        return model
