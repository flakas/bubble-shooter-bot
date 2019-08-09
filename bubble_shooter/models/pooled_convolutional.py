import tensorflow as tf
from bubble_shooter.utils import huber_loss_mean

class PooledConvolutional:
    def __init__(self, input_shape, output_shape, learning_rate):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.name = 'pooled_convolutional_c512x4_px2_c256x4'

    def build(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(512, (4, 4), input_shape=self.input_shape, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.2))
        # model.add(tf.keras.layers.SpatialDropout2D(0.1))

        model.add(tf.keras.layers.Conv2D(256, (4, 4), kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.2))
        # model.add(tf.keras.layers.SpatialDropout2D(0.1))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(self.output_shape, activation='linear', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))

        model.compile(
                loss=huber_loss_mean,
                optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate),
                metrics=['accuracy'])
        return model

