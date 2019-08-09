class Convolutional:
    def __init__(self, input_shape, output_shape, learning_rate):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.name = 'convolutional_c32x8_c64x4_c128x4_d512'

    def build(self):
        import tensorflow as tf

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=[8, 8], strides=[4, 4], padding='valid', activation='relu'), input_shape=self.input_shape)
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=[4, 4], strides=[2, 2], padding='valid', activation='relu'))
        model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=[4, 4], strides=[2, 2], padding='valid', activation='relu'))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(512, activation='relu'))
        model.add(tf.keras.layers.Dense(self.output_shape, activation='linear'))

        model.compile(
                loss='mse',
                optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate),
                metrics=['accuracy'])
        return model

