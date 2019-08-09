class Dueling:
    def __init__(self, input_shape, output_shape, learning_rate):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.name = 'dueling_c32x4_c64x2_c64x1_d64'

    def build(self):
        import tensorflow as tf
        from bubble_shooter.utils import huber_loss_mean

        model = tf.keras.models.Sequential()

        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        conv1 = tf.keras.layers.Conv2D(32, (4, 4), activation='relu')(input_layer)
        conv2 = tf.keras.layers.Conv2D(64, (2, 2), activation='relu')(conv1)
        conv3 = tf.keras.layers.Conv2D(64, (1, 1), activation='relu')(conv2)
        flatten = tf.keras.layers.Flatten()(conv3)

        fc1 = tf.keras.layers.Dense(64)(flatten)

        advantage = tf.keras.layers.Dense(self.output_shape)(fc1)
        value = tf.keras.layers.Dense(1)(fc1)

        advantage = tf.keras.layers.Lambda(lambda advantage: advantage - tf.reduce_mean(advantage, axis=-1, keep_dims=True))(advantage)
        value = tf.keras.layers.Lambda(lambda value: tf.tile(value, [1, self.output_shape]))(value)
        policy = tf.keras.layers.Add()([value, advantage])

        model = tf.keras.models.Model(inputs=[input_layer], outputs=[policy], name='agent_model')

        model.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            metrics=['accuracy'])

        return model

