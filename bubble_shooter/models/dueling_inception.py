import tensorflow as tf

class DuelingInception:
    def __init__(self, input_shape, output_shape, learning_rate):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.name = 'duelinginception_2i16x64_c64_d64'

    def build(self):
        model = tf.keras.models.Sequential()

        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        first_layer = input_layer

        def inception_block(input_layer):
            first_branch = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same')(input_layer)
            second_branch = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same')(input_layer)
            second_branch = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(second_branch)
            third_branch = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same')(input_layer)
            third_branch = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same')(third_branch)
            fourth_branch = tf.keras.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_layer)
            fourth_branch = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same')(fourth_branch)

            return tf.keras.layers.concatenate([first_branch, second_branch, third_branch, fourth_branch])

        after_inception = inception_block(first_layer)
        after_inception = inception_block(after_inception)

        conv = tf.keras.layers.Conv2D(64, (1, 1), activation='relu')(concat)
        flatten = tf.keras.layers.Flatten()(conv)

        fc1 = tf.keras.layers.Dense(64, activation='relu')(flatten)
        dropout = tf.keras.layers.Dropout(0.7)(fc1)

        advantage = tf.keras.layers.Dense(self.output_shape)(fc1)
        value = tf.keras.layers.Dense(1)(fc1)

        advantage = tf.keras.layers.Lambda(lambda advantage: advantage - tf.reduce_mean(advantage, axis=-1, keep_dims=True))(advantage)
        value = tf.keras.layers.Lambda(lambda value: tf.tile(value, [1, self.output_shape]))(value)
        policy = tf.keras.layers.Add()([value, advantage])

        model = tf.keras.models.Model(inputs=[input_layer], outputs=[policy], name='agent_model')

        model.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.RMSprop(self.learning_rate),
            metrics=['accuracy'])

        return model

