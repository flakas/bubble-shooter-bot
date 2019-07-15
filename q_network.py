import tensorflow as tf

class QNetwork:
    def __init__(self, actions):
        self.actions = actions

    def build_network(self):
        self.model = tf.keras.layers.Sequential([
            tf.keras.layers.Conv2D(
                filters = 32,
                kernel_size = [8, 8],
                strides = [4, 4],
                padding = 'valid'
            ),
            tf.keras.layers.Conv2D(
                filters = 64,
                kernel_size = [4, 4],
                strides = [2, 2],
                padding = 'valid',
            ),
            tf.keras.layers.Conv2D(
                kernles = 128,
                kernel_size = [4, 4],
                strides = [2, 2],
                padding = 'valid'
            ),
            tf.keras.layers.BatchNormalization(
                epsilon = 1e-5
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                units = 512,
                activation = 'relu'
            ),
            tf.keras.layers.Dense(
                units = actions,
                activation = 'none'
            )
        ])

    def prepare_for_training(self, model):
        self.model.compile(
            optimizer='rmsprop',
            loss='mean_squared_error'
        )

    def load_model_from_checkpoint(self, checkpoint_dir):
        self.build_model()
        self.model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
        self.model.build(tf.TensorShape([1, None]))

    def train(self):
        pass

    def predict(self):
        pass
