import os
import numpy as np
from collections import deque
import pickle

class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)

        indexes = np.random.choice(
                np.arange(buffer_size),
                size = batch_size,
                # replace = False
                )

        return [self.buffer[i] for i in indexes]

    def persist_to_file(self, filename):
        # return # disable for now
        with open(filename, 'wb') as f:
            pickle.dump(self.buffer, f, pickle.HIGHEST_PROTOCOL)
            print(f'Saved {len(self.buffer)} snapshots')

    def load_from_file(self, filename):
        if not os.path.isfile(filename):
            return False

        with open(filename, 'rb') as f:
            self.buffer = pickle.load(f)
            print(f'Loaded {len(self.buffer)} snapshots')

    def has_enough_samples(self, num_samples):
        return len(self.buffer) >= num_samples
