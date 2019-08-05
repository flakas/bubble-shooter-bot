import random
import numpy
import os
import pickle

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros( 2*capacity - 1 )
        self.data = numpy.zeros( capacity, dtype=object )
        self.samples_seen = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        self.samples_seen += 1

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

    def count(self):
        return self.samples_seen if self.capacity > self.samples_seen else self.capacity

class Memory:   # stored as ( s, a, r, s_ ) in SumTree

    def __init__(self, capacity, epsilon, alpha):
        self.tree = SumTree(capacity)
        self.epsilon = epsilon
        self.alpha = alpha

    def get_priority(self, error):
        return (error + self.epsilon) ** self.alpha

    def add(self, error, experience):
        priority = self.get_priority(error)
        self.tree.add(priority, experience) 

    def sample(self, number_of_experiences):
        batch = []
        segment = self.tree.total() / number_of_experiences

        for i in range(number_of_experiences):
            a = segment * i
            b = segment * (i + 1)

            experience = random.uniform(a, b)
            (index, priority, data) = self.tree.get(experience)
            batch.append( (index, data) )

        return batch

    def update(self, index, error):
        priority = self.get_priority(error)
        self.tree.update(index, priority)

    def has_enough_samples(self, num_samples):
        return self.tree.count() >= num_samples

    def persist_to_file(self):
        print('[MEMORY] Memory persisting disabled')
        return

    def load_from_file(self):
        print('[MEMORY] Memory persisting disabled')
        return
