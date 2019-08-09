import numpy as np
import pickle
import os

class Pretrainer:
    def __init__(self, gameplay_filename):
        self.gameplay_filename = gameplay_filename

    def pretrain(self, agent, only_fill_memory=False):
        step = 0
        episode = 0
        experiences = 0
        agent.start_pretraining()
        print(f'[PRETRAINER] Starting pretraining')
        for (state, action, reward, next_state, done) in self.get_experiences():
            experiences += 1
            step += 1
            agent.remember(state, action, reward, next_state, done)
            #if only_fill_memory:
                #continue
            agent.after_step(step)
            if done:
                agent.after_episode(episode)
                step = 0
                episode += 1
        agent.stop_pretraining()
        print(f'[PRETRAINER] Pretrained with {experiences} experiences')

    def store_experience(self, experience):
        print(f'[PRETRAINER] Experience storage in pretrainer is disabled')
        return

        with open(self.gameplay_filename, 'ab') as f:
            pickle.dump(experience, f)

    def get_experiences(self):
        if not os.path.isfile(self.gameplay_filename):
            return []
        with open(self.gameplay_filename, 'rb') as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    return

