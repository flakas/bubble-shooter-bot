from vision import SeleniumSource, Vision
from game import Game
from prioritized_memory import Memory
import numpy as np
from selenium_browser import SeleniumBrowser
import logging
import queue
import multiprocessing
from multiprocessing import Process, Queue
from pretrainer import Pretrainer

#logging.basicConfig(level=logging.DEBUG)

GAME_BOARD_DIMENSION = 64
COLOR_SPACE = 3
GAME_BOARD_X = 35
GAME_BOARD_Y = 15
GAME_BOARD_DEPTH = 4

class AgentProcess:
    def __init__(self, config, my_queue, worker_queues):
        import tensorflow
        from agent import Agent
        self.config = config
        self.my_queue = my_queue
        self.worker_queues = worker_queues
        self.memory = Memory(config['memory_size'], epsilon=config['memory_epsilon'], alpha=config['memory_alpha'])
        self.memory.load_from_file()
        self.pretrainer = None # Pretrainer('observed_episodes.pickle')
        self.agent = Agent(
                state_size=GAME_BOARD_X*GAME_BOARD_Y*GAME_BOARD_DEPTH,
                action_size=560,
                move_size=35,
                memory=self.memory,
                epsilon=config['epsilon'],
                gamma=config['gamma'],
                learning_rate=config['learning_rate'],
                batch_size=config['batch_size'],
                update_target_frequency=config['target_update_frequency'],
                replay_frequency=config['replay_frequency'],
                name=f"dueling1_mse_vsinit_{config['epsilon']}eps_{config['gamma']}gamma_{config['learning_rate']}lr_{config['replay_frequency']}refr_{config['target_update_frequency']}upfr_{config['memory_alpha']}memal_{config['batch_size']}bs_normbinaryrewards_parsedstate_onlycurnext",
                pretrainer=self.pretrainer)
        self.agent.pretrain()
        self.episodes_seen = 0

    def send_message(self, worker, message):
        self.worker_queues[worker].put_nowait(message)

    def work(self):
        try:
            while True:
                message = self.my_queue.get(block=True, timeout=60)
                if message['command'] == 'act':
                    self.send_message(message['worker'], self.act(message['state']))
                elif message['command'] == 'remember':
                    self.remember(message['state'], message['action'], message['reward'], message['next_state'], message['done'])
                elif message['command'] == 'after_step':
                    self.after_step(message['step'])
                elif message['command'] == 'after_episode':
                    self.after_episode(message['episode'])
                elif message['command'] == 'remember_episode_rewards':
                    self.remember_episode_rewards(
                        message['total_reward'],
                        message['min_reward'],
                        message['average_reward'],
                        message['max_reward'],
                        message['episode_action_variance'],
                        message['steps_taken'])
                else:
                    raise Exception('Unknown message', message)
        except queue.Empty:
            return False

    def act(self, state):
        return self.agent.act(state)

    def remember(self, state, action, reward, next_state, done):
        self.agent.remember(state, action, reward, next_state, done)

    def remember_episode_rewards(self, *rewards):
        self.agent.remember_episode_rewards(*rewards)

    def after_step(self, step):
        self.agent.after_step(step)

    def after_episode(self, episode):
        self.agent.after_episode(episode)
        self.episodes_seen += 1
        if self.episodes_seen % 50 == 0:
            self.memory.persist_to_file()

def agent_worker(config, my_queue, worker_queues):
    from agent import Agent
    agent = AgentProcess(config, my_queue, worker_queues)
    return agent.work()

class TrainerActor:

    def __init__(self, agent_queue, my_queue, worker_name):
        self.agent_queue = agent_queue
        self.my_queue = my_queue
        self.worker_name = worker_name

    def send_to_agent(self, message={}):
        message['worker'] = self.worker_name
        self.agent_queue.put_nowait(message)

    def get_from_agent(self):
        return self.my_queue.get(block=True)

    def start(self):
        self.selenium = SeleniumBrowser()
        self.selenium_source = SeleniumSource(self.selenium)
        self.vision = Vision(self.selenium_source, templates_path='templates/')
        self.controller = self.selenium
        self.game = Game(self.vision, self.controller)

        self.selenium.setup()

    def stop(self):
        self.selenium.cleanup()

    def train(self, episodes, steps, minibatch_size, replay_frequency=1):
        episode_rewards = []
        for e in range(episodes):
            state = self.game.get_state()
            total_reward = 0
            steps_taken = 0

            actions_taken = []
            for time_t in range(steps):
                steps_taken += 1
                self.send_to_agent({'command': 'act', 'state': state})
                action = self.get_from_agent()
                actions_taken.append(action)

                reward = self.game.perform_move(action, 400)
                total_reward += reward
                next_state = self.game.get_state()
                done = self.game.is_finished()

                self.send_to_agent({'command': 'remember', 'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'done': done})
                self.send_to_agent({'command': 'after_step', 'step': time_t})
                print(f'[AGENT] Episode: {e}/{episodes}, step: {time_t}/{steps}, action: {action}, reward: {reward}/{total_reward}, done: {done}')

                state = next_state

                if done:
                    print(f'[AGENT] episode: {e}/{episodes}, score: {total_reward}')
                    self.game.restart_the_game()
                    break


            self.send_to_agent({'command': 'after_episode', 'episode': e})

            episode_rewards.append(total_reward)
            episode_action_variance = np.var(actions_taken)
            self.send_to_agent({
                'command': 'remember_episode_rewards',
                'total_reward': total_reward,
                'min_reward': min(episode_rewards),
                'average_reward': sum(episode_rewards)/len(episode_rewards),
                'max_reward': max(episode_rewards),
                'episode_action_variance': episode_action_variance,
                'steps_taken': steps_taken
            })

def trainer_worker(config, agent_queue, my_queue, my_name):
    trainer = TrainerActor(agent_queue, my_queue, my_name)
    trainer.start()
    trainer.train(config['episodes'], config['steps'], config['batch_size'], config['replay_frequency'])
    trainer.stop()

class TrainingSupervisorActor:
    def __init__(self, config, total_trainers=2):
        self.total_trainers = total_trainers
        self.agent_queue = Queue()
        self.worker_queues = [Queue() for _ in range(total_trainers)]
        self.config = config
        self.agent = Process(target=agent_worker, args=(config, self.agent_queue, self.worker_queues))
        self.workers = [Process(target=trainer_worker, args=(config, self.agent_queue, self.worker_queues[name], name)) for name in range(total_trainers)]

    def start(self):
        self.agent.start()
        for worker in self.workers:
            worker.start()

    def wait_to_finish(self):
        for worker in self.workers:
            worker.join()
        self.agent.terminate()

TOTAL_TRAINERS = 24
configurations = [
        { 'epsilon': 0.99, 'gamma': 0.9, 'learning_rate': 0.00025, 'replay_frequency': 4, 'target_update_frequency': 1000, 'memory_epsilon': 0.01, 'memory_alpha': 0.6, 'memory_size': 20000, 'batch_size': 32, 'episodes': 500, 'steps': 500, },
]

if __name__ == '__main__':
    for config in configurations:
        multiprocessing.set_start_method('spawn', force=True)
        supervisor = TrainingSupervisorActor(config, total_trainers=TOTAL_TRAINERS)
        supervisor.start()
        supervisor.wait_to_finish()
