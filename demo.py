from vision import SeleniumSource, Vision
from game import Game
from prioritized_memory import Memory
from selenium_browser import SeleniumBrowser
import queue
import multiprocessing
from multiprocessing import Process, Queue
from visualizer import Visualizer

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
        self.agent = Agent(
                state_size=GAME_BOARD_X*GAME_BOARD_Y*GAME_BOARD_DEPTH,
                action_size=config['action_size'],
                move_size=config['move_size'],
                memory=self.memory,
                epsilon=config['epsilon'],
                gamma=config['gamma'],
                learning_rate=config['learning_rate'],
                batch_size=config['batch_size'],
                update_target_frequency=config['target_update_frequency'],
                replay_frequency=config['replay_frequency'],
                name=f"dueling1_mse_vsinit_{config['epsilon']}eps_{config['gamma']}gamma_{config['learning_rate']}lr_{config['replay_frequency']}refr_{config['target_update_frequency']}upfr_{config['memory_alpha']}memal_{config['batch_size']}bs_normbinaryrewards_parsedstate_onlycurnext")
        self.episodes_seen = 0

    def send_message(self, worker, message):
        self.worker_queues[worker].put_nowait(message)

    def work(self):
        try:
            while True:
                message = self.my_queue.get(block=True, timeout=60)
                if message['command'] == 'act_with_stats':
                    self.send_message(message['worker'], self.act_with_stats(message['state']))
                else:
                    raise Exception('Unknown message', message)
        except queue.Empty:
            return False

    def act_with_stats(self, state):
        return self.agent.act_with_stats(state)

def agent_worker(config, my_queue, worker_queues):
    from agent import Agent
    agent = AgentProcess(config, my_queue, worker_queues)
    return agent.work()

class PlayerActor:

    def __init__(self, agent_queue, my_queue, visualizer_queue, worker_name):
        self.agent_queue = agent_queue
        self.my_queue = my_queue
        self.worker_name = worker_name
        self.visualizer_queue = visualizer_queue

    def send_to_agent(self, message={}):
        message['worker'] = self.worker_name
        self.agent_queue.put_nowait(message)

    def send_to_visualizer(self, message={}):
        message['worker'] = self.worker_name
        self.visualizer_queue.put_nowait(message)

    def get_from_agent(self):
        return self.my_queue.get(block=True)

    def start(self, action_size, move_size):
        self.selenium = SeleniumBrowser(headless=False)
        self.selenium_source = SeleniumSource(self.selenium)
        self.vision = Vision(self.selenium_source, templates_path='templates/')
        self.controller = self.selenium
        self.game = Game(self.vision, self.controller)

        self.selenium.setup()

    def stop(self):
        self.selenium.cleanup()

    def play(self, episodes, steps):
        episode_rewards = []
        for e in range(episodes):
            state = self.game.get_state()
            total_reward = 0
            steps_taken = 0

            actions_taken = []
            for time_t in range(steps):
                steps_taken += 1
                self.send_to_agent({'command': 'act_with_stats', 'state': state})
                response = self.get_from_agent()
                action = response['recommended_action']
                evaluations = response['action_evaluations']
                self.send_to_visualizer({'command': 'show_evaluations', 'action': action, 'evaluations': evaluations})
                actions_taken.append(action)

                reward = self.game.perform_move(action, 400)
                total_reward += reward
                next_state = self.game.get_state()
                done = self.game.is_finished()

                print(f'[AGENT] Episode: {e}/{episodes}, step: {time_t}/{steps}, action: {action}, reward: {reward}/{total_reward}, done: {done}')

                state = next_state

                if done:
                    print(f'[AGENT] episode: {e}/{episodes}, score: {total_reward}')
                    self.game.restart_the_game()
                    break

def player_worker(config, agent_queue, my_queue, visualizer_queue, my_name):
    player = PlayerActor(agent_queue, my_queue, visualizer_queue, my_name)
    player.start(config['action_size'], config['move_size'])
    player.play(config['episodes'], config['steps'])
    player.stop()

class PlayerActionVisualizerActor:
    def __init__(self, action_size, move_size, my_queue):
        self.action_size = action_size
        self.move_size = move_size
        self.my_queue = my_queue
        self.timeout = 100

    def start(self):
        self.visualizer = Visualizer(self.action_size, self.move_size, self.timeout)

    def work(self):
        old_message = None
        while True:
            try:
                message = self.my_queue.get_nowait()
            except queue.Empty:
                if old_message == None:
                    continue
                message = old_message

            if message['command'] == 'show_evaluations':
                self.visualizer.show_evaluations(message['action'], message['evaluations'])
            else:
                raise Exception('Unknown message', message)

            old_message = message

def visualizer_worker(config, my_queue):
    visualizer = PlayerActionVisualizerActor(config['action_size'], config['move_size'], my_queue)
    visualizer.start()
    visualizer.work()

class GameplaySupervisorActor:
    def __init__(self, config, total_players=2):
        self.total_players = total_players
        self.agent_queue = Queue()
        self.worker_queues = [Queue() for _ in range(total_players)]
        self.visualizer_queues = [Queue() for _ in range(total_players)]
        self.config = config
        self.agent = Process(target=agent_worker, args=(config, self.agent_queue, self.worker_queues))
        self.workers = [Process(target=player_worker, args=(config, self.agent_queue, self.worker_queues[name], self.visualizer_queues[name], name)) for name in range(total_players)]
        self.visualizers = [Process(target=visualizer_worker, args=(config, self.visualizer_queues[name])) for name in range(total_players)]

    def start(self):
        self.agent.start()
        for visualizer in self.visualizers:
            visualizer.start()
        for worker in self.workers:
            worker.start()

    def wait_to_finish(self):
        for worker in self.workers:
            worker.join()
        self.agent.terminate()
        for visualizer in self.visualizers:
            visualizer.terminate()

TOTAL_PLAYERS = 1
configurations = [
        { 'epsilon': 0.99, 'gamma': 0.9, 'learning_rate': 0.00025, 'replay_frequency': 4, 'target_update_frequency': 1000, 'memory_epsilon': 0.01, 'memory_alpha': 0.6, 'memory_size': 20000, 'batch_size': 32, 'episodes': 500, 'steps': 500, 'action_size': 560, 'move_size': 35},
]

if __name__ == '__main__':
    for config in configurations:
        multiprocessing.set_start_method('spawn', force=True)
        supervisor = GameplaySupervisorActor(config, total_players=TOTAL_PLAYERS)
        supervisor.start()
        supervisor.wait_to_finish()
