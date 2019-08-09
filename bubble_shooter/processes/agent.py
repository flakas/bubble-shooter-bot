import queue
from bubble_shooter.prioritized_memory import Memory

COLOR_SPACE = 3
GAME_BOARD_X = 17
GAME_BOARD_Y = 15
GAME_BOARD_DEPTH = 8

class AgentProcess:
    def __init__(self, config, my_queue, worker_queues):
        # Import tensorflow within the class to avoid multiple memory allocation attempts
        import tensorflow
        from bubble_shooter.agent import Agent

        self.config = config
        self.my_queue = my_queue
        self.worker_queues = worker_queues
        self.memory = Memory(config['memory_size'], epsilon=config['memory_epsilon'], alpha=config['memory_alpha'])
        self.memory.load_from_file()
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
                name=f"dueling_inception16_mse_vsinit_{config['epsilon']}eps_{config['gamma']}gamma_{config['learning_rate']}lr_{config['replay_frequency']}refr_{config['target_update_frequency']}upfr_{config['memory_alpha']}memal_{config['batch_size']}bs_normbinaryrewards_parsedsmallstate_allcolors")
        self.episodes_seen = 0

    def send_message(self, worker, message):
        self.worker_queues[worker].put_nowait(message)

    def work(self):
        try:
            while True:
                message = self.my_queue.get(block=True, timeout=60)
                if message['command'] == 'act':
                    self.send_message(message['worker'], self.act(message['state']))
                elif message['command'] == 'act_with_stats':
                    self.send_message(message['worker'], self.act_with_stats(message['state']))
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

    def act_with_stats(self, state):
        return self.agent.act_with_stats(state)

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
    agent = AgentProcess(config, my_queue, worker_queues)
    return agent.work()