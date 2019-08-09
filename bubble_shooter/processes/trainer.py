from bubble_shooter.selenium_browser import SeleniumBrowser
from bubble_shooter.vision import SeleniumSource, Vision
from bubble_shooter.state_preprocessors.all_color import AllColor as AllColorPreprocessor
from bubble_shooter.game import Game

class TrainerProcess:

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
        self.state_preprocessor = AllColorPreprocessor()
        self.game = Game(self.vision, self.controller, self.state_preprocessor)

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
    trainer = TrainerProcess(agent_queue, my_queue, my_name)
    trainer.start()
    trainer.train(config['episodes'], config['steps'], config['batch_size'], config['replay_frequency'])
    trainer.stop()
