from bubble_shooter.selenium_browser import SeleniumBrowser
from bubble_shooter.vision import SeleniumSource, Vision
from bubble_shooter.state_preprocessors.all_color import AllColor as AllColorPreprocessor
from bubble_shooter.game import Game

class PlayerProcess:
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

    def start(self, state_preprocessor):
        self.selenium = SeleniumBrowser()
        self.selenium_source = SeleniumSource(self.selenium)
        self.vision = Vision(self.selenium_source, templates_path='templates/')
        self.controller = self.selenium
        self.state_preprocessor = state_preprocessor
        self.game = Game(self.vision, self.controller, self.state_preprocessor)

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
    player = PlayerProcess(agent_queue, my_queue, visualizer_queue, my_name)
    player.start(config['state_preprocessor'])
    player.play(config['episodes'], config['steps'])
    player.stop()
