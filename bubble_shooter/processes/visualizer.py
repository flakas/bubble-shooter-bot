import queue
from bubble_shooter.visualizer import Visualizer

class PlayerActionVisualizerProcess:
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
    visualizer = PlayerActionVisualizerProcess(config['game_board_width'], config['agent_move_size'], my_queue)
    visualizer.start()
    visualizer.work()


