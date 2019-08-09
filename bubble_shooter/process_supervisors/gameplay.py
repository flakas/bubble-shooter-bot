import multiprocessing
from multiprocessing import Process, Queue
from bubble_shooter.processes.agent import agent_worker
from bubble_shooter.processes.player import player_worker
from bubble_shooter.processes.visualizer import visualizer_worker

class GameplaySupervisor:
    def __init__(self, config, total_players=2):
        multiprocessing.set_start_method('spawn', force=True)

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


