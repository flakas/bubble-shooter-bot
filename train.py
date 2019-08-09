import multiprocessing
from multiprocessing import Process, Queue
from bubble_shooter.processes.agent import agent_worker
from bubble_shooter.processes.trainer import trainer_worker

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
        { 'epsilon': 0.99, 'gamma': 0.9, 'learning_rate': 0.00025, 'replay_frequency': 4, 'target_update_frequency': 1000, 'memory_epsilon': 0.01, 'memory_alpha': 0.6, 'memory_size': 10000, 'batch_size': 32, 'episodes': 500, 'steps': 500, },
]

if __name__ == '__main__':
    for config in configurations:
        multiprocessing.set_start_method('spawn', force=True)
        supervisor = TrainingSupervisorActor(config, total_trainers=TOTAL_TRAINERS)
        supervisor.start()
        supervisor.wait_to_finish()
