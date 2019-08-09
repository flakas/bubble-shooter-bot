import multiprocessing
from multiprocessing import Process, Queue
from bubble_shooter.processes.agent import agent_worker
from bubble_shooter.processes.trainer import trainer_worker

class TrainingSupervisor:
    def __init__(self, config, total_trainers=2):
        multiprocessing.set_start_method('spawn', force=True)

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
