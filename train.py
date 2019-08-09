from bubble_shooter.process_supervisors.train import TrainingSupervisor

TOTAL_TRAINERS = 24
configurations = [
        { 'epsilon': 0.99, 'gamma': 0.9, 'learning_rate': 0.00025, 'replay_frequency': 4, 'target_update_frequency': 1000, 'memory_epsilon': 0.01, 'memory_alpha': 0.6, 'memory_size': 10000, 'batch_size': 32, 'episodes': 500, 'steps': 500, },
]

if __name__ == '__main__':
    for config in configurations:
        supervisor = TrainingSupervisor(config, total_trainers=TOTAL_TRAINERS)
        supervisor.start()
        supervisor.wait_to_finish()
