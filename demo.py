from bubble_shooter.process_supervisors.gameplay import GameplaySupervisor

TOTAL_PLAYERS = 1
configurations = [
        { 'epsilon': 0.99, 'gamma': 0.9, 'learning_rate': 0.00025, 'replay_frequency': 4, 'target_update_frequency': 1000, 'memory_epsilon': 0.01, 'memory_alpha': 0.6, 'memory_size': 20000, 'batch_size': 32, 'episodes': 500, 'steps': 500, 'action_size': 560, 'move_size': 35},
]

if __name__ == '__main__':
    for config in configurations:
        supervisor = GameplaySupervisor(config, total_players=TOTAL_PLAYERS)
        supervisor.start()
        supervisor.wait_to_finish()
