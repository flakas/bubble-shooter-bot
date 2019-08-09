from bubble_shooter.process_supervisors.training import TrainingSupervisor
from bubble_shooter.state_preprocessors.all_color import AllColor as AllColorPreprocessor
from bubble_shooter.models.dueling_inception import DuelingInception

TOTAL_TRAINERS = 24
configurations = [
    {
        'epsilon': 0.99,
        'gamma': 0.9,
        'learning_rate': 0.00025,
        'replay_frequency': 4,
        'target_update_frequency': 1000,
        'memory_epsilon': 0.01,
        'memory_alpha': 0.6,
        'memory_size': 10000,
        'batch_size': 32,
        'episodes': 500,
        'steps': 500,
        'agent_move_size': 35,
        'game_board_width': 560,
        'state_preprocessor': AllColorPreprocessor(),
        'model_builder': DuelingInception,
    },
]

if __name__ == '__main__':
    for config in configurations:
        supervisor = TrainingSupervisor(config, total_trainers=TOTAL_TRAINERS)
        supervisor.start()
        supervisor.wait_to_finish()
