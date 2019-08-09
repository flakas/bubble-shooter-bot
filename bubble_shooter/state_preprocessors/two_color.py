import numpy as np

class TwoColor:
    """
    Preprocesses vision state for the agent.
    Includes balls only of current and next colors, but not remaining ones
    """

    def __init__(self):
        self.width = 17
        self.height = 15
        self.depth = 4
        self.name='twocolor'

    def preprocess_state(self, board):
        total_number_of_colors = 7
        state = np.eye(total_number_of_colors)[board.board]
        known_balls = state[:, :, 1:]

        current_matching_balls = state[:, :, board.current_ball]
        current_non_matching_balls = np.sum(known_balls, axis=2) - current_matching_balls

        next_matching_balls = state[:, :, board.next_ball]
        next_non_matching_balls = np.sum(known_balls, axis=2) - next_matching_balls

        state = np.dstack((current_matching_balls, current_non_matching_balls, next_matching_balls, next_non_matching_balls))

        return state

    def shape(self):
        return (self.height, self.width, self.depth)
