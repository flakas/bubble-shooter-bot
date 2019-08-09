import numpy as np

class AllColor:
    """
    Preprocesses vision state for the agent.
    Includes balls of current, next steps and all colors.
    """

    def __init__(self):
        self.width = 17
        self.height = 15
        self.depth = 8

    def preprocess(self, board):
        total_number_of_colors = 7
        state = np.eye(total_number_of_colors)[board.board]
        state = state[:, 1:-1:2, :] # eliminate edge blank spaces, turn 2-cells-per-ball into one

        known_balls = state[:, :, 1:] # ignore empty spaces

        current_matching_balls = state[:, :, board.current_ball]
        next_matching_balls = state[:, :, board.next_ball]

        return np.dstack((current_matching_balls, next_matching_balls, known_balls))

    def shape(self):
        return (self.height, self.width, self.depth)
