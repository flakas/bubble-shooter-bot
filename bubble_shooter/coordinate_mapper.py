class CoordinateMapper:
    def __init__(self, board_width, agent_moves):
        self.board_width = board_width
        self.agent_moves = agent_moves
        self.scaling_factor = board_width // agent_moves

        assert(self.scaling_factor * self.agent_moves == self.board_width, "Number of moves does not evenly cover the action space")

    def agent_to_game(self, x):
        new_x = x * self.scaling_factor
        return new_x

    def game_to_agent(self, x):
        new_x = x // self.scaling_factor
        return new_x
