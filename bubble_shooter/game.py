import time
import numpy as np

class Game:
    def __init__(self, vision, controller, state_preprocessor):
        self.vision = vision
        self.controller = controller
        self.state_preprocessor = state_preprocessor
        self.min_reward = -1
        self.max_reward = 1
        self.rewards = {
            'step': 0,
            'eliminated_bubble': 1,
            'added_bubble': 0,
            'lose': 0,
            'win': 1,
            'for_each_unsuccessful_step_in_a_row': 0,
            'for_each_successful_step_in_a_row': 0,
        }
        self.per_bubble_reward_scaling = False
        self.steps_made = 0
        self.steps_without_reduction_in_bubbles = 0
        self.steps_reducing_bubbles = 0

    def review_game_board(self):
        self.vision.refresh()

    def get_state(self):
        board = self.vision.parse_game_board()
        state = self.state_preprocessor.preprocess(board)
        return state

    def perform_move(self, target_x, target_y):
        self.steps_made += 1
        start = time.time()
        self.wait_for_animations_to_stop()

        board = self.vision.get_game_board()
        after_first_board = time.time()

        target_x += 24 # offset for the left game border

        initial_score = self.vision.get_bubble_count()

        self.controller.move_to(target_x, target_y)
        after_initial_score = time.time()
        self.wait_for_animations_to_stop()
        after_mouse_click = time.time()

        new_score = self.vision.get_bubble_count()
        after_new_score = time.time()

        if new_score > initial_score:
            self.steps_without_reduction_in_bubbles += 1
            self.steps_reducing_bubbles = 0
        else:
            self.steps_without_reduction_in_bubbles = 0
            self.steps_reducing_bubbles += 1

        bad_gameplay_offset = self.steps_without_reduction_in_bubbles * self.rewards['for_each_unsuccessful_step_in_a_row']
        good_gameplay_offset = self.steps_reducing_bubbles * self.rewards['for_each_successful_step_in_a_row']

        gameplay_offset = bad_gameplay_offset + good_gameplay_offset

        if self.is_finished():
            return self.limit_rewards(self.rewards['step'] + self.rewards['lose'] + gameplay_offset)
        else:
            bubble_delta = new_score - initial_score
            if not self.per_bubble_reward_scaling:
                bubble_delta = 1 if bubble_delta > 0 else -1
            if bubble_delta > 0:
                return self.limit_rewards(self.rewards['step'] + self.rewards['added_bubble'] * bubble_delta + gameplay_offset)
            else:
                return self.limit_rewards(self.rewards['step'] + self.rewards['eliminated_bubble'] * bubble_delta * -1 + gameplay_offset)

    def limit_rewards(self, reward):
        return np.clip(reward, self.min_reward, self.max_reward)

    def wait_for_animations_to_stop(self):
        self.review_game_board()
        old_board = self.vision.get_game_board()
        while True:
            self.wait_for_game_to_catch_up()
            self.review_game_board()
            new_board = self.vision.get_game_board()
            if (old_board.screen == new_board.screen).all():
                return
            else:
                old_board = new_board
                if self.is_finished():
                    return

    def wait_for_game_to_catch_up(self):
        number_of_frames_to_wait = 20
        frames_per_second = 30
        assert number_of_frames_to_wait > 1
        assert frames_per_second > 1
        time.sleep(number_of_frames_to_wait/frames_per_second)

    def is_finished(self):
        self.review_game_board()
        return self.vision.is_game_over() or self.vision.is_requesting_name()

    def reset_steps(self):
        self.steps_made = 0
        self.steps_without_reduction_in_bubbles = 0

    def restart_the_game(self):
        self.reset_steps()
        offset = 30 # pixels
        board = self.vision.get_game_board()
        if self.vision.is_game_over():
            print('The game is over, attempting to click OK')
            button = self.vision.get_ok_button_location()
            self.controller.move_to(button.x + offset, button.y + offset)
            self.wait_for_animations_to_stop()

        self.wait_for_animations_to_stop()
        board = self.vision.get_game_board()

        if self.vision.is_requesting_name():
            print('Attempting to cancel "Enter your name"')
            button = self.vision.get_cancel_button_location()
            self.controller.move_to(button.x + offset, button.y + offset)
            self.wait_for_animations_to_stop()
