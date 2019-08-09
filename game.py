import time
import cv2
import numpy as np

GAME_BOARD_DIMENSION = 64
COLOR_SPACE = 3
GAME_BOARD_X = 17
GAME_BOARD_Y = 15
GAME_BOARD_DEPTH = 8

class Game:
    def __init__(self, vision, controller):
        self.vision = vision
        self.controller = controller
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
        board = self.get_vision_state()
        return self.preprocess_state(board)

    def get_vision_state(self):
        return self.vision.parse_game_board()

    def preprocess_state(self, board):
        # one hot
        total_number_of_colors = 7
        #number_of_relevant_colors = 6
        #pieces = board.board.reshape(GAME_BOARD_Y, GAME_BOARD_X, 1)
        #print(board.board.shape, board.board.dtype)
        state = np.eye(total_number_of_colors)[board.board]
        reduced_state = state[:, 1:-1:2, :] # eliminate edge blank spaces, turn 2-cells-per-ball into one
        state = reduced_state

        #print(state.shape)
        known_balls = state[:, :, 1:]
        #other_color_indexes = list(set(range(total_number_of_colors)) - set([0, board.current_ball, board.next_ball]))

        current_matching_balls = state[:, :, board.current_ball]
        #current_non_matching_balls = np.sum(known_balls, axis=2) - current_matching_balls

        next_matching_balls = state[:, :, board.next_ball]
        #next_non_matching_balls = np.sum(known_balls, axis=2) - next_matching_balls

        #other_color_balls = state[:, :, other_color_indexes]
        #extra_info = np.zeros((GAME_BOARD_Y, GAME_BOARD_X, 1))
        #extra_info[0, :max_state_number, 0] = np.eye(max_state_number)[board.current_ball].T
        #extra_info[1, :max_state_number, 0] = np.eye(max_state_number)[board.next_ball].T
        #print(extra_info)
        #print(extra_info.shape)
        #non_matching_balls = np.sum(known_balls, axis=2) - matching_balls
        #print(non_matching_balls)
        #print(non_matching_balls.shape)

        #state = np.dstack((matching_balls, non_matching_balls, state))
        #state = np.dstack((current_matching_balls, current_non_matching_balls, next_matching_balls, next_non_matching_balls))
        state = np.dstack((current_matching_balls, next_matching_balls, known_balls))
        #state = np.dstack((current_matching_balls, next_matching_balls, other_color_balls))
        #print(state.shape)
        # print(state)
        #print(state[:, :, 0])

        #normalized_state = state / max_state_number
        #print(state.shape, board.current_ball, board.next_ball)

        return state

    def get_state_categorized(self):
        board = self.vision.parse_game_board()
        pieces = board.board.reshape(GAME_BOARD_Y, GAME_BOARD_X, 1)
        extra_info = np.zeros((GAME_BOARD_Y, GAME_BOARD_X, 1))
        extra_info[0, 0, 0] = board.current_ball

        state = np.dstack((extra_info, pieces))

        max_state_number = 6
        normalized_state = state / max_state_number

        return normalized_state

    def get_state_screenshot(self):
        board = self.vision.get_game_board()
        # crop out sidebar, leave the game board and the next steps possible
        screen = board.screen[:board.h, :board.w-200]
        if COLOR_SPACE == 1:
            grayscale = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(grayscale, (GAME_BOARD_DIMENSION, GAME_BOARD_DIMENSION))
        elif COLOR_SPACE == 3:
            img = cv2.resize(screen, (GAME_BOARD_DIMENSION, GAME_BOARD_DIMENSION))
        else:
            raise Exception("Unknown COLOR_SPACE", COLOR_SPACE)

        img = np.array(img)
        img = np.reshape(img, (GAME_BOARD_DIMENSION, GAME_BOARD_DIMENSION, COLOR_SPACE,))
        normalized_img = img / 255.0
        return normalized_img

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
