from vision import ScreenshotSource, Vision, SeleniumSource
from selenium_browser import SeleniumBrowser
from game import Game
from agent import Agent
from memory import Memory
import cv2
import numpy as np
import time

# vision = Vision(ScreenshotSource(), templates_path='templates/')
selenium_browser = SeleniumBrowser()
vision = Vision(SeleniumSource(selenium_browser), templates_path='templates/')
# game = Game(vision, controller)
game = Game(vision, selenium_browser)

board = vision.get_game_board()
# screen = cv2.resize(board.screen, (100, 100))
# cv2.imshow('Screenshot', board.screen)
# print(board.x, board.y, board.w, board.h)
# cv2.waitKey(0)

def test_parse_board(game):
    board = game.vision.parse_game_board()
    game.get_state()

test_parse_board(game)

def test_game_movements(game):
    for i in range(4):
        target_x = 50 + i * 100
        target_y = 400
        game.perform_move(target_x, target_y)

# test_game_movements(game)

def test_game_ui_interactions(game):
    vision = game.vision

    print(vision.is_game_over(), vision.is_requesting_name())
    print(vision.get_ok_button_location(), vision.get_cancel_button_location())

    # game.restart_the_game()
    print(game.get_state())

# test_game_ui_interactions(game)

def continuously_lose_the_game(game):
    attempts = 3
    board = game.vision.get_game_board()
    for i in range(attempts):
        while not game.is_finished():
            target = np.random.randint(0, board.w)
            reward = game.perform_move(target, 400)
            print(f'Got a reward of {reward}')

        game.restart_the_game()

# continuously_lose_the_game(game)

def get_bubble_count(game):
    vision = game.vision

    print(vision.get_bubble_count())

# get_bubble_count(game)

def retrain_the_model(game):
    memory = Memory(max_size=9000)
    agent = Agent(state_size=100*100*3, action_size=560, memory=Memory(9000))
    agent.memory.load_from_file('game_states.pickle')

    episodes = 50
    minibatch_size = 64
    for episode in range(episodes):
        agent.replay(minibatch_size, episode)

    play_the_game = True
    if not play_the_game:
        return
    steps = 10
    for step in range(steps):
        state = game.get_state()
        action = agent.act(state)

        reward = game.perform_move(action, 400)
        print(f'[AGENT] Step: {step}/{steps}, action: {action}, reward: {reward}')

# retrain_the_model(game)
