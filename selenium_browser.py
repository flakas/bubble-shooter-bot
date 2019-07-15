from collections import namedtuple
from selenium import webdriver
import cv2
import time
import numpy as np
import base64
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as expected
import random

class SeleniumBrowser:

    def __init__(self, headless=False, game='bubble-shooter'):
        self.game = game
        if self.game == 'bubble-shooter':
            self.game_url = 'http://localhost/'
        elif self.game == 'qwop':
            self.game_url = 'http://www.foddy.net/Athletics.html'
        else:
            raise Exception('Unknown game')

        profile = webdriver.FirefoxProfile()
        profile.set_preference("plugin.state.flash", 2);
        options = webdriver.firefox.options.Options()
        # options.set_headless(headless)
        if headless:
            options.headless = True
            # options.add_argument('-headless')
        self.driver = webdriver.Firefox(profile, options=options)
        self.driver.implicitly_wait(30)
        self.setup()

    def setup(self):
        # print('[SELENIUM] Setting up')
        self.driver.get(self.game_url)
        time.sleep(1)
        self.game_board_element = self.get_game_board_element()

    def cleanup(self):
        # print('[SELENIUM] Cleaning up')
        self.driver.close()

    def get_game_board_element(self):
        if self.game == 'bubble-shooter':
            return self.driver.find_element_by_tag_name('embed')
        elif self.game == 'qwop':
            return self.driver.find_element_by_id('gameContent')

    def get_game_board(self):
        embed = self.get_game_board_element()
        base64_png = embed.screenshot_as_png
        file_bytes = np.asarray(bytearray(base64_png), dtype=np.uint8)
        decoded_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        rect = embed.rect
        (x, y, w, h) = (int(rect['x']), int(rect['y']), int(rect['width']), int(rect['height']))
        cropped_image = decoded_img[y:y+h, x:x+w]
        # cv2.imshow('decoded', cropped_image)
        # cv2.waitKey(0)
        Board = namedtuple('Board', ['x', 'y', 'w', 'h', 'screen'])
        return Board(x, y, w, h, cropped_image)

    def move_to(self, x, y):
        # print(f'[SELENIUM] Moving to ({x}, {y})')
        action_chains = webdriver.common.action_chains.ActionChains(self.driver)
        action_chains.move_to_element_with_offset(self.game_board_element, x, y)
        action_chains.click()
        action_chains.perform()

    def press_buttons(self, buttons=[]):
        print('Pressing', buttons)
        action_chains = webdriver.common.action_chains.ActionChains(self.driver)
        # for button in buttons:
            # action_chains.key_down(button, self.game_board_element)
        # action_chains.send_keys('e')
        # action_chains.pause(0.2)
        # for button in buttons:
            # action_chains.key_up(button, self.game_board_element)
        # action_chains.send_keys_to_element(self.game_board_element, *buttons).pause(0.2)
        action_chains.send_keys(buttons[0])
        action_chains.pause(0.1)
        action_chains.perform()

# source = SeleniumBrowser(headless=False, game='qwop')
# source.setup()
# print(source.get_game_board())
# source.move_to(50, 50)
# time.sleep(1)
# for i in range(10):
    # source.press_buttons(['q', random.choice(['p', 'o'])])
    # source.press_buttons(['R'])
    # time.sleep(0.5)
# time.sleep(5)
# source.cleanup()
