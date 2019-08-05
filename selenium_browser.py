from collections import namedtuple
from selenium import webdriver
import cv2
import time
import numpy as np

class SeleniumBrowser:

    def __init__(self):
        self.game_url = 'http://localhost/'
        profile = webdriver.FirefoxProfile()
        profile.set_preference("plugin.state.flash", 2);
        self.driver = webdriver.Firefox(profile)
        self.driver.implicitly_wait(30)
        self.setup()

    def setup(self):
        self.driver.get(self.game_url)
        time.sleep(1)
        self.game_board_element = self.get_game_board_element()

    def cleanup(self):
        self.driver.close()

    def get_game_board_element(self):
        return self.driver.find_element_by_tag_name('embed')

    def get_game_board(self):
        embed = self.get_game_board_element()
        base64_png = embed.screenshot_as_png
        file_bytes = np.asarray(bytearray(base64_png), dtype=np.uint8)
        decoded_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        rect = embed.rect
        (x, y, w, h) = (int(rect['x']), int(rect['y']), int(rect['width']), int(rect['height']))
        cropped_image = decoded_img[y:y+h, x:x+w]

        Board = namedtuple('Board', ['x', 'y', 'w', 'h', 'screen'])
        return Board(x, y, w, h, cropped_image)

    def move_to(self, x, y):
        action_chains = webdriver.common.action_chains.ActionChains(self.driver)
        action_chains.move_to_element_with_offset(self.game_board_element, x, y)
        action_chains.click()
        action_chains.perform()

if __name__ == '__main__':
    browser = SeleniumBrowser()
    browser.setup()
    print(source.get_game_board())
    browser.move_to(50, 50)
    time.sleep(1)
    for i in range(10):
        browser.move_to(50, 50)
        time.sleep(1)
    browser.cleanup()
