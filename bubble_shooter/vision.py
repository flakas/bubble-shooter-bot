import cv2
import numpy as np
import time
from PIL import Image
from mss import mss
from collections import namedtuple
import os
import itertools

class ScreenshotSource:
    def __init__(self):
        self.monitor = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
        self.screen = mss()
        self.image = None

    def get(self):
        if self.image is None:
            self.image = self.refresh()

        return self.image

    def refresh(self):
        source_image = self.screen.grab(self.monitor)
        rgb_image = Image.frombytes('RGB', source_image.size, source_image.rgb)
        rgb_image = np.array(rgb_image)
        bgr_image = self.convert_rgb_to_bgr(rgb_image)

        self.image = bgr_image

        return bgr_image

    def convert_rgb_to_bgr(self, img):
        return img[:, :, ::-1]

    def needs_further_processing(self):
        return True

class SeleniumSource(ScreenshotSource):
    def __init__(self, selenium_source):
        self.selenium_source = selenium_source
        self.board = None

    def refresh(self):
        board = self.selenium_source.get_game_board()
        self.board = board
        self.image = board.screen

        return self.image

    def get_game_board(self):
        if self.board is None:
            self.refresh()
        return self.board

    def needs_further_processing(self):
        return False

def cache_until_refresh(func):
    def wrapper(self):
        if func in self.cache:
            return self.cache[func]

        result = func(self)
        self.cache[func] = result
        return result

    return wrapper

class Vision:
    def __init__(self, source, templates_path):
        self.source = source
        self.templates_path = templates_path
        self.cache = {}
        self.board = None
        assert(self.source != None)

    def refresh(self):
        old_image = np.array(self.source.get())
        self.source.refresh()
        new_image = np.array(self.source.get())

        if (old_image != new_image).any():
            # Refresh the cache only if the image body changes
            self.cache = {}

    @cache_until_refresh
    def get_game_board(self):
        """ Detects the game window area within a computer screen """

        if not self.source.needs_further_processing():
            self.board = self.source.get_game_board()
            return self.board

        Board = namedtuple('Board', ['x', 'y', 'w', 'h', 'screen'])
        screen_image = self.source.get()

        original_screen_image = screen_image.copy()

        if self.board != None:
            # The board's location and dimensions are known, we need to only update its contents
            cropped = original_screen_image[self.board.y:self.board.y+self.board.h, self.board.x:self.board.x+self.board.w]
            return Board(self.board.x, self.board.y, self.board.w, self.board.h, cropped)

        grayscale = cv2.cvtColor(screen_image, cv2.COLOR_BGR2GRAY)

        # Find black background around the game screen
        ret, mask = cv2.threshold(grayscale, 1, 255, cv2.THRESH_BINARY)
        binary_grayscale = cv2.bitwise_not(mask)

        # Eliminate noise and smaller elements
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        dilated = cv2.dilate(binary_grayscale, kernel, iterations=1)

        _, contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


        for contour in contours:
            # get rectangle bounding contour
            [x, y, w, h] = cv2.boundingRect(contour)

            # Discard small pieces, we're looking for a game window roughly 800x600
            if w < 700 or h < 500 or w > 800:
                continue

            cropped = original_screen_image[y:y+h, x:x+w]

            self.board = Board(x, y, w, h, cropped)
            return self.board

        return False

    @cache_until_refresh
    def parse_game_board(self):
        board = self.get_game_board()
        pieces_area_offset_x, pieces_area_offset_y, pieces_area_width, pieces_area_height = [16, 16, 560, 480]
        start_x, start_y, end_x, end_y = [pieces_area_offset_x, pieces_area_offset_y, pieces_area_offset_x+pieces_area_width, pieces_area_offset_y+pieces_area_height]
        pieces_area = board.screen[start_y:end_y, start_x:end_x]
        cells_horizontally = 35
        cells_vertically = 15
        (cell_width, cell_height) = (pieces_area_width//cells_horizontally, pieces_area_height//cells_vertically)
        (cell_center_offset_x, cell_center_offset_y) = (cell_width//2, cell_height//2)

        centers = np.zeros((cells_vertically, cells_horizontally), dtype=np.int8)
        for x in range(cells_horizontally):
            for y in range(cells_vertically):
                absolute_x, absolute_y = (x * cell_width + cell_center_offset_x, y * cell_height + cell_center_offset_y)
                centers[y, x] = self.predict_color(pieces_area[absolute_y, absolute_x])

        (current_ball_offset_x, current_ball_offset_y) = (272, 528)
        (current_ball_absolute_x, current_ball_absolute_y) = (current_ball_offset_x + cell_center_offset_x, current_ball_offset_y + cell_center_offset_y)
        current_ball = self.predict_color(board.screen[current_ball_absolute_y, current_ball_absolute_x])

        (next_ball_offset_x, next_ball_offset_y) = (16, 528)
        (next_ball_absolute_x, next_ball_absolute_y) = (next_ball_offset_x + cell_center_offset_x, next_ball_offset_y + cell_center_offset_y)
        next_ball = self.predict_color(board.screen[next_ball_absolute_y, next_ball_absolute_x])

        ParsedBoard = namedtuple('ParsedBoard', ['current_ball', 'next_ball', 'board'])

        return ParsedBoard(current_ball, next_ball, centers)

    def predict_color(self, colors):
        UNKNOWN, BLUE, RED, YELLOW, PURPLE, TEAL, GREEN = (0, 1, 2, 3, 4, 5, 6)
        if np.all(colors == (253, 0, 0)) or np.all(colors == (126, 0, 0)):
            return BLUE
        if np.all(colors == (0, 0, 253)) or np.all(colors == (0, 0, 126)):
            return RED
        if np.all(colors == (0, 253, 253)) or np.all(colors == (0, 126, 126)):
            return YELLOW
        if np.all(colors == (253, 0, 253)) or np.all(colors == (126, 0, 126)):
            return PURPLE
        if np.all(colors == (253, 253, 0)) or np.all(colors == (126, 126, 0)):
            return TEAL
        if np.all(colors == (0, 253, 0)) or np.all(colors == (0, 126, 0)):
            return GREEN
        return UNKNOWN


    @cache_until_refresh
    def is_game_over(self):
        return self._contains_matches('game-over.png')

    @cache_until_refresh
    def is_requesting_name(self):
        return self._contains_matches('enter-your-name.png')

    def _contains_matches(self, template_filename):
        board = self.get_game_board()
        template = self.get_template(template_filename)
        name_text_matches = self.match_template(board.screen, template, threshold=0.99)
        return len(name_text_matches) > 0

    @cache_until_refresh
    def get_ok_button_location(self):
        return self._get_template_location('ok-button.png')

    @cache_until_refresh
    def get_cancel_button_location(self):
        return self._get_template_location('cancel-button.png')

    def _get_template_location(self, template_filename):
        Button = namedtuple('Button', ['x', 'y'])
        board = self.get_game_board()
        template = self.get_template(template_filename)
        button_matches = self.match_template(board.screen, template, threshold=0.99)
        if len(button_matches) > 0:
            return Button(button_matches[0][1], button_matches[0][0])
        else:
            return None

    def match_template(self, img, template, threshold=0.9):
        """
        Matches template image in a target grayscaled image
        """

        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        matches = np.where(res >= threshold)
        matches = np.transpose(matches)
        if not np.size(matches):
            return []
        return matches

    def get_template(self, filename):
        template = os.path.join(self.templates_path, filename)
        template = cv2.imread(template)
        return template

    @cache_until_refresh
    def get_bubble_count(self):
        board = self.get_game_board()

        hsv = cv2.cvtColor(board.screen, cv2.COLOR_BGR2HSV)
        lower_background = np.array([0, 150, 100])
        upper_background = np.array([255, 255, 255])
        background_mask = cv2.inRange(hsv, lower_background, upper_background)
        thresh = cv2.bitwise_and(board.screen, board.screen, mask=background_mask)

        grayscale = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(grayscale, 10, 255, 0)

        _, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        def isBigEnough(contour):
            [x, y, w, h] = cv2.boundingRect(contour)

            return 25 < w < 40 and 25 < h < 40 and x < 600 and y < 500

        contours = list(filter(isBigEnough, contours))

        return max(len(contours) - 2, 0)

