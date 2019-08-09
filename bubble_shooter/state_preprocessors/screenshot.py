import numpy as np
import cv2

class Screenshot:
    """
    Preprocesses vision state for the agent.
    Takes in a screenshot based state
    """

    def __init__(self, width=64, height=64, color_space='rgb'):
        self.width = 64
        self.height = 64
        self.color_space = color_space

        if color_space == 'rgb':
            self.depth = 3
        elif color_space == 'grayscale':
            self.depth = 1

    def preprocess(self, board):
        # crop out sidebar, leave the game board and the next steps possible
        screen = board.screen[:board.h, :board.w-200]
        if self.color_space == 'grayscale':
            grayscale = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(grayscale, (self.width, self.height))
        elif self.color_space == 'rgb':
            img = cv2.resize(screen, (self.width, self.height))
        else:
            raise Exception("Unknown COLOR_SPACE", self.color_space)

        img = np.array(img)
        img = np.reshape(img, (self.height, self.width, self.depth,))

        normalized_img = img / 255.0
        return normalized_img

    def shape(self):
        return (self.height, self.width, self.depth)

