import cv2
import numpy as np
import time
import random

class Visualizer:
    """
    Visualizes agent's Q-values for every action as a bar graph.
    """

    def __init__(self, width, action_space, timeout=1000):
        self.width = width
        self.action_space = action_space
        self.cell_width = width//action_space
        self.max_bar_height = 20 # px
        self.timeout = timeout
        self.bar_color = (0, 255, 0) # green
        self.chosen_bar_color = (0, 0, 255) # red

    def show_evaluations(self, chosen_action, evaluations):
        img = np.zeros((self.max_bar_height, self.width, 3), dtype=np.uint8)

        min_q = min(evaluations, key=lambda i: i[1])[1]
        max_q = max(evaluations, key=lambda i: i[1])[1]
        q_per_pixel = (max_q - min_q) / self.max_bar_height

        bar_heights = [(i, max(1, (q-min_q)/q_per_pixel)) for (i, q) in evaluations]

        corners = []
        for (x, bar_height) in bar_heights:
            top_left = (int(x), int(self.max_bar_height-bar_height))
            bottom_right = (int(x+self.cell_width-1), self.max_bar_height)
            corners.append((x, top_left, bottom_right))
            color = self.bar_color if x == chosen_action else self.chosen_bar_color
            img = cv2.rectangle(img, top_left, bottom_right, color, -1)

        cv2.imshow('evaluations', img)
        cv2.waitKey(self.timeout)

    def cleanup(self):
        cv2.destroyAllWindows()

if __name__ == '__main__':
    visualizer = Visualizer(560, 35)
    start_time = time.time()
    end_time = start_time + 5
    while end_time > time.time():
        evaluations = [(i*16, random.random()) for i in range(35)]
        chosen_action = max(evaluations, key=lambda i: i[1])[0]
        visualizer.show_evaluations(chosen_action, evaluations)
    visualizer.cleanup()
