import math

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QObject

from grid import START, FREE_CELL, OBSTACLE, GOAL


class Algorithm(QObject):
    step_signal = QtCore.pyqtSignal(str)
    write_signal = QtCore.pyqtSignal(list)
    finished = QtCore.pyqtSignal()

    def __init__(self, size: (None, None), cell_types: np.ndarray, kings_move=True, what_to_show=0):
        super().__init__()
        self._size = size
        self._kings_move = kings_move
        self.states = np.zeros(size)
        self.cell_types = cell_types.astype(int)
        self.what_to_show = what_to_show
        # states
        it = np.nditer(self.states, flags=['multi_index'])
        self._n_states = 0
        while not it.finished:
            x, y = it.multi_index
            is_finish_cell = self.cell_types[x, y] == GOAL
            self._n_states += 0 if is_finish_cell else 1
            self.states[x, y] = 0 if is_finish_cell else self._n_states
            it.iternext()
        self._n_states += 1
        self.V = np.zeros(self._n_states)

        # actions
        self.arrows = ['↑', '↗', '→', '↘', '↓', '↙', '←', '↖', ''] if kings_move else ['↑', '→', '↓', '←']
        self.new_indexes = [
            lambda x, y: (np.clip(x - 1, 0, self._size[0] - 1), y),  # Up
            lambda x, y: (x, y) if self.is_out_of_bounds((x - 1, y + 1)) else (x - 1, y + 1),  # Up-Right
            lambda x, y: (x, np.clip(y + 1, 0, self._size[0] - 1)),  # Right
            lambda x, y: (x, y) if self.is_out_of_bounds((x + 1, y + 1)) else (x + 1, y + 1),  # Down-Right
            lambda x, y: (np.clip(x + 1, 0, self._size[0] - 1), y),  # Down
            lambda x, y: (x, y) if self.is_out_of_bounds((x + 1, y - 1)) else (x + 1, y - 1),  # Down-Left
            lambda x, y: (x, np.clip(y - 1, 0, self._size[0] - 1)),  # Left
            lambda x, y: (x, y) if self.is_out_of_bounds((x - 1, y - 1)) else (x - 1, y - 1),  # Up-Left
            lambda x, y: (x, y),  # Still
        ] if kings_move else [
            lambda x, y: (np.clip(x - 1, 0, self._size[0] - 1), y),  # Up
            lambda x, y: (x, np.clip(y + 1, 0, self._size[0] - 1)),  # Right
            lambda x, y: (np.clip(x + 1, 0, self._size[0] - 1), y),  # Down
            lambda x, y: (x, np.clip(y - 1, 0, self._size[0] - 1)),  # Left
        ]
        self._n_actions = len(self.new_indexes)
        self.current_step = 0
        self.policy = np.ones((self._n_states, self._n_actions)) / self._n_actions

    def move(self, x, y, n_action):
        if self.cell_types[x, y] == GOAL:
            next_state = 0
            reward = 0
        elif self.cell_types[x, y] == OBSTACLE:
            next_state = self.states[x, y]
            reward = 0
        else:
            new_index = self.new_indexes[n_action](x, y)
            if self._kings_move:
                reward = -100 if self.cell_types[new_index] == OBSTACLE else -1 if n_action % 2 == 0 else -math.sqrt(2)
            else:
                reward = -100 if self.cell_types[new_index] == OBSTACLE else -1
            next_state = self.states[new_index] if reward != -100 else self.states[x, y]
        return int(next_state), reward

    def is_out_of_bounds(self, new_index):
        return np.any(np.array(new_index) > (np.array(self.states.shape) - 1)) or np.any(np.array(new_index) < 0)

    def step(self):
        self.current_step += 1

    def run(self):
        pass

    def show(self):
        return self.values_to_str() if self.what_to_show == 0 else self.policy_to_arrows()

    def values_to_str(self):
        values = [["" for _ in range(self._size[0])] for _ in range(self._size[0])]
        it = np.nditer(self.states, flags=['multi_index'])
        while not it.finished:
            x, y = it.multi_index
            values[x][y] = "{:.2f}".format(self.V[int(self.states[x, y])])
            it.iternext()
        return values

    def policy_to_arrows(self):
        arrows = [["" for _ in range(self._size[0])] for _ in range(self._size[0])]
        it = np.nditer(self.states, flags=['multi_index'])
        while not it.finished:
            x, y = it.multi_index
            s = self.states[x, y]
            actions = self.policy[int(s)]
            best_action = np.argmax(actions)
            arrows[x][y] = self.arrows[best_action] if self.cell_types[x, y] in [FREE_CELL, START, 4] else ''
            it.iternext()
        return arrows

    def reset_grid(self, size, cell_types):
        self.cell_types = cell_types
        self.states = np.zeros(size)
        self._size = size
        it = np.nditer(self.states, flags=['multi_index'])
        self._n_states = 0
        while not it.finished:
            x, y = it.multi_index
            is_finish_cell = self.cell_types[x, y] == GOAL
            self._n_states += 0 if is_finish_cell else 1
            self.states[x, y] = 0 if is_finish_cell else self._n_states
            it.iternext()
        self._n_states += 1
        self.V = np.zeros(self._n_states)
        self.policy = np.ones((self._n_states, self._n_actions)) / self._n_actions
