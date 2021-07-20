import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QObject

from algorithms.value_iteration import ValueIteration
from grid import START


class Agent(QObject):
    policy_signal = QtCore.pyqtSignal(list)
    error_signal = QtCore.pyqtSignal()

    def __init__(self, size, cell_types, kings_move):
        super().__init__()
        self.sizes = size
        self.cell_types = cell_types
        self.kings_move = kings_move
        self.algo = ValueIteration(self.sizes, self.cell_types, self.kings_move)
        self.cell_types_changed = True
        self.moves_to_do = []

    def change_cell_types(self, cell_types):
        self.cell_types = cell_types
        self.cell_types_changed = True
        self.moves_to_do = []
        self.algo.deleteLater()
        self.algo = ValueIteration(self.sizes, self.cell_types, self.kings_move)

    def policy_to_arrows(self):
        arrows = [["" for _ in range(self.sizes[0])] for _ in range(self.sizes[0])]
        current_position = np.transpose(np.nonzero(self.cell_types == START))[0]
        for action in self.moves_to_do:
            x = current_position[0]
            y = current_position[1]
            arrows[x][y] = self.algo.arrows[action]
            current_position = self.algo.new_indexes[action](x, y)
        return arrows

    def find_policy(self):
        self.moves_to_do = []
        self.algo.run()
        current_position = np.transpose(np.nonzero(self.cell_types == START))[0]
        while True:
            x = current_position[0]
            y = current_position[1]
            state = int(self.algo.states[x, y])
            if state == 0:
                break
            action = np.argmax(self.algo.policy[state])
            self.moves_to_do.append(action)
            current_position = self.algo.new_indexes[action](x, y)
        self.policy_signal.emit(self.policy_to_arrows())
