from time import sleep

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QObject


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
            is_finish_cell = self.cell_types[x, y] == 3
            self._n_states += 0 if is_finish_cell else 1
            self.states[x, y] = 0 if is_finish_cell else self._n_states
            it.iternext()
        self._n_states += 1
        self.V = np.zeros(self._n_states)

        # actions
        self._arrows = ['↑', '↗', '→', '↘', '↓', '↙', '←', '↖'] if kings_move else ['↑', '→', '↓', '←']
        self.new_indexes = [
            lambda x, y: (np.clip(x - 1, 0, self._size[0] - 1), y),  # Up
            lambda x, y: (x, y) if self.is_out_of_bounds((x - 1, y + 1)) else (x - 1, y + 1),  # Up-Right
            lambda x, y: (x, np.clip(y + 1, 0, self._size[0] - 1)),  # Right
            lambda x, y: (x, y) if self.is_out_of_bounds((x + 1, y + 1)) else (x + 1, y + 1),  # Down-Right
            lambda x, y: (np.clip(x + 1, 0, self._size[0] - 1), y),  # Down
            lambda x, y: (x, y) if self.is_out_of_bounds((x + 1, y - 1)) else (x + 1, y - 1),  # Down-Left
            lambda x, y: (x, np.clip(y - 1, 0, self._size[0] - 1)),  # Left
            lambda x, y: (x, y) if self.is_out_of_bounds((x - 1, y - 1)) else (x - 1, y - 1),  # Up-Left
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
        if self.cell_types[x, y] == 3:
            next_state = 0
            reward = 0
        elif self.cell_types[x, y] == 1:
            next_state = self.states[x, y]
            reward = 0
        else:
            new_index = self.new_indexes[n_action](x, y)
            reward = -100 if self.cell_types[new_index] == 1 else -1
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
            arrows[x][y] = self._arrows[best_action] if self.cell_types[x, y] in [0, 2, 4] else ''
            it.iternext()
        return arrows


class PolicyEvaluation(Algorithm):
    def __init__(self,
                 size: (None, None),
                 cell_types: np.ndarray,
                 kings_move=True,
                 what_to_show=0,
                 gamma=1.0,
                 theta=0.0001):
        super().__init__(size, cell_types, kings_move, what_to_show)
        self.gamma = gamma
        self.theta = theta
        self.delta = 0

    def step(self):
        super().step()
        self.step_signal.emit(f"Steps: {self.current_step}")

        self.delta = 0.0
        it = np.nditer(self.states, flags=['multi_index'])
        while not it.finished:
            x, y = it.multi_index
            s = int(self.states[x, y])
            v = self.V[s]
            sum_p = 0
            for a, p_a in enumerate(self.policy[s]):
                next_state, reward = self.move(x, y, a)
                v_next_state = self.V[next_state]
                sum_p += p_a * (reward + self.gamma * v_next_state)
            self.V[s] = sum_p
            self.delta = max(self.delta, abs(v - self.V[s]))
            it.iternext()

        self.write_signal.emit(self.show())

    def run(self, with_sleep=True):
        while True:
            self.step()
            if with_sleep:
                sleep(0.005)
            if self.delta < self.theta:
                break
        self.finished.emit()


class PolicyIteration(Algorithm):

    def __init__(self,
                 size: (None, None),
                 cell_types: np.ndarray,
                 kings_move=True,
                 what_to_show=0,
                 gamma=1.0,
                 theta=0.0001):
        super().__init__(size, cell_types, kings_move, what_to_show)
        self.gamma = gamma
        self.theta = theta
        self.policy_stable = False

    def evaluate(self):
        pol_eval = PolicyEvaluation(self._size, self.cell_types, self._kings_move, self.what_to_show, self.gamma,
                                    self.theta)
        pol_eval.write_signal = self.write_signal
        pol_eval.policy = self.policy
        pol_eval.run()
        pol_eval.deleteLater()
        return pol_eval.V

    def step(self):
        super().step()
        self.step_signal.emit(f"Steps: {self.current_step}")
        self.V = self.evaluate()
        it = np.nditer(self.states, flags=['multi_index'])
        while not it.finished:
            x, y = it.multi_index
            s = int(self.states[x, y])
            old_actions = np.copy(self.policy[s])
            new_actions = np.zeros(self._n_actions)

            for a in range(self._n_actions):
                next_state, reward = self.move(x, y, a)
                new_actions[a] = reward + self.gamma * self.V[next_state]

            best_action = np.argmax(new_actions)
            self.policy[s] = np.zeros(self._n_actions)
            self.policy[s][best_action] = 1

            if not np.array_equal(self.policy[s], old_actions):
                self.policy_stable = False
            it.iternext()
        self.write_signal.emit(self.show())

    def run(self):
        while True:
            self.policy_stable = True
            self.step()
            if self.policy_stable:
                break
        self.finished.emit()


class ValueIteration(Algorithm):
    def __init__(self,
                 size: (None, None),
                 cell_types: np.ndarray,
                 kings_move=True,
                 what_to_show=0,
                 gamma=1.0,
                 theta=0.0001):
        super().__init__(size, cell_types, kings_move, what_to_show)
        self.gamma = gamma
        self.theta = theta
        self.delta = 0

    def step(self):
        super().step()
        self.step_signal.emit(f"Steps: {self.current_step}")

        self.delta = 0.0
        it = np.nditer(self.states, flags=['multi_index'])
        while not it.finished:
            x, y = it.multi_index
            s = int(self.states[x, y])
            v = self.V[s]
            actions = np.zeros(self._n_actions)
            for a in range(self._n_actions):
                next_state, reward = self.move(x, y, a)
                actions[a] = reward + self.gamma * self.V[next_state]

            self.V[s] = actions.max()
            self.delta = max(self.delta, abs(v - self.V[s]))
            best_action = np.argmax(actions)
            self.policy[s] = np.zeros(self._n_actions)
            self.policy[s][best_action] = 1
            it.iternext()

        self.write_signal.emit(self.show())

    def run(self, with_sleep=True):
        while True:
            self.step()
            if with_sleep:
                sleep(0.005)
            if self.delta < self.theta:
                break
        self.finished.emit()
