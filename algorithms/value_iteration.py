from time import sleep

import numpy as np

from algorithms.algorithm import Algorithm


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
