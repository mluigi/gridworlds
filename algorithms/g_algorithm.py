from threading import Thread

import numpy as np
from PyQt5 import QtCore

from algorithms.algorithm import Algorithm
from grid import START, GOAL, OBSTACLE


class GAlgorithm(Algorithm):
    plot_signal = QtCore.pyqtSignal(str, np.ndarray, int)

    def __init__(self, size: (None, None), cell_types: np.ndarray, kings_move=True, what_to_show=0):
        super().__init__(size, cell_types, True, what_to_show)
        self.L = 9
        self.T = 60
        self.bas = np.zeros((self._size[0], self._size[0], self.L, self.T))
        self.fas = np.zeros((self._size[0], self._size[0], self.L, self.T))
        self.pas = np.zeros((self._size[0], self._size[0], self.L, self.T))
        self.bs = np.zeros((self._size[0], self._size[0], self.T))
        self.fs = np.zeros((self._size[0], self._size[0], self.T))
        self.ps = np.zeros((self._size[0], self._size[0], self.T))
        self.ba = np.zeros((self.L, self.T))
        self.fa = np.zeros((self.L, self.T))
        self.pa = np.zeros((self.L, self.T))

        # initial distribution

        start = np.copy(cell_types)
        start = np.where(start != START, 0, start)
        start = np.where(start == START, 1, start)

        goal = np.copy(cell_types)
        goal = np.where(goal != GOAL, 0, goal)
        goal = np.where(goal == GOAL, 1, goal)

        self.fs[:, :, 0] = start
        self.bs[:, :, self.T - 1] = goal

        self.total_steps = self.T * 4 - 3
        self.distr = np.ones((self._n_actions, self._n_actions))
        for action in range(self._n_actions):
            if action == 8:
                self.distr[action] = np.ones(self._n_actions) / ((self._n_actions - 1) * 2)
                self.distr[action, 8] = 1 / 2
            else:
                self.distr[action] = np.ones(self._n_actions) / (self._n_actions * 2)
                self.distr[action, action] = 1 / 3

                next_action = action + 1 if action + 1 < self._n_actions - 1 else 0
                self.distr[action][next_action] = 1 / 6

                next_action = action - 1 if action > 0 else self._n_actions - 2
                self.distr[action][next_action] = 1 / 6

    def has_obstacles_nearby(self, x, y):
        obstacles = []
        for i in range(self._n_actions):
            if self.cell_types[self.new_indexes[i](x, y)] == 1:
                obstacles.append(i)
        return obstacles

    def state_trnstn_distr(self, x, y, action):
        distr = np.copy(self.distr[action])
        obstacles = self.has_obstacles_nearby(x, y)
        if obstacles:
            prob = 0
            for obs in obstacles:
                prob += distr[obs]
                distr[obs] = 0
            for i, p_i in enumerate(distr):
                if i not in obstacles:
                    distr[i] = p_i / (1 - prob)

        reshaped_distr = np.zeros(distr.shape)
        reshaped_distr[0] = distr[7]
        reshaped_distr[1] = distr[0]
        reshaped_distr[2] = distr[1]
        reshaped_distr[3] = distr[6]
        reshaped_distr[4] = distr[8]
        reshaped_distr[5] = distr[2]
        reshaped_distr[6] = distr[5]
        reshaped_distr[7] = distr[4]
        reshaped_distr[8] = distr[3]
        reshaped_distr = reshaped_distr.reshape((3, 3))
        return reshaped_distr

    def forward(self):
        self.fa[:, 0] = np.ones(self.L) / self.L

        for a in range(self.L):
            self.fas[:, :, a, 0] = self.fs[:, :, 0] * self.fa[a, 0]

        for t in range(1, self.T):
            super().step()
            self.step_signal.emit(f"Steps: {self.current_step}/{self.total_steps}")
            for l in range(self.L):
                for x in range(1, self._size[0] - 1):
                    for y in range(1, self._size[0] - 1):
                        mask = self.state_trnstn_distr(x, y, l)
                        for a in range(self._n_actions):
                            self.fas[x - 1:x + 2, y - 1:y + 2, l, t] += mask * self.fas[
                                x, y, a, t - 1] / self._n_actions
            self.fas[:, :, :, t] /= np.sum(self.fas[:, :, :, t])
            self.fs[:, :, t] = np.sum(self.fas[:, :, :, t], 2)
        self.plot_signal.emit("Forward", self.fs, self.T)

    def backward(self):
        self.ba[:, self.T - 1] = np.ones(self.L) / self.L

        for a in range(self.L):
            self.bas[:, :, a, self.T - 1] = self.bs[:, :, self.T - 1] * self.ba[a, self.T - 1]
        self.bas[:, :, :, self.T - 1] /= np.sum(self.bas[:, :, :, self.T - 1])

        for t in reversed(range(self.T - 1)):
            super().step()
            self.step_signal.emit(f"Steps: {self.current_step}/{self.total_steps}")

            for l in range(self.L):
                for x in range(1, self._size[0] - 1):
                    for y in range(1, self._size[0] - 1):
                        skip = 0
                        if self.cell_types[x, y] == OBSTACLE:
                            skip = 1

                        mask = self.state_trnstn_distr(x, y, l)
                        for a in range(self.L):
                            self.bas[x, y, l, t] = (1 - skip) * self.bas[x, y, l, t] + np.sum(
                                mask * self.bas[x - 1:x + 2, y - 1:y + 2, l, t + 1]) / self._n_actions

            self.bas[:, :, :, t] /= np.sum(self.bas[:, :, :, t])
            self.bs[:, :, t] = np.sum(self.bas[:, :, :, t], 2)
        self.plot_signal.emit("Backward", self.bs, self.T)

    def posteriori(self):
        for t in range(self.T):
            super().step()
            self.step_signal.emit(f"Steps: {self.current_step}/{self.total_steps}")
            self.pas[:, :, :, t] = self.fas[:, :, :, t] * self.bas[:, :, :, t]
            self.pas[:, :, :, t] /= np.sum(self.pas[:, :, :, t])
            self.ps[:, :, t] = np.sum(self.pas[:, :, :, t], 2)
        self.plot_signal.emit("Posterior", self.ps, self.T)

    def find_best_path(self):
        k = self.fas[:, :, :, 0] * self.bas[:, :, :, 0]
        idmax = np.argwhere(k == np.max(k)).transpose()
        newfas = np.zeros((self._size[0], self._size[0], self.L))
        newfas[idmax[0], idmax[1], idmax[2]] = 1
        self.fas = np.zeros((self._size[0], self._size[0], self.L, self.T))
        for t in range(1, self.T):
            super().step()
            self.step_signal.emit(f"Steps: {self.current_step}/{self.total_steps}")
            for l in range(self.L):
                for x in range(1, self._size[0] - 1):
                    for y in range(1, self._size[0] - 1):
                        mask = self.state_trnstn_distr(x, y, l)
                        for a in range(self._n_actions):
                            self.fas[x - 1:x + 2, y - 1:y + 2, a, t] += (mask * newfas[x, y, l] / self._n_actions)

            k = self.fas[:, :, :, t] * self.bas[:, :, :, 0]
            idmax = np.argwhere(k == np.max(k)).transpose()
            print(idmax.transpose())
            newfas = np.zeros((self._size[0], self._size[0], self.L))
            newfas[idmax[0], idmax[1], idmax[2]] = 1

    def run(self):
        for_thread = Thread(target=lambda: self.forward())
        bac_thread = Thread(target=lambda: self.backward())
        for_thread.start()
        bac_thread.start()
        for_thread.join()
        bac_thread.join()
        self.posteriori()
        self.find_best_path()
        self.finished.emit()

    def reset_grid(self, size, cell_types):
        super(GAlgorithm, self).reset_grid(size, cell_types)

        start = np.copy(cell_types)
        start = np.where(start != START, 0, start)
        start = np.where(start == START, 1, start)

        goal = np.copy(cell_types)
        goal = np.where(goal != GOAL, 0, goal)
        goal = np.where(goal == GOAL, 1, goal)

        self.bas = np.zeros((self._size[0], self._size[0], self.L, self.T))
        self.fas = np.zeros((self._size[0], self._size[0], self.L, self.T))
        self.pas = np.zeros((self._size[0], self._size[0], self.L, self.T))
        self.bs = np.zeros((self._size[0], self._size[0], self.T))
        self.fs = np.zeros((self._size[0], self._size[0], self.T))
        self.ps = np.zeros((self._size[0], self._size[0], self.T))

        self.fs[:, :, 0] = start
        self.bs[:, :, self.T - 1] = goal
