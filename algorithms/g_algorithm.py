import numpy as np

from algorithms.algorithm import Algorithm
from grid import START, GOAL, OBSTACLE


class GAlgorithm(Algorithm):
    def __init__(self, size: (None, None), cell_types: np.ndarray, what_to_show=0):
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

    def has_obstacles_nearby(self, x, y):
        obstacles = []
        for i in range(self._n_actions):
            if self.cell_types[self.new_indexes[i](x, y)] == 1:
                obstacles.append(i)
        return obstacles

    def state_trnstn_distr(self, x, y, action):
        if action == 8:
            distr = np.ones(self._n_actions) / ((self._n_actions - 1) * 2)
            distr[0] = 1 / 2
        else:
            distr = np.ones(self._n_actions) / (self._n_actions * 2)
            distr[action] = 1 / 3

            next_action = action + 1
            if action + 1 == self._n_actions:
                next_action = 1
            distr[next_action] = 1 / 6

            next_action = action - 1
            if next_action == 0:
                next_action = self._n_actions - 1
            distr[next_action] = 1 / 6
        obstacles = self.has_obstacles_nearby(x, y)
        if obstacles:
            prob = 0
            for obs in obstacles:
                prob += distr[obs]
                distr[obs] = 0
            for i, p_i in enumerate(distr):
                if i not in obstacles:
                    distr[i] = p_i / (1 - prob)

        prn = np.zeros(distr.shape)
        prn[0] = distr[7]
        prn[1] = distr[0]
        prn[2] = distr[1]
        prn[3] = distr[6]
        prn[4] = distr[8]
        prn[5] = distr[2]
        prn[6] = distr[5]
        prn[7] = distr[4]
        prn[8] = distr[3]
        prn = prn.reshape((3, 3))
        return prn

    def forward(self):
        self.fa[:, 0] = np.ones(self.L) / self.L

        for a in range(self.L):
            self.fas[:, :, a, 0] = self.fs[:, :, 0] * self.fa[a, 0]

        for t in range(1, self.T):
            for l in range(self.L):
                it = np.nditer(self.cell_types, flags=['multi_index'])
                while not it.finished:
                    x, y = it.multi_index
                    if self.cell_types[x, y] == OBSTACLE:
                        it.iternext()
                        continue
                    for a in range(self._n_actions):
                        mask = self.state_trnstn_distr(x, y, a)
                        self.fas[x - 1:x + 2, y - 1:y + 2, l, t] = self.fas[x - 1:x + 2, y - 1:y + 2, l, t] + mask * \
                                                                   self.fas[x, y, a, t - 1] / self._n_actions
                    it.iternext()
            self.fas[:, :, :, t] = self.fas[:, :, :, t] / np.sum(self.fas[:, :, :, t])
            self.fs[:, :, t] = np.sum(self.fas[:, :, :, t], 2)
