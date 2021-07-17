import numpy as np

from algorithms.algorithm import Algorithm
from algorithms.policy_evaluation import PolicyEvaluation


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
