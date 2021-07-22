from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QWidget, QGridLayout

from agent import Agent
from algorithms.algorithm import Algorithm
from algorithms.g_algorithm import GAlgorithm
from algorithms.policy_evaluation import PolicyEvaluation
from algorithms.policy_iteration import PolicyIteration
from algorithms.value_iteration import ValueIteration
from grid import GridWidget
from options_widget import OptionsWidget
from plot_window import PlotWindow


class MainWidget(QWidget):

    def __init__(self):
        super().__init__()
        self.thread = QThread()
        self.grid = GridWidget((4, 4))
        self.options = OptionsWidget()

        self.options.size_slider.valueChanged.connect(self.change_grid_size)

        # Agent test
        self.agent = Agent(self.grid.sizes, self.grid.cell_types, True)
        self.agent.policy_signal.connect(self.grid.write_on_cells)
        self.options.agent_run_button.pressed.connect(self.agent.find_policy)

        self._kings_moves = False
        self.algo_type = 0
        self.algo_types = [PolicyEvaluation, PolicyIteration, ValueIteration, GAlgorithm]
        self.algo = Algorithm(self.grid.sizes, self.grid.cell_types, self._kings_moves)
        self.options.run_once_button.pressed.connect(self.algo.step)
        self.options.run_button.pressed.connect(self.algo.run)
        self.options.run_button.pressed.connect(lambda: self.toggle_ui(False))
        self.algo.finished.connect(lambda: self.toggle_ui(True))
        self.reset_algo()

        self.options.moves_choice.currentIndexChanged.connect(self.change_moves_type)
        self.options.algo_choice.currentIndexChanged.connect(self.change_algo)
        self.options.show_choice.currentIndexChanged.connect(self.change_what_to_show)
        grid_layout = QGridLayout()
        self.setLayout(grid_layout)
        self.grid.changed.connect(self.reset_algo)
        grid_layout.addWidget(self.grid, 0, 0)
        grid_layout.addWidget(self.options, 0, 1)

    def change_algo(self, i):
        self.algo_type = i
        self.reset_algo()

    def change_moves_type(self, i):
        self._kings_moves = True if i == 1 else False
        self.reset_algo()

    def change_what_to_show(self, i):
        self.algo.what_to_show = i
        if i == 0:
            self.grid.write_on_cells(self.algo.values_to_str())
        else:
            self.grid.write_on_cells(self.algo.policy_to_arrows())

    def change_grid_size(self, new_size):
        self.grid.sizes = (new_size, new_size)
        self.grid.reset()
        self.agent.sizes = (new_size, new_size)
        self.reset_algo()

    def reset_thread(self):
        self.thread.quit()
        self.thread.deleteLater()
        self.thread = QThread(parent=self)

    def reset_algo(self):
        self.reset_thread()
        self.algo.deleteLater()
        what_to_show = self.options.show_choice.currentIndex()
        self.algo = self.algo_types[self.algo_type](self.grid.sizes, self.grid.cell_types, self._kings_moves,
                                                    what_to_show)
        self.algo.moveToThread(self.thread)
        self.algo.step_signal.connect(self.options.step_text.setText)
        self.algo.write_signal.connect(self.grid.write_on_cells)
        if isinstance(self.algo, GAlgorithm):
            self.options.run_once_button.setEnabled(False)
            self.algo.finished.connect(self.plot)
        else:
            self.options.run_once_button.setEnabled(True)
            self.options.run_once_button.pressed.disconnect()
            self.options.run_once_button.pressed.connect(self.algo.step)
        self.options.run_button.pressed.disconnect()
        self.options.run_button.pressed.connect(self.algo.run)
        self.options.run_button.pressed.connect(lambda: self.toggle_ui(False))
        self.algo.finished.connect(lambda: self.toggle_ui(True))

        self.agent.change_cell_types(self.grid.cell_types)
        self.thread.start()

    def toggle_ui(self, enable):
        self.options.run_button.setEnabled(enable)
        self.options.run_once_button.setEnabled(enable)
        self.options.show_choice.setEnabled(enable)
        self.options.algo_choice.setEnabled(enable)
        self.options.moves_choice.setEnabled(enable)
        self.options.size_slider.setEnabled(enable)
        self.grid.setEnabled(enable)

    def plot(self):
        thread1 = QThread(parent=self)
        fs = PlotWindow("Forward", self.algo.fs, self.algo.T)
        fs.moveToThread(thread1)
        thread1.started.connect(lambda: fs.show())

        thread2 = QThread(parent=self)
        bs = PlotWindow("Backward", self.algo.bs, self.algo.T)
        bs.moveToThread(thread2)
        thread2.started.connect(lambda: bs.show())

        thread3 = QThread(parent=self)
        ps = PlotWindow("Posterior", self.algo.ps, self.algo.T)
        ps.moveToThread(thread3)
        thread3.started.connect(lambda: ps.show())

        thread1.start()
        thread2.start()
        thread3.start()
