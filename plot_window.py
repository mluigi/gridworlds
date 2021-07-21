from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QSlider, QDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure, Axes

from algorithms.g_algorithm import GAlgorithm


class PlotWindow(QDialog):
    def __init__(self, g_algo: GAlgorithm):
        super().__init__()
        self.g_algo = g_algo
        ver_lay = QVBoxLayout()
        self.slider = QSlider()
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.g_algo.T - 1)
        self.slider.valueChanged.connect(self.change_plot)
        ver_lay.addWidget(self.slider)
        self.figure = Figure()
        self.ax: Axes = self.figure.add_subplot()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        ver_lay.addWidget(self.toolbar)
        ver_lay.addWidget(self.canvas)
        self.setLayout(ver_lay)
        self.exec_()

    def change_plot(self, i):
        self.ax.clear()
        self.ax.matshow(self.g_algo.fs[:, :, i])
        self.canvas.draw()
