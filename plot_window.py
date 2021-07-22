import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QSlider, QMainWindow, QFrame
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure, Axes


class PlotWindow(QMainWindow):
    def __init__(self, title, matrices: np.ndarray, T):
        super().__init__()
        self.frame = QFrame()
        self.setWindowTitle(title)
        self.setWindowFlag(Qt.Tool)
        self.setWindowModality(Qt.NonModal)
        self.matrices = matrices
        self.T = T
        self.slider = QSlider()
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.ax: Axes = self.figure.add_subplot()
        ver_lay = QVBoxLayout()
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.T - 1)
        self.slider.valueChanged.connect(self.change_plot)
        ver_lay.addWidget(self.slider)
        self.ax.matshow(self.matrices[:, :, 0])
        self.ax.set_title(f"T=1")
        ver_lay.addWidget(self.toolbar)
        ver_lay.addWidget(self.canvas)
        self.frame.setLayout(ver_lay)
        self.setCentralWidget(self.frame)

    def change_plot(self, i):
        self.ax.clear()
        self.ax.matshow(self.matrices[:, :, i])
        self.ax.set_title(f"T={i + 1}")
        self.canvas.draw()
