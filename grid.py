import math

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMouseEvent, QFont, QResizeEvent
from PyQt5.QtWidgets import QWidget, QGridLayout, QLabel

FREE_CELL = 0
OBSTACLE = 1
START = 2
GOAL = 3


class GridWidget(QWidget):
    COLORS = {
        FREE_CELL: ('White', 'black'),
        OBSTACLE: ('Black', 'white'),  # Obstacle
        START: ('Green', 'white'),  # Start
        GOAL: ('Blue', 'white'),  # Finish
        4: ('Red', 'white'),  # Multi agent test
        5: ('Yellow', 'black')  # Current Position
    }
    changed = QtCore.pyqtSignal(tuple, np.ndarray)

    def __init__(self, size: (None, None)):
        super().__init__()
        self.sizes = size
        self.setMinimumSize(500, 500)
        self.cell_types = np.zeros(self.sizes)
        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(1)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.grid_layout)
        self.cells = None
        self.draw_grid()

    def mousePressEvent(self, mouse_position: QMouseEvent):
        x = mouse_position.x()
        y = mouse_position.y()
        x = math.floor(x / (self.size().width() / self.sizes[0]))
        y = math.floor(y / (self.size().height() / self.sizes[1]))
        self.cell_types[y, x] = (self.cell_types[y, x] + 1) % 5
        self.update_cell(x, y)
        self.changed.emit(self.sizes, self.cell_types)

    def get_cell(self, x, y):
        cell = QLabel()
        cell.setGeometry(0, 0, 25, 25)
        cell.setAlignment(Qt.AlignCenter)
        cell.setFont(QFont("Arial", 15))
        colors = self.COLORS[self.cell_types[y, x]]
        background_color = colors[0]
        text_color = colors[1]
        cell.setStyleSheet(
            f"QWidget {{ background-color: {background_color};color: {text_color}; }}")
        return cell

    def update_cell(self, x, y):
        colors = self.COLORS[self.cell_types[y, x]]
        background_color = colors[0]
        text_color = colors[1]
        self.cells[y][x].setStyleSheet(
            f"QWidget {{ background-color: {background_color};color: {text_color}; }}")

    def resizeEvent(self, resize_event: QResizeEvent):
        width = resize_event.size().width()
        height = resize_event.size().height()
        if width > height:
            self.setGeometry(0, 0, height, height)
        else:
            self.setGeometry(0, 0, width, width)

    def draw_grid(self, reset=True):
        if reset:
            self.cell_types = np.zeros(self.sizes)
        while self.grid_layout.count():
            child = self.grid_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.cells = [[self.get_cell(x, y) for x in range(self.sizes[0])] for y in range(self.sizes[1])]
        for x in range(self.sizes[0]):
            for y in range(self.sizes[1]):
                cell = self.cells[x][y]
                self.grid_layout.addWidget(cell, x, y)

    def write_on_cells(self, strings):
        for x in range(len(strings)):
            for y in range(len(strings[x])):
                self.cells[x][y].setText(strings[x][y])

    def set_borders(self):
        self.cell_types[0, :] = 1
        self.cell_types[:, 0] = 1
        self.cell_types[self.sizes[0] - 1, :] = 1
        self.cell_types[:, self.sizes[0] - 1] = 1
        self.draw_grid(reset=False)
