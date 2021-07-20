from PyQt5.QtWidgets import QMainWindow

from main_widget import MainWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        title = "GridWorld"
        self.setWindowTitle(title)
        self.setGeometry(0, 0, 500, 300)
        self.setCentralWidget(MainWidget())
        self.setContentsMargins(0, 0, 0, 0)
        self.setMinimumSize(900, 550)
        self.show()
