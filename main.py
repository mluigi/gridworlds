import sys

from PyQt5.QtWidgets import QApplication

from main_window import MainWindow


def main():
    app = QApplication([])
    window = MainWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
