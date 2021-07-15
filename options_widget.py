from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QSlider, QLabel, QHBoxLayout, QGroupBox, QPushButton, QFrame, \
    QVBoxLayout, QComboBox


class OptionsWidget(QWidget):

    def __init__(self):
        super().__init__()
        self.setMaximumWidth(500)
        self.setMinimumWidth(100)

        self.options_layout = QVBoxLayout()

        self.options_layout.addWidget(QLabel("Sizes (only squares"))
        self.size_slider = QSlider()
        self.size_slider.setOrientation(Qt.Horizontal)
        self.size_slider.setMinimum(4)
        self.size_slider.setMaximum(32)
        self.size_slider.setTickPosition(QSlider.TicksBelow)
        self.size_slider.setTickInterval(4)
        size_layout = QHBoxLayout()
        text = QLabel('4')
        self.size_slider.valueChanged.connect(text.setNum)
        size_layout.addWidget(self.size_slider)
        size_layout.addWidget(text)
        size_frame = QFrame()
        size_frame.setLayout(size_layout)
        self.options_layout.addWidget(size_frame)

        self.options_layout.addWidget(QLabel("Type of moves"))
        self.moves_choice = QComboBox()
        self.moves_choice.addItems(["4 moves", "King's moves"])
        self.options_layout.addWidget(self.moves_choice)

        self.options_layout.addWidget(QLabel("Algorithm type"))
        self.algo_choice = QComboBox()
        self.algo_choice.addItems(["Policy Evaluation", "Policy Iteration", "Value Iteration"])
        self.options_layout.addWidget(self.algo_choice)

        self.options_layout.addWidget(QLabel("Controls"))
        self.run_button = QPushButton("Run")
        self.run_once_button = QPushButton("Run one step")
        self.stop_button = QPushButton("Stop")
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.run_button)
        controls_layout.addWidget(self.run_once_button)
        controls_layout.addWidget(self.stop_button)
        control_frame = QFrame()
        control_frame.setLayout(controls_layout)
        self.options_layout.addWidget(control_frame)

        self.step_text = QLabel('Steps: 0')
        self.options_layout.addWidget(self.step_text)
        controls_group = QGroupBox("Options")
        controls_group.setLayout(self.options_layout)

        self.options_layout.addWidget(QLabel("What to show"))
        self.show_choice = QComboBox()
        self.show_choice.addItems(["Values", "Policy"])
        self.options_layout.addWidget(self.show_choice)

        # Layout
        self.options_group = QGroupBox("Options")
        self.options_group.setLayout(self.options_layout)
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.options_group)
        main_layout.addStretch(0)
        self.setLayout(main_layout)
