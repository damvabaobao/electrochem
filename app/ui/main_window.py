from PyQt5.QtWidgets import (
    QMainWindow, QFileDialog, QAction,
    QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout
)

import pandas as pd

from ui.plot_widget import PlotWidget
from ui.table_widget import TableWidget
from core.predictor import Predictor   # ✅ sửa import


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Electrochemical AI System")
        self.setGeometry(100, 100, 1200, 700)

        self.predictor = Predictor()
        self.data = None

        # ===== MENU =====
        self.create_menu()

        # ===== CENTRAL =====
        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout()

        # ===== LEFT: PLOT =====
        self.plot = PlotWidget()
        layout.addWidget(self.plot, 2)

        # ===== RIGHT =====
        right_layout = QVBoxLayout()

        self.label = QLabel("No data")
        right_layout.addWidget(self.label)

        # BUTTONS
        btn_load = QPushButton("Load CSV")
        btn_load.clicked.connect(self.load_file)
        right_layout.addWidget(btn_load)

        btn_predict = QPushButton("Predict")
        btn_predict.clicked.connect(self.predict)
        right_layout.addWidget(btn_predict)

        btn_reset = QPushButton("Reset")
        btn_reset.clicked.connect(self.reset)
        right_layout.addWidget(btn_reset)

        # TABLE
        self.table = TableWidget()
        self.table.cellClicked.connect(self.on_row_selected)  # ✅ thêm chọn dòng
        right_layout.addWidget(self.table)

        layout.addLayout(right_layout, 1)
        central.setLayout(layout)

    # ===== MENU =====
    def create_menu(self):
        menu = self.menuBar()
        file_menu = menu.addMenu("File")

        open_action = QAction("Open", self)
        open_action.triggered.connect(self.load_file)

        save_action = QAction("Save", self)
        save_action.triggered.connect(self.save_file)

        file_menu.addAction(open_action)
        file_menu.addAction(save_action)

    # ===== LOAD =====
    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open CSV", "", "CSV Files (*.csv)"
        )

        if path:
            try:
                df = pd.read_csv(path, header=None)

                if df.empty:
                    self.label.setText("File rỗng")
                    return

                self.data = df

                self.table.set_data(df)

                signal = df.iloc[0].values.astype(float)
                self.plot.plot(signal)

                self.label.setText(f"Loaded: {path}")

            except Exception as e:
                self.label.setText(f"Lỗi file: {str(e)}")

    # ===== SAVE =====
    def save_file(self):
        if self.data is None:
            self.label.setText("Không có dữ liệu để lưu")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", "", "CSV Files (*.csv)"
        )

        if path:
            self.data.to_csv(path, index=False)
            self.label.setText(f"Saved: {path}")

    # ===== SELECT ROW =====
    def on_row_selected(self, row, col):
        if self.data is None:
            return

        try:
            signal = self.data.iloc[row].values.astype(float)
            self.plot.plot(signal)

        except Exception as e:
            self.label.setText(f"Lỗi dữ liệu: {str(e)}")

    # ===== PREDICT =====
    def predict(self):
        if self.data is None:
            self.label.setText("Chưa có dữ liệu")
            return

        try:
            signal = self.data.iloc[0].values.astype(float)

            results = self.predictor.predict_all(signal)

            text = ""
            for name, (label, prob) in results.items():
                text += f"{name.upper()}: {label} ({prob*100:.2f}%)\n"

            self.label.setText(text)

        except Exception as e:
            self.label.setText(f"Lỗi predict: {str(e)}")

    # ===== RESET =====
    def reset(self):
        self.data = None
        self.label.setText("No data")
        self.plot.clear()
        self.table.clear()