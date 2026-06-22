import os
import sys

# 🔥 FIX PATH: để Python nhận diện toàn bộ project
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 🔥 FIX DLL conflict (torch + PyQt)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# 🔥 IMPORT TORCH TRƯỚC (rất quan trọng)
import torch

# 🔥 UI
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())