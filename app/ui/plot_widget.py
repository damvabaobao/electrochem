from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PlotWidget(FigureCanvas):
    def __init__(self):
        self.figure = Figure()
        super().__init__(self.figure)

    def plot(self, signal):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(signal)
        ax.set_title("Signal")
        self.draw()

    def clear(self):
        self.figure.clear()
        self.draw()