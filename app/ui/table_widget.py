from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem


class TableWidget(QTableWidget):
    def set_data(self, df):
        self.setRowCount(df.shape[0])
        self.setColumnCount(df.shape[1])

        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                self.setItem(i, j, QTableWidgetItem(str(df.iat[i, j])))

    def clear(self):
        self.setRowCount(0)
        self.setColumnCount(0)