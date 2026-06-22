import pandas as pd
import numpy as np


class DataManager:

    def __init__(self):
        self.data = None

    # ===== LOAD =====
    def load_csv(self, path):
        try:
            df = pd.read_csv(path, header=None)
            self.validate(df)

            self.data = df
            return df

        except Exception as e:
            print("Load error:", e)
            return None

    # ===== SAVE =====
    def save_csv(self, path):
        if self.data is not None:
            self.data.to_csv(path, index=False)

    # ===== VALIDATE =====
    def validate(self, df):
        if df.shape[1] < 50:
            raise ValueError("Signal quá ngắn")

        if df.isnull().values.any():
            raise ValueError("Data chứa NaN")

    # ===== GET SIGNAL =====
    def get_signal(self, index=0):
        if self.data is None:
            return None

        return self.data.iloc[index].values.astype(float)

    # ===== GET ALL =====
    def get_all(self):
        return self.data

    # ===== RESET =====
    def reset(self):
        self.data = None