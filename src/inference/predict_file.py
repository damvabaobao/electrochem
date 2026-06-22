import pandas as pd
import numpy as np

from src.inference.predict import Predictor


def predict_from_csv(file_path):
    predictor = Predictor()

    data = pd.read_csv(file_path)

    print("\n===== PREDICTION RESULT =====\n")

    for i, row in data.iterrows():
        signal = row.values.astype(float)

        cnn_label, cnn_prob = predictor.predict_cnn(signal)
        hybrid_label, hybrid_prob = predictor.predict_hybrid(signal)
        att_label, att_prob = predictor.predict_attention(signal)

        print(f"Sample {i+1}:")
        print(f"  CNN       : {cnn_label} ({cnn_prob*100:.2f}%)")
        print(f"  Hybrid    : {hybrid_label} ({hybrid_prob*100:.2f}%)")
        print(f"  Attention : {att_label} ({att_prob*100:.2f}%)")

        print(f"  ==> FINAL : {att_label}")
        print("-" * 40)


if __name__ == "__main__":
    file_path = "data/test/sample.csv"   # đổi path tại đây
    predict_from_csv(file_path)