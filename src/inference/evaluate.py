import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.inference.predict import Predictor


def evaluate():
    predictor = Predictor()

    # load test data
    X = pd.read_csv("data/test/test_signals.csv", header=None).values
    y_true = pd.read_csv("data/test/test_labels.csv")["label"].values

    y_pred = []

    for signal in X:
        label, prob = predictor.predict_attention(signal)   # dùng model tốt nhất
        y_pred.append(label)

    print("\n===== CLASSIFICATION REPORT =====\n")
    print(classification_report(y_true, y_pred))

    print("\n===== CONFUSION MATRIX =====\n")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    evaluate()