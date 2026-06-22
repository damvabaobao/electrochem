import torch
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.models.cnn_model import CNNModel, CNNEncoder
from src.models.attention_model import AttentionModel
from src.features.feature_extraction import extract_features
from src.data.preprocess import normalize, pad_signal


# ======================
# LOAD DATA
# ======================
def load_data():
    data = pd.read_csv("data/processed/dataset.csv")

    le = LabelEncoder()
    data["label"] = le.fit_transform(data["label"])

    X = data.drop("label", axis=1).values
    y = data["label"].values

    return X, y


# ======================
# FEATURE MODEL
# ======================
def evaluate_feature_model(X_train, X_test, y_train, y_test):
    from xgboost import XGBClassifier

    model = XGBClassifier()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("=== Feature Model ===")
    print(classification_report(y_test, preds, zero_division=0))


# ======================
# CNN MODEL
# ======================
def evaluate_cnn(X_test, y_test, model):

    preds = []
    y_true = []

    for signal, label in zip(X_test, y_test):
        signal = normalize(signal)
        signal = pad_signal(signal, 200)

        x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            out = model(x)
            pred = torch.argmax(out, dim=1).item()

        preds.append(pred)
        y_true.append(label)

    print("=== CNN Model ===")
    print(classification_report(y_true, preds, zero_division=0))


# ======================
# HYBRID MODEL
# ======================
def evaluate_hybrid(X_test, y_test, encoder):

    model = joblib.load("models/hybrid/hybrid.pkl")

    X_combined = []

    for signal in X_test:
        signal = normalize(signal)
        signal = pad_signal(signal, 200)

        # handcrafted
        feat = extract_features(signal)

        # deep
        s = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            deep = encoder(s).numpy().flatten()

        combined = np.concatenate([feat, deep])
        X_combined.append(combined)

    X_combined = np.array(X_combined)

    preds = model.predict(X_combined)

    print("=== Hybrid Model ===")
    print(classification_report(y_test, preds, zero_division=0))


# ======================
# ATTENTION MODEL
# ======================
def evaluate_attention(X_test, y_test, encoder, num_classes):

    # tự động lấy feature_dim
    sample_feat = extract_features(pad_signal(normalize(X_test[0]), 200))
    feat_dim = sample_feat.shape[0]

    model = AttentionModel(
        feat_dim=feat_dim,
        deep_dim=128,
        num_classes=num_classes
    )

    model.load_state_dict(torch.load("models/attention/attention.pth"))
    model.eval()

    preds = []
    y_true = []

    for signal, label in zip(X_test, y_test):
        signal = normalize(signal)
        signal = pad_signal(signal, 200)

        feat = extract_features(signal)
        feat = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)

        s = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            deep = encoder(s)
            out = model(feat, deep)
            pred = torch.argmax(out, dim=1).item()

        preds.append(pred)
        y_true.append(label)

    print("=== Attention Model ===")
    print(classification_report(y_true, preds, zero_division=0))


# ======================
# MAIN
# ======================
if __name__ == "__main__":

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    num_classes = len(set(y))

    # ======================
    # LOAD CNN + ENCODER (DÙNG CHUNG)
    # ======================
    cnn = CNNModel(num_classes=num_classes)
    cnn.load_state_dict(torch.load("models/cnn/cnn_best.pth"))
    cnn.eval()

    encoder = CNNEncoder(cnn)
    encoder.eval()

    # ======================
    # EVALUATION
    # ======================
    evaluate_feature_model(X_train, X_test, y_train, y_test)
    evaluate_cnn(X_test, y_test, cnn)
    evaluate_hybrid(X_test, y_test, encoder)
    evaluate_attention(X_test, y_test, encoder, num_classes)