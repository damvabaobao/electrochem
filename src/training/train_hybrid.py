import os
import pandas as pd
import numpy as np
import torch
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from src.models.hybrid_model import HybridFeatureBuilder
from src.models.cnn_model import CNNModel, CNNEncoder
from src.models.xgb_model import build_xgb
from src.data.preprocess import normalize, pad_signal
from src.features.feature_extraction import extract_features


def train_hybrid():
    # LOAD DATA
    data = pd.read_csv("data/processed/dataset.csv")

    le = LabelEncoder()
    data["label"] = le.fit_transform(data["label"])

    signals = data.drop("label", axis=1).values
    labels = data["label"].values
    num_classes = len(set(labels))

    print("Num classes:", num_classes)
    # LOAD CNN ENCODER
    cnn = CNNModel(num_classes=num_classes)
    cnn.load_state_dict(torch.load("models/cnn/cnn_best.pth"))
    cnn.eval()

    encoder = CNNEncoder(cnn)
    encoder.eval()
    # BUILD FEATURE
    X_feat = []
    X_deep = []

    for signal in signals:
        signal = normalize(signal)
        signal = pad_signal(signal, 200)

        # handcrafted
        feat = extract_features(signal)

        # deep embedding
        s = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            emb = encoder(s).numpy().flatten()

        X_feat.append(feat)
        X_deep.append(emb)

    X_feat = np.array(X_feat)
    X_deep = np.array(X_deep)
    y = np.array(labels)

    # HYBRID FUSION (CHUẨN)
    builder = HybridFeatureBuilder(alpha=0.7)

    X_combined = builder.fit_transform(X_feat, X_deep)

    # TRAIN / TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42
    )

    # TRAIN XGBOOST
    model = build_xgb()
    model.fit(X_train, y_train)

    # EVALUATE
    y_pred = model.predict(X_test)
    print("=== HYBRID MODEL ===")
    print(classification_report(y_test, y_pred))

    # SAVE
    os.makedirs("models/hybrid", exist_ok=True)

    joblib.dump(model, "models/hybrid/hybrid.pkl")
    joblib.dump(builder, "models/hybrid/builder.pkl")
    joblib.dump(le, "models/hybrid/label_encoder.pkl")

    print("HYBRID DONE")


if __name__ == "__main__":
    train_hybrid()