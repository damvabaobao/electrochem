import numpy as np
import torch
import joblib

from src.models.cnn_model import CNNModel, CNNEncoder
from src.models.attention_model import AttentionModel
from src.features.feature_extraction import extract_features
from src.data.preprocess import normalize, pad_signal


class Predictor:
    def __init__(self):

        # ===== LOAD LABEL =====
        self.le = joblib.load("models/hybrid/label_encoder.pkl")

        # ===== LOAD CNN =====
        num_classes = len(self.le.classes_)
        self.cnn = CNNModel(num_classes=num_classes)
        self.cnn.load_state_dict(torch.load("models/cnn/cnn_best.pth"))
        self.cnn.eval()

        self.encoder = CNNEncoder(self.cnn)
        self.encoder.eval()

        # ===== LOAD HYBRID =====
        self.hybrid_model = joblib.load("models/hybrid/hybrid.pkl")
        self.builder = joblib.load("models/hybrid/builder.pkl")

        # ===== LOAD ATTENTION =====
        self.attention_model = AttentionModel(
            feat_dim=15,
            deep_dim=128,
            num_classes=num_classes
        )
        self.attention_model.load_state_dict(
            torch.load("models/attention/attention.pth")
        )
        self.attention_model.eval()

    # ========================
    # PREPROCESS
    # ========================
    def preprocess(self, signal):
        signal = normalize(signal)
        signal = pad_signal(signal, 200)
        return signal

    # ========================
    # FEATURE EXTRACT
    # ========================
    def extract(self, signal):
        feat = extract_features(signal)

        s = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            deep = self.encoder(s).numpy().flatten()

        return feat, deep

    # ========================
    # CNN
    # ========================
    def predict_cnn(self, signal):
        signal = self.preprocess(signal)

        x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            out = self.cnn(x)
            prob = torch.softmax(out, dim=1)

            pred = torch.argmax(prob, dim=1).item()
            confidence = prob[0][pred].item()

        label = self.le.inverse_transform([pred])[0]
        return label, confidence

    # ========================
    # HYBRID
    # ========================
    def predict_hybrid(self, signal):
        signal = self.preprocess(signal)
        feat, deep = self.extract(signal)

        combined = self.builder.transform(
            feat.reshape(1, -1),
            deep.reshape(1, -1)
        )

        pred = self.hybrid_model.predict(combined)[0]
        prob = self.hybrid_model.predict_proba(combined)[0][pred]

        label = self.le.inverse_transform([pred])[0]
        return label, prob

    # ========================
    # ATTENTION
    # ========================
    def predict_attention(self, signal):
        signal = self.preprocess(signal)

        feat = extract_features(signal)
        feat = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)

        s = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            deep = self.encoder(s)
            out = self.attention_model(feat, deep)

            prob = torch.softmax(out, dim=1)
            pred = torch.argmax(prob, dim=1).item()
            confidence = prob[0][pred].item()

        label = self.le.inverse_transform([pred])[0]
        return label, confidence


# ========================
# TEST
# ========================
if __name__ == "__main__":
    predictor = Predictor()

    # test signal giả
    signal = np.sin(np.linspace(0, 10, 200))

    cnn_label, cnn_prob = predictor.predict_cnn(signal)
    hybrid_label, hybrid_prob = predictor.predict_hybrid(signal)
    att_label, att_prob = predictor.predict_attention(signal)

    print(f"CNN       : {cnn_label} ({cnn_prob*100:.2f}%)")
    print(f"Hybrid    : {hybrid_label} ({hybrid_prob*100:.2f}%)")
    print(f"Attention : {att_label} ({att_prob*100:.2f}%)")