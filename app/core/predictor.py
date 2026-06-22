import os
import sys
import numpy as np
import torch
import joblib

# ===== FIX IMPORT PATH =====
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(BASE_DIR)

# ===== IMPORT =====
from src.models.cnn_model import CNNModel, CNNEncoder
from src.models.attention_model import AttentionModel
from src.features.feature_extraction import extract_features
from src.data.preprocess import normalize, pad_signal


# ===== MODEL PATH =====
MODEL_DIR = os.path.join(BASE_DIR, "models_save")


class Predictor:
    def __init__(self, base_path=None, device=None):

        # ===== DEVICE =====
        self.device = device if device else (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # ===== PATH =====
        self.base_path = base_path if base_path else MODEL_DIR

        print("===== INIT PREDICTOR =====")
        print("Device:", self.device)
        print("Model path:", self.base_path)

        # ===== LABEL =====
        self.le = joblib.load(
            os.path.join(self.base_path, "hybrid", "label_encoder.pkl")
        )
        self.num_classes = len(self.le.classes_)

        # ===== CNN =====
        self.cnn = CNNModel(num_classes=self.num_classes).to(self.device)

        self.cnn.load_state_dict(
            torch.load(
                os.path.join(self.base_path, "cnn", "cnn_best.pth"),
                map_location=self.device
            )
        )
        self.cnn.eval()

        # ===== ENCODER =====
        self.encoder = CNNEncoder(self.cnn).to(self.device)
        self.encoder.eval()

        # ===== HYBRID =====
        self.hybrid_model = joblib.load(
            os.path.join(self.base_path, "hybrid", "hybrid.pkl")
        )

        self.builder = joblib.load(
            os.path.join(self.base_path, "hybrid", "builder.pkl")
        )

        # ===== ATTENTION =====
        self.attention_model = AttentionModel(
            feat_dim=15,
            deep_dim=128,
            num_classes=self.num_classes
        ).to(self.device)

        self.attention_model.load_state_dict(
            torch.load(
                os.path.join(self.base_path, "attention", "attention.pth"),
                map_location=self.device
            )
        )
        self.attention_model.eval()

    # =============================
    # PREPROCESS
    # =============================
    def preprocess(self, signal):
        signal = normalize(signal)
        signal = pad_signal(signal, 200)
        return signal

    # =============================
    # FEATURE EXTRACTION
    # =============================
    def extract(self, signal):

        # feature thủ công
        feat = extract_features(signal)

        # deep feature
        s = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            deep = self.encoder(s).cpu().numpy().flatten()

        return feat, deep

    # =============================
    # CNN
    # =============================
    def predict_cnn(self, signal):
        signal = self.preprocess(signal)

        x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.cnn(x)
            prob = torch.softmax(out, dim=1)
            conf, pred = torch.max(prob, dim=1)

        label = self.le.inverse_transform([pred.item()])[0]
        return label, conf.item()

    # =============================
    # HYBRID
    # =============================
    def predict_hybrid(self, signal):
        signal = self.preprocess(signal)

        feat, deep = self.extract(signal)

        combined = self.builder.transform(
            feat.reshape(1, -1),
            deep.reshape(1, -1)
        )

        prob = self.hybrid_model.predict_proba(combined)
        pred = np.argmax(prob)

        label = self.le.inverse_transform([pred])[0]
        return label, prob[0][pred]

    # =============================
    # ATTENTION
    # =============================
    def predict_attention(self, signal):
        signal = self.preprocess(signal)

        feat = extract_features(signal)
        feat = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self.device)

        s = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            deep = self.encoder(s)
            out = self.attention_model(feat, deep)

            prob = torch.softmax(out, dim=1)
            conf, pred = torch.max(prob, dim=1)

        label = self.le.inverse_transform([pred.item()])[0]
        return label, conf.item()

    # =============================
    # ALL
    # =============================
    def predict_all(self, signal):
        return {
            "cnn": self.predict_cnn(signal),
            "hybrid": self.predict_hybrid(signal),
            "attention": self.predict_attention(signal)
        }


# =============================
# TEST
# =============================
if __name__ == "__main__":
    print("\n===== RUN TEST =====")

    try:
        signal = np.random.randn(200)

        predictor = Predictor()

        results = predictor.predict_all(signal)

        print("\n=== RESULTS ===")
        for model, (label, conf) in results.items():
            print(f"{model}: {label} ({conf:.4f})")

    except Exception as e:
        print("\n❌ ERROR:", e)