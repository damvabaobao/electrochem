import torch
import joblib
import numpy as np

from src.models.cnn_model import CNNModel, CNNEncoder
from src.features.feature_extraction import extract_features
from src.models.hybrid_model import combine_features
from src.data.preprocess import normalize, pad_signal

def load_models():
    cnn = CNNModel()
    cnn.load_state_dict(torch.load("models/cnn/cnn_best.pth"))
    cnn.eval()

    encoder = CNNEncoder(cnn)
    encoder.eval()

    model = joblib.load("models/hybrid/hybrid.pkl")

    return encoder, model

def predict(signal):

    encoder, model = load_models()

    signal = normalize(signal)
    signal = pad_signal(signal, 200)

    feat = extract_features(signal)

    s = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        emb = encoder(s).numpy().flatten()

    x = combine_features(feat, emb).reshape(1, -1)

    return model.predict(x)