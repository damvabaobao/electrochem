from src.inference.predict import Predictor
import numpy as np

predictor = Predictor()

correct = 0
total = 50

for i in range(total):
    # giả lập Glucose (ví dụ peak ~0.2)
    signal = np.exp(-(np.linspace(-1,1,200)-0.2)**2 * 20)

    pred = predictor.predict_attention(signal)

    if pred == "Glucose":
        correct += 1

if __name__ == "__main__":
    import numpy as np

    predictor = Predictor()

    signal = np.sin(np.linspace(0, 10, 200))

    cnn_label, cnn_prob = predictor.predict_cnn(signal)
    hybrid_label, hybrid_prob = predictor.predict_hybrid(signal)
    att_label, att_prob = predictor.predict_attention(signal)

    print(f"CNN       : {cnn_label} ({cnn_prob*100:.2f}%)")
    print(f"Hybrid    : {hybrid_label} ({hybrid_prob*100:.2f}%)")
    print(f"Attention : {att_label} ({att_prob*100:.2f}%)")