import numpy as np
import pandas as pd
import os


# ====== CONFIG ======
NUM_SAMPLES_PER_CLASS = 300
SIGNAL_LENGTH = 200
SAVE_PATH = "data/processed/dataset.csv"


# ====== GENERATE SIGNAL ======
def generate_cv_signal(peak_positions, noise_level=0.02, shift=0.05):
    x = np.linspace(-1, 1, SIGNAL_LENGTH)

    signal = np.zeros_like(x)

    # tạo nhiều peak
    for p in peak_positions:
        p_shifted = p + np.random.uniform(-shift, shift)
        signal += np.exp(-(x - p_shifted)**2 * 15)

    # thêm noise
    noise = np.random.normal(0, noise_level, SIGNAL_LENGTH)
    signal += noise

    # normalize về [-1, 1]
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    signal = signal * 2 - 1

    return signal


# ====== BUILD DATASET ======
def build_dataset():
    data = []
    labels = []

    # Glucose
    for _ in range(NUM_SAMPLES_PER_CLASS):
        signal = generate_cv_signal([0.2])
        data.append(signal)
        labels.append("Glucose")

    # Dopamine
    for _ in range(NUM_SAMPLES_PER_CLASS):
        signal = generate_cv_signal([-0.2])
        data.append(signal)
        labels.append("Dopamine")

    # Uric Acid (multi-peak)
    for _ in range(NUM_SAMPLES_PER_CLASS):
        signal = generate_cv_signal([0.4, -0.4])
        data.append(signal)
        labels.append("UricAcid")

    # convert to DataFrame
    df = pd.DataFrame(data)
    df["label"] = labels

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(SAVE_PATH, index=False)

    print("DONE: Dataset saved at", SAVE_PATH)
    print("Shape:", df.shape)


if __name__ == "__main__":
    build_dataset()