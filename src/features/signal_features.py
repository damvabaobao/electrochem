import numpy as np
from scipy.signal import find_peaks

def signal_features(signal):
    features = []

    features += [np.mean(signal), np.std(signal)]
    features += [np.max(signal), np.min(signal)]

    peaks, _ = find_peaks(signal)
    features += [len(peaks)]
    features += [np.mean(signal[peaks]) if len(peaks) else 0]

    grad = np.gradient(signal)
    features += [np.mean(grad), np.std(grad)]

    features += [np.trapz(signal), np.argmax(signal)]

    return features