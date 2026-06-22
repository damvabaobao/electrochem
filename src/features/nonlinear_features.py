import numpy as np
import nolds
from scipy.stats import entropy, skew, kurtosis

def nonlinear_features(signal):
    features = []

    hist, _ = np.histogram(signal, bins=50)
    features += [entropy(hist + 1e-6)]

    features += [skew(signal), kurtosis(signal)]

    try:
        features += [nolds.dfa(signal), nolds.hurst_rs(signal)]
    except:
        features += [0, 0]

    return features