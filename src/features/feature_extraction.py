import numpy as np

from .signal_features import signal_features
from .nonlinear_features import nonlinear_features
from src.features.physics_features import (
    extract_physics_features
)


def extract_features(current, voltage):

    # =====================================================
    # Traditional signal features
    # =====================================================

    f1 = signal_features(current)

    # =====================================================
    # Nonlinear features
    # =====================================================

    f2 = nonlinear_features(current)

    # =====================================================
    # Physics-informed features
    # =====================================================

    physics_dict = extract_physics_features(
        current,
        voltage
    )

    physics_vector = list(
        physics_dict.values()
    )

    # =====================================================
    # Feature names
    # =====================================================

    feature_names = (

        [f"signal_{i}" for i in range(len(f1))]

        +

        [f"nonlinear_{i}" for i in range(len(f2))]

        +

        list(physics_dict.keys())
    )

    # =====================================================
    # Merge features
    # =====================================================

    all_features = np.array(
        f1 + f2 + physics_vector,
        dtype=np.float32
    )

    # =====================================================
    # Return
    # =====================================================

    return all_features, feature_names