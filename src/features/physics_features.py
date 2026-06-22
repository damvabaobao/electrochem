import numpy as np

from scipy.signal import find_peaks
from scipy.integrate import simpson
from scipy.stats import skew, kurtosis


class ElectrochemicalPhysicsFeatures:
    """
    Research-grade electrochemical physics feature extractor.

    Features included:
    ------------------
    1. Peak Features
    2. Diffusion Features
    3. Kinetic Features
    4. Thermodynamic Features
    5. Statistical Physics Features
    """

    def __init__(self):

        # Peak detection parameters
        self.prominence = 0.02
        self.width = 3
        self.distance = 5

    # =========================================================
    # MAIN EXTRACTION FUNCTION
    # =========================================================

    def extract_all_features(self, current, voltage):

        current = np.asarray(current)
        voltage = np.asarray(voltage)

        features = {}

        # -----------------------------------------------------
        # Peak Features
        # -----------------------------------------------------
        features.update(
            self.extract_peak_features(current, voltage)
        )

        # -----------------------------------------------------
        # Diffusion Features
        # -----------------------------------------------------
        features.update(
            self.extract_diffusion_features(current, voltage)
        )

        # -----------------------------------------------------
        # Kinetic Features
        # -----------------------------------------------------
        features.update(
            self.extract_kinetic_features(current, voltage)
        )

        # -----------------------------------------------------
        # Thermodynamic Features
        # -----------------------------------------------------
        features.update(
            self.extract_thermodynamic_features(
                current,
                voltage
            )
        )

        # -----------------------------------------------------
        # Statistical Physics Features
        # -----------------------------------------------------
        features.update(
            self.extract_statistical_features(current)
        )

        return features

    # =========================================================
    # 1. PEAK FEATURES
    # =========================================================

    def extract_peak_features(self, current, voltage):

        features = {}

        peaks, properties = find_peaks(
            current,
            prominence=self.prominence,
            width=self.width,
            distance=self.distance
        )

        # -----------------------------------------------------
        # No peak detected
        # -----------------------------------------------------
        if len(peaks) == 0:

            features["num_peaks"] = 0

            features["ox_peak_current"] = 0
            features["ox_peak_voltage"] = 0

            features["peak_width"] = 0
            features["peak_prominence"] = 0

            features["peak_area"] = 0

            return features

        # -----------------------------------------------------
        # Main oxidation peak
        # -----------------------------------------------------

        main_peak_idx = peaks[np.argmax(current[peaks])]

        ox_peak_current = current[main_peak_idx]
        ox_peak_voltage = voltage[main_peak_idx]

        # -----------------------------------------------------
        # Peak width
        # -----------------------------------------------------

        peak_width = properties["widths"][
            np.argmax(current[peaks])
        ]

        # -----------------------------------------------------
        # Peak prominence
        # -----------------------------------------------------

        peak_prominence = properties["prominences"][
            np.argmax(current[peaks])
        ]

        # -----------------------------------------------------
        # Peak area
        # -----------------------------------------------------

        peak_area = simpson(
            np.abs(current),
            voltage
        )

        # -----------------------------------------------------
        # Peak symmetry
        # -----------------------------------------------------

        left_side = current[:main_peak_idx]
        right_side = current[main_peak_idx:]

        left_mean = np.mean(np.abs(left_side)) + 1e-8
        right_mean = np.mean(np.abs(right_side)) + 1e-8

        symmetry_ratio = left_mean / right_mean

        # -----------------------------------------------------
        # Save features
        # -----------------------------------------------------

        features["num_peaks"] = len(peaks)

        features["ox_peak_current"] = float(ox_peak_current)
        features["ox_peak_voltage"] = float(ox_peak_voltage)

        features["peak_width"] = float(peak_width)
        features["peak_prominence"] = float(peak_prominence)

        features["peak_area"] = float(peak_area)

        features["peak_symmetry"] = float(symmetry_ratio)

        return features

    # =========================================================
    # 2. DIFFUSION FEATURES
    # =========================================================

    def extract_diffusion_features(
        self,
        current,
        voltage
    ):

        features = {}

        # -----------------------------------------------------
        # First derivative
        # -----------------------------------------------------

        gradient = np.gradient(current)

        # -----------------------------------------------------
        # Second derivative
        # -----------------------------------------------------

        curvature = np.gradient(gradient)

        # -----------------------------------------------------
        # Diffusion-related descriptors
        # -----------------------------------------------------

        features["max_gradient"] = float(
            np.max(gradient)
        )

        features["min_gradient"] = float(
            np.min(gradient)
        )

        features["mean_gradient"] = float(
            np.mean(gradient)
        )

        features["gradient_std"] = float(
            np.std(gradient)
        )

        features["max_curvature"] = float(
            np.max(curvature)
        )

        features["mean_curvature"] = float(
            np.mean(curvature)
        )

        # -----------------------------------------------------
        # Diffusion asymmetry
        # -----------------------------------------------------

        positive_grad = np.sum(gradient > 0)
        negative_grad = np.sum(gradient < 0)

        asymmetry = (
            positive_grad + 1e-8
        ) / (
            negative_grad + 1e-8
        )

        features["diffusion_asymmetry"] = float(
            asymmetry
        )

        return features

    # =========================================================
    # 3. KINETIC FEATURES
    # =========================================================

    def extract_kinetic_features(
        self,
        current,
        voltage
    ):

        features = {}

        # -----------------------------------------------------
        # Butler-Volmer inspired descriptors
        # -----------------------------------------------------

        abs_current = np.abs(current) + 1e-8

        log_current = np.log(abs_current)

        # -----------------------------------------------------
        # Tafel slope estimation
        # -----------------------------------------------------

        try:

            slope, intercept = np.polyfit(
                voltage,
                log_current,
                1
            )

        except:

            slope = 0
            intercept = 0

        # -----------------------------------------------------
        # Reaction activity
        # -----------------------------------------------------

        reaction_activity = np.mean(abs_current)

        # -----------------------------------------------------
        # Kinetic energy-like feature
        # -----------------------------------------------------

        kinetic_energy = np.sum(
            current ** 2
        )

        # -----------------------------------------------------
        # Signal entropy approximation
        # -----------------------------------------------------

        normalized = abs_current / np.sum(abs_current)

        entropy = -np.sum(
            normalized * np.log(normalized + 1e-12)
        )

        # -----------------------------------------------------
        # Save features
        # -----------------------------------------------------

        features["tafel_slope"] = float(slope)

        features["tafel_intercept"] = float(intercept)

        features["reaction_activity"] = float(
            reaction_activity
        )

        features["kinetic_energy"] = float(
            kinetic_energy
        )

        features["signal_entropy"] = float(
            entropy
        )

        return features

    # =========================================================
    # 4. THERMODYNAMIC FEATURES
    # =========================================================

    def extract_thermodynamic_features(
        self,
        current,
        voltage
    ):

        features = {}

        # -----------------------------------------------------
        # Equilibrium potential estimate
        # -----------------------------------------------------

        equilibrium_idx = np.argmin(
            np.abs(current)
        )

        equilibrium_potential = voltage[
            equilibrium_idx
        ]

        # -----------------------------------------------------
        # Potential range
        # -----------------------------------------------------

        potential_range = np.max(voltage) - np.min(voltage)

        # -----------------------------------------------------
        # Current range
        # -----------------------------------------------------

        current_range = np.max(current) - np.min(current)

        # -----------------------------------------------------
        # Baseline drift
        # -----------------------------------------------------

        baseline_drift = np.mean(current[:10]) - np.mean(current[-10:])

        # -----------------------------------------------------
        # Electrochemical stability
        # -----------------------------------------------------

        stability = 1.0 / (
            np.std(current) + 1e-8
        )

        # -----------------------------------------------------
        # Save features
        # -----------------------------------------------------

        features["equilibrium_potential"] = float(
            equilibrium_potential
        )

        features["potential_range"] = float(
            potential_range
        )

        features["current_range"] = float(
            current_range
        )

        features["baseline_drift"] = float(
            baseline_drift
        )

        features["electrochemical_stability"] = float(
            stability
        )

        return features

    # =========================================================
    # 5. STATISTICAL FEATURES
    # =========================================================

    def extract_statistical_features(self, current):

        features = {}

        features["mean_current"] = float(
            np.mean(current)
        )

        features["std_current"] = float(
            np.std(current)
        )

        features["max_current"] = float(
            np.max(current)
        )

        features["min_current"] = float(
            np.min(current)
        )

        features["rms_current"] = float(
            np.sqrt(np.mean(current ** 2))
        )

        features["skewness"] = float(
            skew(current)
        )

        features["kurtosis"] = float(
            kurtosis(current)
        )

        return features


# =============================================================
# HELPER FUNCTION
# =============================================================

def extract_physics_features(
    current,
    voltage
):

    extractor = ElectrochemicalPhysicsFeatures()

    features = extractor.extract_all_features(
        current,
        voltage
    )

    return features


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":

    # Fake electrochemical signal

    voltage = np.linspace(-1, 1, 500)

    current = (
        np.exp(-(voltage - 0.2) ** 2 / 0.01)
        + 0.05 * np.random.randn(500)
    )

    features = extract_physics_features(
        current,
        voltage
    )

    print("\\n===== ELECTROCHEMICAL PHYSICS FEATURES =====\\n")

    for k, v in features.items():
        print(f"{k}: {v}")
        