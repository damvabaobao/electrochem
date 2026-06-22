import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from scipy.integrate import simpson


# ============================================================
# GENERATE SYNTHETIC ELECTROCHEMICAL SIGNALS
# ============================================================

# Voltage axis
voltage = np.linspace(-1, 1, 500)


# ------------------------------------------------------------
# Substance A
# Example: Glucose-like
# ------------------------------------------------------------

current_A = (
    1.0 * np.exp(-(voltage - 0.25) ** 2 / 0.02)
    + 0.15 * np.exp(-(voltage + 0.4) ** 2 / 0.03)
    + 0.03 * np.random.randn(500)
)


# ------------------------------------------------------------
# Substance B
# Example: Uric-acid-like
# ------------------------------------------------------------

current_B = (
    0.7 * np.exp(-(voltage - 0.05) ** 2 / 0.008)
    + 0.5 * np.exp(-(voltage - 0.45) ** 2 / 0.015)
    + 0.03 * np.random.randn(500)
)


# ============================================================
# FEATURE EXTRACTION FUNCTION
# ============================================================


def extract_peak_info(current, voltage):

    peaks, properties = find_peaks(
        current,
        prominence=0.05,
        width=3
    )

    if len(peaks) == 0:
        return None

    main_peak_idx = peaks[np.argmax(current[peaks])]

    peak_current = current[main_peak_idx]
    peak_voltage = voltage[main_peak_idx]

    peak_area = simpson(
        np.abs(current),
        voltage
    )

    gradient = np.gradient(current)
    curvature = np.gradient(gradient)

    return {
        "peak_idx": main_peak_idx,
        "peak_current": peak_current,
        "peak_voltage": peak_voltage,
        "peak_area": peak_area,
        "gradient": gradient,
        "curvature": curvature
    }


# ============================================================
# EXTRACT FEATURES
# ============================================================

features_A = extract_peak_info(current_A, voltage)
features_B = extract_peak_info(current_B, voltage)
# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(16, 10)
)

# ============================================================
# PLOT 1 - SUBSTANCE A
# ============================================================

ax1 = axes[0, 0]

ax1.plot(
    voltage,
    current_A,
    linewidth=2,
    label="Substance A"
)

# Peak point
ax1.scatter(
    features_A["peak_voltage"],
    features_A["peak_current"],
    s=120,
    marker="o",
    label="Oxidation Peak"
)

# Peak voltage line
ax1.axvline(
    features_A["peak_voltage"],
    linestyle="--"
)

# Peak area
ax1.fill_between(
    voltage,
    current_A,
    alpha=0.2
)

ax1.set_title(
    "Electrochemical Fingerprint - Substance A"
)

ax1.set_xlabel("Voltage (V)")
ax1.set_ylabel("Current (A)")

ax1.legend()

ax1.grid(True)

# ============================================================
# PLOT 2 - SUBSTANCE B
# ============================================================

ax2 = axes[0, 1]

ax2.plot(
    voltage,
    current_B,
    linewidth=2,
    label="Substance B"
)

ax2.scatter(
    features_B["peak_voltage"],
    features_B["peak_current"],
    s=120,
    marker="o",
    label="Oxidation Peak"
)

ax2.axvline(
    features_B["peak_voltage"],
    linestyle="--"
)

ax2.fill_between(
    voltage,
    current_B,
    alpha=0.2
)

ax2.set_title(
    "Electrochemical Fingerprint - Substance B"
)

ax2.set_xlabel("Voltage (V)")
ax2.set_ylabel("Current (A)")

ax2.legend()

ax2.grid(True)

# ============================================================
# PLOT 3 - GRADIENT
# ============================================================

ax3 = axes[1, 0]

ax3.plot(
    voltage,
    features_A["gradient"],
    linewidth=2,
    label="Gradient A"
)

ax3.plot(
    voltage,
    features_B["gradient"],
    linewidth=2,
    label="Gradient B"
)

ax3.set_title(
    "Diffusion / Transport Gradient"
)

ax3.set_xlabel("Voltage (V)")
ax3.set_ylabel("dI/dV")

ax3.legend()

ax3.grid(True)

# ============================================================
# PLOT 4 - CURVATURE
# ============================================================

ax4 = axes[1, 1]

ax4.plot(
    voltage,
    features_A["curvature"],
    linewidth=2,
    label="Curvature A"
)

ax4.plot(
    voltage,
    features_B["curvature"],
    linewidth=2,
    label="Curvature B"
)

ax4.set_title(
    "Electrochemical Curvature"
)

ax4.set_xlabel("Voltage (V)")
ax4.set_ylabel("d²I/dV²")

ax4.legend()

ax4.grid(True)

# ============================================================
# SHOW
# ============================================================

plt.tight_layout()

plt.show()