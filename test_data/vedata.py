import numpy as np
import matplotlib.pyplot as plt

V = np.linspace(-1, 1, 300)

I1 = 0.8*np.exp(-(V-0.15)**2/(2*0.08**2))
I2 = 1.0*np.exp(-(V-0.35)**2/(2*0.06**2))
I3 = 0.65*np.exp(-(V-0.55)**2/(2*0.10**2))

plt.figure(figsize=(8,5))

plt.plot(V, I1, label='Analyte A')
plt.plot(V, I2, label='Analyte B')
plt.plot(V, I3, label='Analyte C')

plt.xlabel("Potential (V)")
plt.ylabel("Current (A)")
plt.title("Simulated Voltammetric Responses")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()