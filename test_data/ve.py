# read_pssession.py

import re
import json
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# LOAD FILE
# =====================================================

path = r"D:\electrochem_ai(1)\data\100_100mV_s.pssession"

with open(path, "r", encoding="utf-16") as f:
    text = f.read()

print("File loaded")

# =====================================================
# EXTRACT DATA POINTS
# =====================================================

pattern = r'\{"V":(.*?),"C":(.*?),"S":(.*?)\}'

matches = re.findall(pattern, text)

print("Total points:", len(matches))

# =====================================================
# CONVERT TO DATAFRAME
# =====================================================

voltage = []
current = []
state = []

for m in matches:
    try:
        voltage.append(float(m[0]))
        current.append(float(m[1]))
        state.append(int(float(m[2])))
    except:
        pass

df = pd.DataFrame({
    "Voltage": voltage,
    "Current": current,
    "State": state
})

# =====================================================
# PRINT TERMINAL DATA
# =====================================================

print(df.head())

# =====================================================
# SAVE CSV
# =====================================================

df.to_csv("electrochemical_data.csv", index=False)

print("CSV exported")

# =====================================================
# PLOT
# =====================================================

plt.figure(figsize=(8,5))

plt.plot(df["Voltage"], df["Current"])

plt.xlabel("Potential (V)")
plt.ylabel("Current")
plt.title("Electrochemical Curve")

plt.grid(True)

plt.show()