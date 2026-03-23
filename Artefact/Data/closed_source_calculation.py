import csv
import numpy as np
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path   = os.path.join(script_dir, "plant_data.csv")

temp, moisture = [], []
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        temp.append(float(row["Temp_C"]))
        moisture.append(float(row["Moisture_Percent"]))

temp     = np.array(temp)
moisture = np.array(moisture)

# Linear regression
slope, intercept = np.polyfit(temp, moisture, 1)
r      = np.corrcoef(temp, moisture)[0, 1]
mean_T = float(np.mean(temp))
std_T  = float(np.std(temp))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Plant Sensor Data → Spread Function Derivation", fontweight="bold")

# Left side: raw scatter plot + regression line
ax = axes[0]
ax.scatter(temp, moisture, alpha=0.55, color="#27ae60", s=30, label="Sensor readings")
fit_x = np.linspace(temp.min(), temp.max(), 100)
fit_y = slope * fit_x + intercept
ax.plot(fit_x, fit_y, color="#c0392b", lw=2,
        label=f"y = {slope:.2f}x + {intercept:.2f}\nR² = {r**2:.3f}")
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Soil Moisture (%)")
ax.set_title("Moisture vs Temperature")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Right side: derived p_spread curve over wildfire temperature range
T_range       = np.linspace(10, 50, 200)
pred_moisture = np.clip((slope * T_range + intercept) / 100.0, 0, 1)
dryness       = (1 - pred_moisture) ** 2
temp_effect   = 1 / (1 + np.exp(-(T_range - mean_T) / std_T))
p_spread      = 0.30 * temp_effect * dryness

ax = axes[1]
ax.plot(T_range, p_spread, color="#c0392b", lw=2.5)
ax.axvspan(temp.min(), temp.max(), alpha=0.12, color="#27ae60", label="Sensor data range")
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("p_spread")
ax.set_title("Derived spread probability")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "plant_analysis.png"), dpi=150, bbox_inches="tight")
plt.show()

# Values to be used in main.py
print(f"  SLOPE     = {slope:.4f}")
print(f"  INTERCEPT = {intercept:.4f}")
print(f"  MEAN_T    = {mean_T:.2f}")
print(f"  STD_T     = {std_T:.2f}")