import csv
import numpy as np
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path   = os.path.join(script_dir, "Algerian_forest_fires_dataset.csv")

isi_vals, fwi_vals, fire_labels = [], [], []

with open(csv_path, newline='', encoding='utf-8') as f:
    for raw in f:
        row = [c.strip() for c in raw.strip().split(',')]
        try:
            int(row[0])
        except (ValueError, IndexError):
            continue
        try:
            isi   = float(row[10])
            fwi   = float(row[12])
            label = 1 if 'fire' in row[13].lower() and 'not' not in row[13].lower() else 0
            isi_vals.append(isi)
            fwi_vals.append(fwi)
            fire_labels.append(label)
        except (ValueError, IndexError):
            continue

isi  = np.array(isi_vals)
fwi  = np.array(fwi_vals)
fire = np.array(fire_labels)

# Linear regression
slope, intercept = np.polyfit(isi, fwi, 1)
r = np.corrcoef(isi, fwi)[0, 1]

fire_fwi   = fwi[fire == 1]
nofire_fwi = fwi[fire == 0]
sig_centre = float((np.mean(nofire_fwi) + np.mean(fire_fwi)) / 2)
sig_scale  = float(np.std(fwi) / 2)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Algerian Forest Fires Dataset → Spread Function Derivation", fontweight="bold")

# ISI vs FWI scatter + regression
ax = axes[0]
ax.scatter(isi[fire==0], fwi[fire==0], alpha=0.5, color="#2980b9", s=25, label="No fire")
ax.scatter(isi[fire==1], fwi[fire==1], alpha=0.5, color="#c0392b", s=25, label="Fire")
x_fit = np.linspace(isi.min(), isi.max(), 100)
ax.plot(x_fit, slope * x_fit + intercept, color="#000000", lw=2,
        label=f"FWI = {slope:.2f}·ISI + {intercept:.2f}\nR² = {r**2:.3f}")
ax.set_xlabel("ISI")
ax.set_ylabel("FWI")
ax.set_title("ISI vs FWI (data-fitted regression)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# FWI distribution: fire vs no-fire
ax = axes[1]
bins = np.linspace(0, fwi.max(), 30)
ax.hist(nofire_fwi, bins=bins, alpha=0.6, color="#2980b9", label="No fire")
ax.hist(fire_fwi,   bins=bins, alpha=0.6, color="#c0392b", label="Fire")
ax.axvline(sig_centre, color="black", lw=2, linestyle="--", label=f"Sigmoid centre = {sig_centre:.1f}")
ax.set_xlabel("FWI")
ax.set_ylabel("Count")
ax.set_title("FWI distribution: fire vs no-fire")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# p_sread curve
ISI_range = np.linspace(0, 20, 200)
FWI_pred  = slope * ISI_range + intercept
p_spread  = 1 / (1 + np.exp(-(FWI_pred - sig_centre) / sig_scale))

ax = axes[2]
ax.plot(ISI_range, p_spread, color="#c0392b", lw=2.5)
ax.set_xlabel("ISI")
ax.set_ylabel("p_spread")
ax.set_title("Derived spread probability vs ISI")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "algerian_analysis.png"), dpi=150, bbox_inches="tight")
plt.show()

# Values to use in main.py
print(f"  SLOPE      = {slope:.4f}")
print(f"  INTERCEPT  = {intercept:.4f}")
print(f"  SIG_CENTRE = {sig_centre:.2f}")
print(f"  SIG_SCALE  = {sig_scale:.2f}")