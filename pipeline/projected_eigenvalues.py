import numpy as np
import matplotlib.pyplot as plt

# =========================
# INPUT: projected eigenvalues
# =========================
bands = ["30–80 Hz", "80–150 Hz", "150–300 Hz"]

lambdas = np.array([
    [7.927e-03, 6.167e-03, 4.526e-03],   # 30–80 Hz
    [1.074e-02, 6.194e-03, 5.701e-03],   # 80–150 Hz
    [1.464e-02, 1.233e-02, 1.145e-02],   # 150–300 Hz
])

# =========================
# Normalize by trace
# =========================
lambdas_norm = lambdas / lambdas.sum(axis=1, keepdims=True)

# =========================
# Plot
# =========================
x = np.arange(len(bands))
width = 0.25

plt.figure(figsize=(8, 5))

plt.bar(x - width, lambdas_norm[:, 0], width, label=r"$\lambda_1$")
plt.bar(x,         lambdas_norm[:, 1], width, label=r"$\lambda_2$")
plt.bar(x + width, lambdas_norm[:, 2], width, label=r"$\lambda_3$")

plt.xticks(x, bands)
plt.ylabel("Normalized projected eigenvalue")
plt.xlabel("Frequency band")
plt.title("Frequency-resolved projected eigenspectrum (GW170814)")
plt.legend()

plt.tight_layout()
plt.show()
