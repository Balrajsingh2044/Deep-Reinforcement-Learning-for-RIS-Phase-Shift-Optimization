import matplotlib.pyplot as plt
import os

csi_noise = [0.0, 0.5, 1.0, 2.0]
# EXACT NUMBERS FROM YOUR FINAL EVALUATION
m1_rates = [5.36, 5.31, 5.22, 5.26]
m2_rates = [5.33, 5.32, 5.29, 5.34]
m3_rates = [5.26, 5.22, 5.31, 5.29]

plt.figure(figsize=(8, 5))
plt.plot(csi_noise, m1_rates, 'b-o', label='M1: DDPG (Model-Free)', linewidth=2, markersize=8)
plt.plot(csi_noise, m2_rates, 'r-^', label='M2: SAC + Codebook (Model-Free)', linewidth=2, markersize=8)
plt.plot(csi_noise, m3_rates, 'g-s', label='M3: Deep Unfolding (Model-Driven)', linewidth=2, markersize=8)

plt.xlabel('CSI Error Variance (Standard Deviation)', fontsize=12)
plt.ylabel('Achievable Sum Rate (bps/Hz)', fontsize=12)
plt.title('Robustness to Imperfect CSI', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(csi_noise)
plt.tight_layout()

os.makedirs('figures', exist_ok=True)
plt.savefig('figures/figure2_robustness.png', dpi=300)
print("Final Figure 2 saved to figures/figure2_robustness.png")