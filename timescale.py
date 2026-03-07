# Graphs a bar graph for decay timescale with error bars

import matplotlib.pyplot as plt
import numpy as np

T1 = 50
T2_array = [150, 250, 350, 450, 550]
forcing_durations = [T2 - T1 for T2 in T2_array]

# Your results
mean_tau = [193.06, 199.76, 242.90, 248.30, 199.64]
std_tau = [44.88, 52.56, 53.77, 81.47, 31.01]

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(forcing_durations, mean_tau,
              width=40,
              yerr=std_tau,
              capsize=8,
              color='steelblue',
              edgecolor='black',
              linewidth=0.8,
              error_kw={'linewidth': 2, 'capthick': 2})

ax.set_xlabel('Forcing Duration (time units)', fontsize=14)
ax.set_ylabel('Mean Decay Timescale τ (time units)', fontsize=14)
ax.set_title('Decay Timescale vs Forcing Duration', fontsize=16)
ax.set_xticks(forcing_durations)
ax.set_ylim(0, 400)
ax.grid(True, axis='y', alpha=0.3)
ax.axhline(y=np.mean(mean_tau), color='red', linestyle='--', 
           linewidth=1.5, label=f'Overall mean τ = {np.mean(mean_tau):.1f}')
ax.legend(fontsize=12)

plt.tight_layout()
plt.savefig('tau_barchart.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved tau_barchart.png")