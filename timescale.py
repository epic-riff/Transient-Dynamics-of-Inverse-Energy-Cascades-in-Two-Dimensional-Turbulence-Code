# Graphs a bar graph for decay timescale with error bars

import matplotlib.pyplot as plt
import numpy as np

T1 = 50
T2_array = [150, 250, 350, 450, 550]
forcing_durations = [T2 - T1 for T2 in T2_array]

# Your results
mean_tau = [191.44138310023456,298.74167325657515,378.4119588675286,379.42942719877243,397.2288353656314]
std_tau = [53.22705748220824,35.2352796507705,23.5740414429127,36.96817937989794,23.96991532246275]

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
ax.set_ylim(0, 500)
ax.grid(True, axis='y', alpha=0.3)
ax.axhline(y=np.mean(mean_tau), color='red', linestyle='--', 
           linewidth=1.5, label=f'Overall mean τ = {np.mean(mean_tau):.1f}')
ax.legend(fontsize=12)

plt.tight_layout()
plt.savefig('tau_barchart.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved tau_barchart.png")
