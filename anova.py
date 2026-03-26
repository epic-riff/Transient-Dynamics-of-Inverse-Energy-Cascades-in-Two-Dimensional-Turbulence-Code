# Calculates one-way ANOVA to test for statistically significant 
# difference in decay timescale across forcing durations
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Your individual tau values for each forcing duration
all_tau = np.load("decay_timescales.npy")

tau_100 = all_tau[:, 0]
tau_200 = all_tau[:, 1]
tau_300 = all_tau[:, 2]
tau_400 = all_tau[:, 3]
tau_500 = all_tau[:, 4]

# Perform ANOVA test
f_stat, p_value = stats.f_oneway(tau_100, tau_200, tau_300, tau_400, tau_500)

# Print results
print(f"One-way ANOVA results:")
print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {p_value}")

if p_value > 0.05:
    print("Result: No statistically significant difference between groups (p > 0.05)")
    print("Cannot reject null hypothesis that all groups have the same mean tau")
else:
    print("Result: Statistically significant difference between groups (p < 0.05)")
    print("Can reject null hypothesis")

# Combine all data into one array with group labels
data = list(tau_100) + list(tau_200) + list(tau_300) + list(tau_400) + list(tau_500)
n = len(tau_100)
labels = (['100'] * n + ['200'] * n + ['300'] * n + ['400'] * n + ['500'] * n)

# Compute Tukey HSD test
tukey = pairwise_tukeyhsd(data, labels, alpha=0.05)
print(f"\n{tukey}")
