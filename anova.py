# Calculates the oneway ANOVA to test for high variance

import numpy as np
from scipy import stats

# Your individual tau values for each forcing duration
duration_100 = [238.31980784, 140.63829478, 241.93006296, 173.21505761, 171.19656138]
duration_200 = [188.85198837, 221.89554328, 150.78266295, 279.18013996, 158.09135451]
duration_300 = [165.04802112, 214.05310707, 271.84147076, 261.66887456, 301.91115446]
duration_400 = [238.55489047, 364.86858445, 141.73516013, 220.67273236, 275.68397351]
duration_500 = [160.00751317, 224.85776799, 189.35006789, 236.7667554,  187.20005722]

f_stat, p_value = stats.f_oneway(duration_100, duration_200, duration_300, duration_400, duration_500)

print(f"One-way ANOVA results:")
print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value > 0.05:
    print("Result: No statistically significant difference between groups (p > 0.05)")
    print("Cannot reject null hypothesis that all groups have the same mean tau")
else:
    print("Result: Statistically significant difference between groups (p < 0.05)")
    print("Can reject null hypothesis")