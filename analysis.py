# Gets decay timescale for every run and forcing duration

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

T1 = 50
T2_array = [150,250,350,450,550] 
plot_rate = 100
dt = 0.01
decay_array = []

def exp_decay(t, A, tau):
    return A * np.exp(-t / tau)

def power_decay(t, A, n):
    return A * t**(-n)

def r_squared(y_actual, y_predicted):
    ss_res = np.sum((y_actual - y_predicted) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    return 1 - (ss_res / ss_tot)

for run in range(1,7):
    if run == 5:
        continue
    decays = []
    for T2 in T2_array:

        energy = np.load(f"energy_{T2}_run{run}.npy")
        times = np.load(f"time_{T2}_run{run}.npy")
        

        # Make decay arrays that contain all values when forcing was off after being on
        decay_mask = (times >= T2)
        decay_time = times[decay_mask]
        decay_energy = energy[decay_mask]
        decay = 0

        decay_time_shifted = decay_time-T2

        popt_exp, pcov = curve_fit(exp_decay, decay_time_shifted, decay_energy, p0=[max(decay_energy), 100])
        A, tau = popt_exp

        print(f"-----Forcing Duration: {T2-T1}   Run: {run}-----\nDecay timescale: {tau:.2f} time units")

        popt_pow, pcov = curve_fit(power_decay, decay_time_shifted + 1, decay_energy, p0=[max(decay_energy), 1])
        A, n = popt_pow
        print(f"Power law decay exponent: {n:.2f}")

        r2_exp = r_squared(decay_energy, exp_decay(decay_time_shifted, *popt_exp))
        r2_pow = r_squared(decay_energy, power_decay(decay_time_shifted + 1, *popt_pow))

        print(f"Exponential R²: {r2_exp:.4f}")
        

        plt.figure(figsize=(10,6))

        plt.plot(decay_time_shifted, decay_energy, label='Data')
        plt.plot(decay_time_shifted, exp_decay(decay_time_shifted, *popt_exp), label='Exponential fit')
        plt.savefig(f"decay_timescale_fit_T{T2}.png")
        plt.legend()
        plt.close()
        
        decays.append(tau)

    decay_array.append(decays)
    decays = []

np.save("decay_timescales.npy", decay_array)
