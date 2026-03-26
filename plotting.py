# Plots all of the enstrophy, energy total, and energy spectrum graphs

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

T1 = 50
T2_array = [150, 250, 350, 450, 550]
N = 256
forcing_scale = 6

def exp_decay(t, A, tau):
    return A * np.exp(-t / tau)

def r_squared(y_actual, y_predicted):
    ss_res = np.sum((y_actual - y_predicted) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    return 1 - (ss_res / ss_tot)

for T2 in T2_array:
    print(f"\n===== Forcing Duration: {T2 - T1} (T2={T2}) =====")

    energy_runs = []
    enstrophy_runs = []
    spectra_runs = []
    tau_values = []

    for run in range(1, 6):
        energy = np.load(f'energy_{T2}_run{run}.npy')
        enstrophy = np.load(f'enstrophy_{T2}_run{run}.npy')
        spectra = np.load(f'spectra_{T2}_run{run}.npy')
        times = np.load(f'time_{T2}_run{run}.npy')

        energy_runs.append(energy)
        enstrophy_runs.append(enstrophy)
        spectra_runs.append(spectra)

        # Decay timescale for this run
        decay_mask = (times >= T2)
        decay_time = times[decay_mask]
        decay_energy = energy[decay_mask]
        decay_time_shifted = decay_time - T2

        popt, _ = curve_fit(exp_decay, decay_time_shifted, decay_energy, p0=[max(decay_energy), 100])
        A, tau = popt
        r2 = r_squared(decay_energy, exp_decay(decay_time_shifted, *popt))
        tau_values.append(tau)
        print(f"  Run {run}: tau={tau:.2f}, R²={r2:.4f}")

    # Average across runs
    avg_energy = np.mean(energy_runs, axis=0)
    avg_enstrophy = np.mean(enstrophy_runs, axis=0)
    avg_spectra = np.mean(spectra_runs, axis=0)

    print(f"  Average tau: {np.mean(tau_values):.2f} ± {np.std(tau_values):.2f}")

    # Save averaged data
    np.save(f'avg_energy_{T2}.npy', avg_energy)
    np.save(f'avg_enstrophy_{T2}.npy', avg_enstrophy)
    np.save(f'avg_spectra_{T2}.npy', avg_spectra)
    np.save(f'avg_tau_{T2}.npy', np.array(tau_values))
    print(f"  Saved averaged data for T2={T2}")

    # Plot averaged energy
    plt.figure(figsize=(10, 6))
    plt.plot(times, avg_energy, linewidth=2)
    plt.axvline(x=T1, color='g', linestyle='--', label='Forcing ON (T1)')
    plt.axvline(x=T2, color='r', linestyle='--', label='Forcing OFF (T2)')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title(f'Averaged Energy Evolution - Forcing Duration {T2-T1}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'avg_energy_{T2}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot averaged enstrophy
    plt.figure(figsize=(10, 6))
    plt.plot(times, avg_enstrophy, linewidth=2, color='orange')
    plt.axvline(x=T1, color='g', linestyle='--', label='Forcing ON (T1)')
    plt.axvline(x=T2, color='r', linestyle='--', label='Forcing OFF (T2)')
    plt.xlabel('Time')
    plt.ylabel('Enstrophy')
    plt.title(f'Averaged Enstrophy Evolution - Forcing Duration {T2-T1}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'avg_enstrophy_{T2}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Smoothed 3D spectrum plot
    smoothed_spectra = gaussian_filter(np.log10(avg_spectra + 1e-10), sigma=1)

    wavenumber = np.arange(N // 2 + 1)
    K, T_grid = np.meshgrid(wavenumber, times)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(K, T_grid, smoothed_spectra,
                   cmap=cm.gnuplot,
                   antialiased=True,
                   rstride=2,
                   cstride=2)
    ax.set_xlabel('Wavenumber')
    ax.set_ylabel('Time')
    ax.set_zlabel('log10(E(k))')
    ax.set_title(f'Averaged Energy Spectrum Evolution - Forcing Duration {T2-T1}')
    ax.view_init(azim=60, elev=30)
    plt.savefig(f'avg_spectrum_3d_{T2}.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved plots for T2={T2}")

print("\nAll done!")
