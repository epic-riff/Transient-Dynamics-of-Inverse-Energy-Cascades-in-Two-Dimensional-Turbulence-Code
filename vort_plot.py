# Creates vorticity snapshots that certain times in 2D and 3D

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

T2 = 550
run = 1

N = 512
L = 2 * np.pi
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y)

time_array = np.load(f"time_{T2-50}_run{run}.npy")
vorticity = np.load(f"vorticity_{T2-50}_run{run}.npy")

# Choose 6-8 specific snapshots by time
target_times = range(1000)

for target_t in target_times:
    # Find closest snapshot to target time
    idx = np.argmin(np.abs(time_array - target_t))
    t = time_array[idx]
    vort = vorticity[idx]

    # --- 2D Plot ---
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, vort, levels=20, cmap='RdBu_r')
    plt.colorbar(label='Vorticity')
    plt.title(f'Vorticity at t={t:.1f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(f'vorticity_2d_t{int(t):04d}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved 2D vorticity at t={t:.1f}")

    # --- 3D Plot ---
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, vort,
    #                 cmap='RdBu_r',
    #                 antialiased=True,
    #                 rstride=4,
    #                 cstride=4)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('Vorticity')
    # ax.set_title(f'Vorticity Field at t={t:.1f}')
    # ax.view_init(30,60)
    # plt.tight_layout()
    # plt.savefig(f'vorticity_3d_t{int(t):04d}.png', dpi=150, bbox_inches='tight')
    # plt.close()
    # print(f"Saved 3D vorticity at t={t:.1f}")

print("All done!")
