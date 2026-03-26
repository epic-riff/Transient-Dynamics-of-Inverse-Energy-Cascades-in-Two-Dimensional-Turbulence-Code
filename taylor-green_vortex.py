# Code for the Taylor-Green vortex test simulation

# Libraries imported
import cupy as cp
from cupy.fft import fft2, fftfreq, ifft2
import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 512                      # Grid Size
L = 2*cp.pi                  # Domain size
nu = 1e-3                    # Viscosity
dt = 0.01                    # Timestep
T = 20                       # Total time
forcing_scale = 6            # Wavenumber to force at
plot_rate = 100              # How often plotting occurs
F_amp = 2000.0               # Forcing amplitude
T1 = 0                       # Start forcing time
T2 = 0 # End forcing time
forcing_update_interval = 10 # Interval into which the forcing is updated in get forcing

# Create simulation space
x = cp.linspace(0, L, N,endpoint=False)
y = cp.linspace(0, L, N,endpoint=False)
X, Y = cp.meshgrid(x,y) 

# Wavenumber grid
kx = fftfreq(N, d=L/N) * 2*cp.pi
ky = fftfreq(N, d=L/N) * 2*cp.pi
KX, KY = cp.meshgrid(kx, ky)

# For calculating energy spectrum
k_mag = cp.sqrt(KX**2 + KY**2)
k_mag = cp.round(k_mag).astype(int)

# Get K^2 to use as Laplacian operator while in Fourier Space
k_squared = KX**2 + KY**2
k_squared[0, 0] = 1  # Avoid division by zero

# Forcing
forcing_mask = (cp.abs(cp.sqrt(KX**2 + KY**2) - forcing_scale) < 1) 
n_forced_modes = int(cp.sum(forcing_mask))
F_amp_scaled = F_amp * N / n_forced_modes

# Dealias mask
k_max = (N/2) * (2/3)
dealias_mask = (cp.sqrt(KX**2 + KY**2) < k_max)

# Gets the velocity vectors (u & v) using the streamfunction
def get_velo(vort_h):
    stream_h = (-vort_h / k_squared) * dealias_mask
    u_h = 1j * KY * stream_h 
    v_h = -1j * KX * stream_h
    u = cp.real(ifft2(u_h))
    v = cp.real(ifft2(v_h))
    return u,v

# Computes advection term
def get_advection(vort_h):
    u,v = get_velo(vort_h)

    do_dx = cp.real(ifft2(1j*KX*vort_h*dealias_mask))
    do_dy = cp.real(ifft2(1j*KY*vort_h*dealias_mask))

    adv = u*do_dx + v*do_dy
    return fft2(adv) * dealias_mask

# Apply staochastic forcing
def get_forcing():
    F_h = cp.zeros((N, N), dtype=complex)
    indices = cp.where(forcing_mask)
    n_modes = len(indices[0])
    phases = cp.random.uniform(0, 2*cp.pi, n_modes)
    F_h[indices] = F_amp_scaled * cp.exp(1j * phases)
    F_h[0, 0] = 0
    return F_h

# Computes energy spectrum
def compute_spectrum(k_mag, vort_h):
    spectrum = cp.zeros(N//2+1)
    for i in range(N//2 + 1):
        mask = (k_mag == i)
        spectrum[i] = cp.sum(cp.abs(vort_h)**2 * mask)

    return spectrum

# Solves for do/dt
def rhs(vort_h, F_h, forcing=True):

    advection = get_advection(vort_h)
    vdt = -nu * k_squared * vort_h

    if forcing:
        return -advection + vdt + F_h
    else: 
        return -advection + vdt


# Inital conditions (set a random vorticity)
x_np = cp.asnumpy(x)
y_np = cp.asnumpy(y)
X_np, Y_np = np.meshgrid(x_np, y_np)
vort = cp.array(2 * np.cos(X_np) * np.cos(Y_np))
vort = vort - cp.mean(vort)
vort_h = fft2(vort) # inital vorticity
vort_h[0, 0] = 0
    
# Create data arrays
error_array = []
forcing = False
F_h = get_forcing()

nsteps = int(T/dt)
for i in range(nsteps):
    # Check if the forcing should be on or off
    if i*dt >= T1 and i*dt < T2:
        forcing = True
    elif i*dt >= T2:
        forcing = False

    if forcing and i % forcing_update_interval == 0:
        F_h = get_forcing()


    # Implement RK4 Time stepping
    k1 = rhs(vort_h, F_h, forcing=forcing)
    k2 = rhs(vort_h+k1*(dt/2), F_h, forcing=forcing)
    k3 = rhs(vort_h+k2*(dt/2), F_h, forcing=forcing)
    k4 = rhs(vort_h+k3*dt, F_h, forcing=forcing)

    vort_h = vort_h + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

    vort_h[0, 0] = 0

    # Check for NaN/Inf values and print statements to help with debugging
    if cp.any(cp.isnan(vort_h)) or cp.any(cp.isinf(vort_h)):
        print(f"ERROR: NaN or Inf detected at step {i}")
        print(f"  Time: {i*dt}")
        print(f"  Max vort_h: {cp.max(cp.abs(vort_h))}")
        print("Simulation stopping - reduce timestep or check parameters")
        break
    
    # Plot after plot_rate steps
    if i % plot_rate == 0:
        print(f"---Step {i}/{nsteps}---")
        vort = cp.real(ifft2(vort_h))
        print(f"Forcing: {forcing}")
        u, v = get_velo(vort_h)
        energy = float(cp.asnumpy(0.5 * cp.mean(u**2 + v**2)))
        enstrophy = float(cp.asnumpy(0.5 * cp.mean(vort**2)))
        spectrum = compute_spectrum(k_mag, vort_h)

        intended = cp.array(2 * np.cos(X_np) * np.cos(Y_np)) * cp.exp(cp.array(-2 * nu * i * dt))        
        error = cp.max(cp.abs(intended - vort))
        percent = error/cp.max(cp.abs(vort)) * 100
        error_array.append(float(cp.asnumpy(percent)))


        print(f"t={i*dt:.1f}, E={energy:.6f}, Z={enstrophy:.6f}")
        print(f"Error: {error} Percent: {percent}%")

print(f"\nSimulation done!")
print(f"Mean Percent Error: {np.mean([float(cp.asnumpy(e)) for e in error_array]):.6f}")
