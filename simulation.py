# Code for the main simulation

# Libraries imported
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftfreq, ifft2

# Parameters
N = 256           # Grid Size
L = 2*np.pi       # Domain size
nu = 1e-3         # Viscosity
dt = 0.01         # Timestep
T = 1000           # Total time
forcing_scale = 6 # Wavenumber to force at
plot_rate = 100   # How often plotting occurs
F_amp = 200.0     # Forcing amplitude
T1 = 50           # Start forcing time
T2_array = [150,250,350,450,550]           # End forcing time

# Create simulation space
x = np.linspace(0, L, N,endpoint=False)
y = np.linspace(0, L, N,endpoint=False)
X, Y = np.meshgrid(x,y) 

# Wavenumber grid
kx = fftfreq(N, d=L/N) * 2*np.pi
ky = fftfreq(N, d=L/N) * 2*np.pi
KX, KY = np.meshgrid(kx, ky)

# For calculating energy spectrum
k_mag = np.sqrt(KX**2 + KY**2)
k_mag = np.round(k_mag).astype(int)

# Get K^2 to use as Laplacian operator while in Fourier Space
k_squared = KX**2 + KY**2
k_squared[0, 0] = 1  # Avoid division by zero

# Forcing
forcing_mask = (np.abs(np.sqrt(KX**2 + KY**2) - forcing_scale) < 1) 
F_h = forcing_mask * F_amp
F_h[0,0] = 0

# Dealias mask
k_max = (N/2) * (2/3)
dealias_mask = (np.sqrt(KX**2 + KY**2) < k_max)

# Gets the velocity vectors (u & v) using the streamfunction
def get_velo(vort_h):
    stream_h = (-vort_h / k_squared) * dealias_mask
    u_h = 1j * KY * stream_h 
    v_h = -1j * KX * stream_h
    u = np.real(ifft2(u_h))
    v = np.real(ifft2(v_h))
    return u,v

# Computes advection term
def get_advection(vort_h):
    u,v = get_velo(vort_h)

    do_dx = np.real(ifft2(1j*KX*vort_h*dealias_mask))
    do_dy = np.real(ifft2(1j*KY*vort_h*dealias_mask))

    adv = u*do_dx + v*do_dy
    return fft2(adv) * dealias_mask

# Computes energy spectrum
def compute_spectrum(k_mag, vort_h):
    spectrum = np.zeros(N//2+1)
    for i in range(N//2 + 1):
        mask = (k_mag == i)
        spectrum[i] = np.sum(np.abs(vort_h)**2 * mask)

    return spectrum

# Solves for do/dt
def rhs(vort_h, forcing=True):

    advection = get_advection(vort_h)
    vdt = -nu * k_squared * vort_h

    if forcing:
        return -advection + vdt + F_h
    else: 
        return -advection + vdt

# Run 5 times
for run in range(1,6):
    # Run for each forcing duration
    for T2 in T2_array:
        print(f"-------------------------------------------Forcing Duration: {T2-T1}-------------------------------------------")

        # Inital conditions (set a random vorticity)
        vort = np.random.randn(N, N)
        vort = vort - np.mean(vort)
        vort_h = fft2(vort) # inital vorticity
        vort_h[0, 0] = 0
            
        # Create data arrays
        time_array = []
        enstrophy_array = []
        energy_array = []
        vorticity_array = []
        spectra_array = []
        timescale_array = []
        forcing = False

        nsteps = int(T/dt)
        for i in range(nsteps):
            # Check if the forcing should be on or off
            if i*dt >= T1 and i*dt < T2:
                forcing = True
            elif i*dt >= T2:
                forcing = False

            # Implement RK4 Time stepping
            k1 = rhs(vort_h, forcing=forcing)
            k2 = rhs(vort_h+k1*(dt/2), forcing=forcing)
            k3 = rhs(vort_h+k2*(dt/2), forcing=forcing)
            k4 = rhs(vort_h+k3*dt, forcing=forcing)

            vort_h = vort_h + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

            vort_h[0, 0] = 0

            # Check for NaN/Inf values and print statements to help with debugging
            if np.any(np.isnan(vort_h)) or np.any(np.isinf(vort_h)):
                print(f"ERROR: NaN or Inf detected at step {i}")
                print(f"  Time: {i*dt}")
                print(f"  Max vort_h: {np.max(np.abs(vort_h))}")
                print("Simulation stopping - reduce timestep or check parameters")
                break
            
            # Plot after plot_rate steps
            if i % plot_rate == 0:
                print(f"---Step {i}/{nsteps}---")

                # Get vorticity values
                vort = np.real(ifft2(vort_h))
                print(f"Forcing: {forcing}")

                # Get the velocity and solve for energy spectrum, enstrophy, and energy total
                u,v = get_velo(vort_h)
                energy = 0.5 * np.mean(u**2 + v**2)
                enstrophy = 0.5 * np.mean(vort**2)
                spectrum = compute_spectrum(k_mag, vort_h)
            
                # Store values
                time_array.append(i * dt)
                energy_array.append(energy)
                enstrophy_array.append(enstrophy)
                vorticity_array.append(vort.copy())
                spectra_array.append(spectrum)
            
                print(f"t={i*dt:.1f}, E={energy:.6f}, Z={enstrophy:.6f}")

        print("Simulation done!\nSaving Data!")

        # Save data into .npy file
        np.save(f'time_{T2}_run{run}.npy', np.array(time_array))
        np.save(f'energy_{T2}_run{run}.npy', np.array(energy_array))
        np.save(f'enstrophy_{T2}_run{run}.npy', np.array(enstrophy_array))
        np.save(f'vorticity_{T2}_run{run}.npy', np.array(vorticity_array))
        np.save(f'spectra_{T2}_run{run}.npy', np.array(spectra_array))

        print("Data Saved!")



