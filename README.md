# Transient Dynamics of Inverse Energy Cascades in Two-Dimensional Turbulence

## Overview
This project studies how the duration of external forcing affects the decay 
timescale of a 2D turbulent inverse energy cascade after forcing is removed. 
In 2D turbulence, energy flows from small scales to large scales, which is a concept known as the inverse 
energy cascade, which is directly relevant to geophysical systems like ocean 
currents and atmospheric flows.

A pseudospectral Navier-Stokes solver was built from scratch in Python to 
simulate 2D turbulence under transient forcing. Five forcing durations (100, 
200, 300, 400, 500 time units) were tested with five replicates each. Decay 
timescale τ was measured by fitting E(t) = Ae^(-t/τ) to the post-forcing 
energy data.

[Vorticity Evolution Video Example](https://youtu.be/ZThQgZiqjNY)

## Repository Structure
```
├── simulation.py            # Main pseudospectral Navier-Stokes solver
├── taylor_green_vortex.py   # Taylor-Green vortex validation script
├── timescale.py             # Creates bar graph of decay timescale values with error bars
├── analysis.py              # Calculates the decay timescale
├── anova.py                 # One-way ANOVA statistical test
├── plotting.py              # Energy, enstrophy, and spectrum plots
├── vort_plot.py             # Vorticity snapshot plots
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Dependencies
```
numpy
scipy
matplotlib
```

Install with:
```bash
pip install numpy scipy matplotlib
```

## Key Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| N | 256 | Grid resolution (256×256) |
| L | 2π | Domain size |
| ν | 0.001 | Viscosity |
| dt | 0.01 | Timestep |
| T | 1000 | Total simulation time |
| forcing_scale | 6 | Wavenumber of forcing |
| F_amp | 200 | Forcing amplitude |
| T1 | 50 | Forcing start time |

## Simulation Overview
The solver implements:
- **Pseudospectral method** using Fast Fourier Transform (FFT) for efficient computation of spatial derivatives
- **RK4 timestepping** for accurate time integration
- **2/3 dealiasing rule** to prevent nonlinear aliasing errors
- **Transient forcing** applied between T1 and T2

## Validation
The simulation was validated against the Taylor-Green vortex, achieving a maximum percentage error of **0.002%**. Run ```taylor_green.py``` to perform the test.
