"""
Equatorial Balloon Thought Experiment: Heliocentric Time Dilation
=================================================================
This script analyzes an observer in a balloon floating stationary over 
the Earth's equator (rotating at ~465 m/s) from the Heliocentric frame.

Updates in this version:
- Plots the instantaneous clock rate deviation, the pure, un-inverted 
  Relativity of Simultaneity phase shift to show the perfect theoretical "mirror" 
  around the x-axis, and isolates the exact mathematical difference between the 
  proper time integral and the RoS mirror, proving the gap is exactly the constant 
  time dilation caused by the balloon's 465 m/s equatorial velocity (-104 ns/day),
  all combined into a single 3-panel figure.

Author: Bart Leplae
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt

# --- 1. Set Constants ---
G = 6.67430e-11           
M_sun = 1.989e30          
AU = 1.495978707e11       
R_earth = 6378137.0       
c = 299792458.0           
DAY_S = 86400             

omega_orb = np.sqrt(G * M_sun / AU**3)  
omega_rot = 2 * np.pi / DAY_S           

# Simulation settings: 3 Days
total_days = 3
dT = 60
t = np.arange(0, total_days * DAY_S, dT)
t_hours = t / 3600.0

# --- 2. Calculate Kinematics ---

# Earth Center (Baseline Orbit)
R_E = np.column_stack((AU * np.cos(omega_orb * t), AU * np.sin(omega_orb * t)))
V_E = np.column_stack((-AU * omega_orb * np.sin(omega_orb * t), AU * omega_orb * np.cos(omega_orb * t)))

# Balloon relative to Earth Center (Rotating at 465 m/s at Equator)
r_B = np.column_stack((R_earth * np.cos(omega_rot * t), R_earth * np.sin(omega_rot * t)))
v_B = np.column_stack((-R_earth * omega_rot * np.sin(omega_rot * t), R_earth * omega_rot * np.cos(omega_rot * t)))

R_abs = R_E + r_B
V_abs = V_E + v_B

# --- 3. Relativistic Proper Time & RoS Calculation ---

def calc_components(r_vec, v_vec):
    R_mag = np.linalg.norm(r_vec, axis=1)
    V_mag_sq = np.sum(v_vec**2, axis=1)
    
    rate_grav = (-G * M_sun / R_mag) / c**2
    rate_kin = -V_mag_sq / (2 * c**2)
    
    return rate_grav + rate_kin, rate_grav, rate_kin

combined_E, grav_E, kin_E = calc_components(R_E, V_E)
combined_B, grav_B, kin_B = calc_components(R_abs, V_abs)

# Accumulated proper time difference
dev_combined = combined_B - combined_E
cum_tau_diff = np.zeros_like(t, dtype=float)
cum_tau_diff[1:] = np.cumsum(0.5 * (dev_combined[:-1] + dev_combined[1:]) * dT)

# Instantaneous Relativity of Simultaneity Phase Shift (v * x / c^2)
ros_shift = np.sum(V_E * r_B, axis=1) / c**2

# Calculate the constant drift caused purely by the balloon's rotation speed
v_balloon_sq = (R_earth * omega_rot)**2
constant_drift = -(v_balloon_sq / (2 * c**2)) * t

# --- 4. Create Plots ---

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 15), gridspec_kw={'height_ratios': [1, 1, 1]})

# Subplot 1: Instantaneous Rates (Relative to Coordinate Time T)
ax1.plot(t_hours, grav_B * 1e10, 'g-', linewidth=2, label=r'Gravitational Penalty ($\Phi_{sun}/c^2$)')
ax1.plot(t_hours, kin_B * 1e10, 'b-', linewidth=2, label=r'Kinematic Penalty ($-V^2/2c^2$)')
ax1.plot(t_hours, combined_B * 1e10, 'r--', linewidth=2, label='Combined Clock Rate Deviation')

ax1.set_xlim(0, total_days * 24)
ax1.set_ylim(-165, 0)  # explicitly set to keep the top empty
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel(r'$\frac{d\tau}{dT} - 1$ ($\times 10^{-10}$ s/s)')
ax1.set_title('Instantaneous Clock Rate Deviation of the Balloon vs Heliocentric Time $T$')
ax1.legend(loc='upper right', fontsize=10) # Moved to empty top space
ax1.grid(True, alpha=0.5)

# Moved text upwards to avoid the data lines (blue is at ~ -50, green is at ~ -98)
ax1.text(2, -88, "Gravity oscillates slightly as balloon\nmoves closer/further from the Sun.", color='green', fontsize=9)
ax1.text(2, -35, "Kinematic penalty oscillates massively as balloon\nrotates into and against Earth's 30 km/s orbit.", color='blue', fontsize=9)


# Subplot 2: The Mirror Effect
ax2.plot(t_hours, cum_tau_diff * 1e6, 'purple', linewidth=2, label=r'Accumulated Proper Time ($\Delta\tau_{Balloon} - \Delta\tau_{EarthCenter}$)')
ax2.plot(t_hours, ros_shift * 1e6, 'orange', linestyle='--', linewidth=2, label=r'RoS Phase Shift ($+v \cdot x / c^2$)')

ax2.set_xlim(0, total_days * 24)
ax2.set_ylim(-3, 3)
ax2.set_xlabel('Time (hours)')
ax2.set_ylabel(r'$\Delta\tau$ (microseconds)')
ax2.set_title('The Mirror Effect: Proper Time vs Relativity of Simultaneity')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.5)

ax2.text(2, -2.7, f"The proper time penalty (-vx/c²) perfectly mirrors the spatial phase shift (+vx/c²).\n"
                  f"Because they are equal and opposite, the balloon's physical location is preserved.", 
         fontsize=10, bbox=dict(facecolor='white', alpha=0.9, edgecolor='purple'))


# Subplot 3: Isolating the Constant Balloon Velocity Drift
ax3.plot(t_hours, (cum_tau_diff + ros_shift) * 1e9, 'red', linewidth=2, label=r'Mathematical Residual (Proper Time + RoS)')
ax3.plot(t_hours, constant_drift * 1e9, 'black', linestyle=':', linewidth=3, label=r'Theoretical Balloon Velocity Drift ($-v_{rot}^2/2c^2$)')

ax3.set_xlim(0, total_days * 24)
ax3.set_xlabel('Time (hours)')
ax3.set_ylabel(r'Drift (nanoseconds)')
ax3.set_title('Isolating the Constant Velocity Penalty of the Balloon (~104 ns/day)')
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(True, alpha=0.5)

ax3.text(2, -250, f"By adding the two mirror curves together, the massive 2.1 µs sine wave is canceled out.\n"
                  f"The only thing left is a perfectly straight line drifting at -104 nanoseconds per day.\n"
                  f"This gap is exactly the constant kinematic time dilation caused by the balloon's 465 m/s speed!", 
         fontsize=10, bbox=dict(facecolor='white', alpha=0.9, edgecolor='red'))

plt.tight_layout()
plt.savefig("equator_balloon.png", dpi=300, bbox_inches="tight")