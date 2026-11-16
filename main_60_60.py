"""
Triplet Experiment Simulation: Constant Relative Launch Velocity (60 km/s)
===================================================================================
This script models the relativistic time dilation of three observers (A, B, and C) 
in a heliocentric (Sun-centered) coordinate system. 

Physics Overview:
Instead of calculating time dilation purely based on relative velocities between 
the observers (which leads to the Twin Paradox), this simulation integrates the 
proper time (tau) of each observer relative to the solar system's coordinate time (T).
The integration includes both:
1. Gravitational time dilation: Phi_sun(R) / c^2
2. Kinematic time dilation: -V(T)^2 / (2c^2)

Complementary Scenario:
- Observer A remains on Earth (approx. 30 km/s orbital speed).
- Observer B stays on Earth for 1 full orbit, then departs at 60 km/s RELATIVE to Earth.
- Observer C stays on Earth for 0.5 orbits, then departs at 60 km/s RELATIVE to Earth.
Because Earth's velocity vector points in opposite directions at 0.5 and 1.0 orbits, 
launching B and C with the exact same relative velocity (60 km/s) from Earth results 
in drastically different absolute heliocentric velocities (approx. 30 km/s and 90 km/s).
Consequently, they experience completely asymmetric time dilation.

Author: Bart Leplae
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt

# --- 1. Set Constants ---
G = 6.67430e-11           # Gravitational constant [m^3 kg^-1 s^-2]
M_sun = 1.989e30          # Mass of the Sun [kg]
AU = 1.495978707e11       # Astronomical Unit [m]
c = 299792458.0           # Speed of light [m/s]
DAY_S = 86400             # Seconds in a day
YEAR_S = 365.25 * DAY_S   # Seconds in a year
HALF_YEAR_S = YEAR_S / 2

# Angular velocity of Earth for a perfect circular orbit
omega = np.sqrt(G * M_sun / AU**3)

# Exact orbital period for 1 full revolution around the Sun
ORBIT_S = 2 * np.pi / omega
HALF_ORBIT_S = ORBIT_S / 2

# Simulation settings
total_days = 900
t = np.arange(0, total_days * DAY_S, DAY_S)
dT = DAY_S

# --- 2. Calculate Positions R(T) and Velocities V(T) ---

r_A = np.column_stack((-AU * np.sin(omega * t), AU * np.cos(omega * t)))
v_A = np.column_stack((-AU * omega * np.cos(omega * t), -AU * omega * np.sin(omega * t)))

v_launch_rel_x = 60000.0  # 60 km/s in m/s

# Observer B
r_B = np.copy(r_A)
v_B = np.copy(v_A)
idx_one_orbit = np.argmax(t >= ORBIT_S)
t_after_B = t[idx_one_orbit:] - t[idx_one_orbit]
v_B_helio = v_A[idx_one_orbit] + np.array([v_launch_rel_x, 0.0])
r_B[idx_one_orbit:, 0] = r_A[idx_one_orbit, 0] + v_B_helio[0] * t_after_B
r_B[idx_one_orbit:, 1] = r_A[idx_one_orbit, 1] + v_B_helio[1] * t_after_B
v_B[idx_one_orbit:, 0] = v_B_helio[0]
v_B[idx_one_orbit:, 1] = v_B_helio[1]

# Observer C
r_C = np.copy(r_A)
v_C = np.copy(v_A)
idx_half_orbit = np.argmax(t >= HALF_ORBIT_S)
t_after_C = t[idx_half_orbit:] - t[idx_half_orbit]
v_C_helio = v_A[idx_half_orbit] + np.array([v_launch_rel_x, 0.0])
r_C[idx_half_orbit:, 0] = r_A[idx_half_orbit, 0] + v_C_helio[0] * t_after_C
r_C[idx_half_orbit:, 1] = r_A[idx_half_orbit, 1] + v_C_helio[1] * t_after_C
v_C[idx_half_orbit:, 0] = v_C_helio[0]
v_C[idx_half_orbit:, 1] = v_C_helio[1]

# --- 3. Verify Departure Velocities ---
v_rel_B = np.linalg.norm(v_B[idx_one_orbit] - v_A[idx_one_orbit]) / 1000.0
v_rel_C = np.linalg.norm(v_C[idx_half_orbit] - v_A[idx_half_orbit]) / 1000.0

# --- 4. Relativistic Proper Time Calculation ---
def calc_proper_time_dilation(r_vec, v_vec, dT):
    R_T = np.linalg.norm(r_vec, axis=1)
    V_T_sq = np.sum(v_vec**2, axis=1)
    Phi_sun = -G * M_sun / R_T
    rate_deviation = (Phi_sun / c**2) - (V_T_sq / (2 * c**2))
    cum_tau_minus_T = np.zeros_like(R_T)
    cum_tau_minus_T[1:] = np.cumsum(0.5 * (rate_deviation[:-1] + rate_deviation[1:]) * dT)
    delta_tau = (t - t[0]) + cum_tau_minus_T
    return rate_deviation, cum_tau_minus_T, delta_tau

rate_A, tau_diff_A, tau_A = calc_proper_time_dilation(r_A, v_A, dT)
rate_B, tau_diff_B, tau_B = calc_proper_time_dilation(r_B, v_B, dT)
rate_C, tau_diff_C, tau_C = calc_proper_time_dilation(r_C, v_C, dT)

# --- 5. Compare with Earth ---
delta_tau_B_vs_Earth = tau_diff_B - tau_diff_A
delta_tau_C_vs_Earth = tau_diff_C - tau_diff_A
delta_tau_A_vs_Earth = np.zeros_like(t)

# --- 6. Create Plots ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 13), gridspec_kw={"height_ratios": [2.2, 1, 1]})

# Subplot 1: Spatial Map
ax1.set_aspect("equal") 
ax1.set_xlim(-3, 49)  # Expanded specifically to ensure C's 9 AU arrow tip at 46 AU is clearly visible
ax1.set_ylim(-8, 8)
ax1.set_xlabel("Heliocentric X Position (AU)")
ax1.set_ylabel("Heliocentric Y Position (AU)")
ax1.set_title("Heliocentric Frame: Positions $R(T)$ & Velocities $V(T)$ (1 AU arrow = 10,000 m/s)")
ax1.plot(0, 0, "yo", markersize=10, label="Sun")

ax1.plot(r_A[:, 0]/AU, r_A[:, 1]/AU, "r-", alpha=0.5, label="Observer A (Earth)")
ax1.plot(r_B[:, 0]/AU, r_B[:, 1]/AU, "b-", alpha=0.5, label=f"Observer B (Launched from Earth at {v_rel_B:.0f} km/s)")
ax1.plot(r_C[:, 0]/AU, r_C[:, 1]/AU, "g-", alpha=0.5, label=f"Observer C (Launched from Earth at {v_rel_C:.0f} km/s)")

ax1.plot(r_A[-1, 0]/AU, r_A[-1, 1]/AU, "ro")
ax1.plot(r_B[-1, 0]/AU, r_B[-1, 1]/AU, "bo")
ax1.plot(r_C[-1, 0]/AU, r_C[-1, 1]/AU, "go")

theta_circ = np.linspace(0, 2*np.pi, 200)
R_B_final = np.linalg.norm(r_B[-1]) / AU
R_C_final = np.linalg.norm(r_C[-1]) / AU
ax1.plot(R_B_final * np.cos(theta_circ), R_B_final * np.sin(theta_circ), color="darkblue", linestyle="--", alpha=0.3, linewidth=1)
ax1.plot(R_C_final * np.cos(theta_circ), R_C_final * np.sin(theta_circ), color="darkgreen", linestyle="--", alpha=0.3, linewidth=1)

vector_scale = 10000.0
q_args = dict(scale=1, scale_units="xy", angles="xy")
ax1.quiver(r_A[-1, 0]/AU, r_A[-1, 1]/AU, v_A[-1, 0]/vector_scale, v_A[-1, 1]/vector_scale, color="red", width=0.003, zorder=5, **q_args)
ax1.quiver(r_B[-1, 0]/AU, r_B[-1, 1]/AU, v_B[-1, 0]/vector_scale, v_B[-1, 1]/vector_scale, color="darkblue", width=0.004, zorder=5, **q_args)
ax1.quiver(r_C[-1, 0]/AU, r_C[-1, 1]/AU, v_C[-1, 0]/vector_scale, v_C[-1, 1]/vector_scale, color="darkgreen", width=0.004, zorder=5, **q_args)

ax1.legend(loc="upper center", fontsize=9) # Moved away from left edge to avoid covering Observer B
ax1.grid(True, alpha=0.5)

# Subplot 2: Instantaneous Integrand
ax2.set_xlim(0, total_days)
min_rate = min(np.min(rate_A), np.min(rate_B), np.min(rate_C)) * 1e8
# Set Y-axis upper limit to precisely 0
ax2.set_ylim(min_rate * 1.15, 0)
ax2.plot(t / DAY_S, rate_A * 1e8, "r--", label="Observer A (Earth)")
ax2.plot(t / DAY_S, rate_B * 1e8, "b-", label=f"Observer B (Launched from Earth at {v_rel_B:.0f} km/s)")
ax2.plot(t / DAY_S, rate_C * 1e8, "g-", label=f"Observer C (Launched from Earth at {v_rel_C:.0f} km/s)")
ax2.set_ylabel(r"$\frac{d\tau}{dT} - 1$ ($\times 10^{-8}$ s/s)")
ax2.set_title(r"Instantaneous Clock Rate vs Coordinate Time $T$: $\frac{\Phi_{sun}(R(T))}{c^2} - \frac{V(T)^2}{2c^2}$")
ax2.legend(loc="lower right", fontsize=9)
ax2.grid(True, alpha=0.5)

# Subplot 3: Accumulated Time Dilation
ax3.set_xlim(0, total_days)
min_dtau = min(np.min(delta_tau_B_vs_Earth * 1000), np.min(delta_tau_C_vs_Earth * 1000))
# Set Y-axis upper limit to precisely 500
ax3.set_ylim(min_dtau * 1.15, 500)
ax3.plot(t / DAY_S, delta_tau_A_vs_Earth * 1000, "r--", label="Observer A (Reference = 0)")
ax3.plot(t / DAY_S, delta_tau_B_vs_Earth * 1000, "b-", label=r"Observer B ($\Delta\tau_B - \Delta\tau_{Earth}$)")
ax3.plot(t / DAY_S, delta_tau_C_vs_Earth * 1000, "g-", label=r"Observer C ($\Delta\tau_C - \Delta\tau_{Earth}$)")
ax3.set_xlabel("Heliocentric Coordinate Time $T$ (days)")
ax3.set_ylabel(r"$\Delta\tau_{obs} - \Delta\tau_{Earth}$ (ms)")
ax3.set_title("Accumulated Time Dilation Relative to Earth")
ax3.legend(loc="lower left", fontsize=9)
ax3.grid(True, alpha=0.5)

plt.tight_layout()
plt.savefig("triplet_heliocentric_graph_60_60.png", dpi=300, bbox_inches="tight")