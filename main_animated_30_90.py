"""
Triplet Experiment Animation: Heliocentric Kinematic & Gravitational Time Dilation
===================================================================================
This script animates the relativistic time dilation of three observers (A, B, and C) 
in a heliocentric (Sun-centered) coordinate system. 

Physics Overview:
Instead of calculating time dilation purely based on relative velocities between 
the observers (which leads to the Twin Paradox), this simulation integrates the 
proper time (tau) of each observer relative to the solar system's coordinate time (T).
The integration includes both:
1. Gravitational time dilation: Phi_sun(R) / c^2
2. Kinematic time dilation: -V(T)^2 / (2c^2)

Scenario:
- Observer A remains on Earth (approx. 30 km/s orbital speed).
- Observer B stays on Earth for 1 full orbit, then departs along the positive x-axis.
- Observer C stays on Earth for 0.5 orbits, then departs along the positive x-axis.
Because Earth's velocity vector points in opposite directions at 0.5 and 1.0 orbits, 
launching B and C with the exact same heliocentric velocity (60 km/s) requires drastically 
different launch velocities relative to Earth (90 km/s and 30 km/s).

Author: Bart Leplae
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- 1. Set Constants ---
G = 6.67430e-11           
AU = 1.495978707e11       
DAY_S = 86400             
c = 299792458.0           

# Adjust M_sun so that the perfect circular orbit speed at 1 AU is EXACTLY 30 km/s
M_sun = (30000.0**2) * AU / G  

omega = np.sqrt(G * M_sun / AU**3)

# Exact orbital period for 1 full revolution around the Sun
ORBIT_S = 2 * np.pi / omega
HALF_ORBIT_S = ORBIT_S / 2

# Simulation length: 1365 days (1 orbit + ~1000 days of travel to reach 35 AU)
total_days = 1365
t = np.arange(0, total_days * DAY_S, DAY_S)
dT = DAY_S

# --- 2. Calculate Heliocentric Positions R(T) and Velocities V(T) ---

# Observer A (Earth)
r_A = np.column_stack((-AU * np.sin(omega * t), AU * np.cos(omega * t)))
v_A = np.column_stack((-AU * omega * np.cos(omega * t), -AU * omega * np.sin(omega * t)))

# Fixed absolute heliocentric departure velocity for B and C
v_const = 60000.0 

# Observer B (Stays on Earth for 1 full orbit, then departs at 60 km/s along x-axis)
r_B = np.copy(r_A)
v_B = np.copy(v_A)

idx_one_orbit = np.argmax(t >= ORBIT_S)
t_after_B = t[idx_one_orbit:] - t[idx_one_orbit]

r_B[idx_one_orbit:, 0] = v_const * t_after_B
r_B[idx_one_orbit:, 1] = AU
v_B[idx_one_orbit:, 0] = v_const
v_B[idx_one_orbit:, 1] = 0

# Observer C (Stays on Earth for 0.5 orbit, then departs at 60 km/s along x-axis)
r_C = np.copy(r_A)
v_C = np.copy(v_A)

idx_half_orbit = np.argmax(t >= HALF_ORBIT_S)
t_after_C = t[idx_half_orbit:] - t[idx_half_orbit]

r_C[idx_half_orbit:, 0] = v_const * t_after_C
r_C[idx_half_orbit:, 1] = -AU
v_C[idx_half_orbit:, 0] = v_const
v_C[idx_half_orbit:, 1] = 0

# --- 3. Calculate Departure Velocities Relative to Earth ---
v_rel_B = np.linalg.norm(v_B[idx_one_orbit] - v_A[idx_one_orbit]) / 1000.0
v_rel_C = np.linalg.norm(v_C[idx_half_orbit] - v_A[idx_half_orbit]) / 1000.0

print(f"Generating animation with relative launches: B={v_rel_B:.0f} km/s, C={v_rel_C:.0f} km/s")

# --- 4. Heliocentric Proper Time Integral ---

def calc_proper_time_dilation(r_vec, v_vec, dT):
    """
    Calculates heliocentric distance R(T), speed V(T), instantaneous clock rate 
    deviation d(tau)/dT - 1, and cumulative proper time difference (tau - T).
    """
    R_T = np.linalg.norm(r_vec, axis=1)
    V_T = np.linalg.norm(v_vec, axis=1)
    
    # Gravitational potential of the Sun (negative)
    Phi_sun = -G * M_sun / R_T
    
    # Integrand minus 1: Phi_sun(R(T))/c^2 - V(T)^2 / (2*c^2)
    rate_deviation = (Phi_sun / c**2) - (V_T**2 / (2 * c**2))
    
    # Cumulative trapezoidal integration over coordinate time T
    cum_tau_minus_T = np.zeros_like(R_T)
    cum_tau_minus_T[1:] = np.cumsum(0.5 * (rate_deviation[:-1] + rate_deviation[1:]) * dT)
    
    return R_T, V_T, rate_deviation, cum_tau_minus_T

R_A, V_A, rate_A, tau_diff_A = calc_proper_time_dilation(r_A, v_A, dT)
R_B, V_B, rate_B, tau_diff_B = calc_proper_time_dilation(r_B, v_B, dT)
R_C, V_C, rate_C, tau_diff_C = calc_proper_time_dilation(r_C, v_C, dT)

# --- 5. Compare with Earth (Delta tau_obs - Delta tau_Earth) ---

delta_tau_A_ms = np.zeros_like(t)
delta_tau_B_ms = (tau_diff_B - tau_diff_A) * 1000.0
delta_tau_C_ms = (tau_diff_C - tau_diff_A) * 1000.0

t_days = t / DAY_S

# --- 6. Create Animation ---

fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=(10, 13), gridspec_kw={'height_ratios': [2.2, 1, 1]}
)

# Subplot 1: Spatial Map (Heliocentric Frame)
ax1.set_aspect('equal') 
ax1.set_xlim(-3, 35)
ax1.set_ylim(-10, 10)
ax1.set_xlabel('Heliocentric X Position (AU)')
ax1.set_ylabel('Heliocentric Y Position (AU)')
ax1.set_title('Heliocentric Frame: Positions $R(T)$ & Velocities $V(T)$ (1 AU arrow = 10,000 m/s)')
ax1.plot(0, 0, 'yo', markersize=10, label='Sun')

theta_circ = np.linspace(0, 2*np.pi, 200)
circle_B, = ax1.plot([], [], color='darkblue', linestyle='--', alpha=0.3, linewidth=1)
circle_C, = ax1.plot([], [], color='darkgreen', linestyle='--', alpha=0.3, linewidth=1)

line_A, = ax1.plot([], [], 'r-', alpha=0.5, label='Observer A (Earth)')
line_B, = ax1.plot([], [], 'b-', alpha=0.5, label=f'Observer B (Launched from Earth at {v_rel_B:.0f} km/s)')
line_C, = ax1.plot([], [], 'g-', alpha=0.5, label=f'Observer C (Launched from Earth at {v_rel_C:.0f} km/s)')

pt_A, = ax1.plot([], [], 'ro')
pt_B, = ax1.plot([], [], 'bo')
pt_C, = ax1.plot([], [], 'go')

vector_scale = 10000.0
q_args = dict(scale=1, scale_units='xy', angles='xy')

# Historical Velocity Traces V(T)
trace_step = 30
h_idx_all = np.arange(0, len(t), trace_step)
num_traces = len(h_idx_all)

Q_B_trace = ax1.quiver(np.zeros(num_traces), np.zeros(num_traces), 
                       np.zeros(num_traces), np.zeros(num_traces), 
                       color='blue', alpha=0.2, width=0.0015, zorder=3, **q_args)
Q_C_trace = ax1.quiver(np.zeros(num_traces), np.zeros(num_traces), 
                       np.zeros(num_traces), np.zeros(num_traces), 
                       color='green', alpha=0.2, width=0.0015, zorder=3, **q_args)

# Current Heliocentric Velocity Vectors V(T)
Q_A_vel = ax1.quiver([0], [0], [0], [0], color='red', width=0.003, zorder=5, **q_args)
Q_B_vel = ax1.quiver([0], [0], [0], [0], color='darkblue', width=0.004, zorder=5, **q_args)
Q_C_vel = ax1.quiver([0], [0], [0], [0], color='darkgreen', width=0.004, zorder=5, **q_args)

t_label_B = ax1.text(0, 0, '', color='darkblue', fontsize=9, weight='bold')
t_label_C = ax1.text(0, 0, '', color='darkgreen', fontsize=9, weight='bold')
time_text = ax1.text(0.02, 0.92, '', transform=ax1.transAxes, fontsize=10, weight='bold')
ax1.legend(loc='lower right', fontsize=9)

# Subplot 2: Instantaneous Integrand (d_tau / dT - 1)
ax2.set_xlim(0, total_days)
min_rate = min(np.min(rate_A), np.min(rate_B), np.min(rate_C)) * 1e8
ax2.set_ylim(min_rate * 1.15, 0)
ax2.set_ylabel(r'$\frac{d\tau}{dT} - 1$ ($\times 10^{-8}$ s/s)')
ax2.set_title(r'Instantaneous Clock Rate vs Coordinate Time $T$: $\frac{\Phi_{sun}(R(T))}{c^2} - \frac{V(T)^2}{2c^2}$')
ax2.grid(True, alpha=0.5)

line_rate_A, = ax2.plot([], [], 'r--', label='Observer A (Earth)')
line_rate_B, = ax2.plot([], [], 'b-', label=f'Observer B (Launched from Earth at {v_rel_B:.0f} km/s)')
line_rate_C, = ax2.plot([], [], 'g-', label=f'Observer C (Launched from Earth at {v_rel_C:.0f} km/s)')
ax2.legend(loc='lower right', fontsize=9)

# Subplot 3: Accumulated Time Dilation relative to Earth (Delta tau_obs - Delta tau_Earth)
ax3.set_xlim(0, total_days)
min_dtau = min(np.min(delta_tau_B_ms), np.min(delta_tau_C_ms))
max_dtau = max(np.max(delta_tau_B_ms), np.max(delta_tau_C_ms))
ax3.set_ylim(min_dtau * 1.15, max(5.0, max_dtau * 1.15))
ax3.set_xlabel('Heliocentric Coordinate Time $T$ (days)')
ax3.set_ylabel(r'$\Delta\tau_{obs} - \Delta\tau_{Earth}$ (ms)')
ax3.set_title('Accumulated Time Dilation Relative to Earth')
ax3.grid(True, alpha=0.5)

line_dtau_A, = ax3.plot([], [], 'r--', label='Observer A (Reference = 0)')
line_dtau_B, = ax3.plot([], [], 'b-', label=r'Observer B ($\Delta\tau_B - \Delta\tau_{Earth}$)')
line_dtau_C, = ax3.plot([], [], 'g-', label=r'Observer C ($\Delta\tau_C - \Delta\tau_{Earth}$)')
ax3.legend(loc='lower left', fontsize=9)

plt.tight_layout()

def init():
    return (line_A, line_B, line_C, circle_B, circle_C, pt_A, pt_B, pt_C,
            Q_A_vel, Q_B_vel, Q_C_vel, Q_B_trace, Q_C_trace,
            t_label_B, t_label_C, time_text,
            line_rate_A, line_rate_B, line_rate_C,
            line_dtau_A, line_dtau_B, line_dtau_C)

def animate(i):
    # 1. Update Spatial Trajectories
    line_A.set_data(r_A[:i+1, 0]/AU, r_A[:i+1, 1]/AU)
    line_B.set_data(r_B[:i+1, 0]/AU, r_B[:i+1, 1]/AU)
    line_C.set_data(r_C[:i+1, 0]/AU, r_C[:i+1, 1]/AU)
    
    # 2. Update Equipotential Circles of radius R(T)
    r_mag_B = R_B[i] / AU
    circle_B.set_data(r_mag_B * np.cos(theta_circ), r_mag_B * np.sin(theta_circ))
    r_mag_C = R_C[i] / AU
    circle_C.set_data(r_mag_C * np.cos(theta_circ), r_mag_C * np.sin(theta_circ))

    # 3. Current Coordinates in AU
    x_A, y_A = r_A[i, 0]/AU, r_A[i, 1]/AU
    x_B, y_B = r_B[i, 0]/AU, r_B[i, 1]/AU
    x_C, y_C = r_C[i, 0]/AU, r_C[i, 1]/AU
    
    pt_A.set_data([x_A], [y_A])
    pt_B.set_data([x_B], [y_B])
    pt_C.set_data([x_C], [y_C])

    # 4. Update Historical Velocity Traces V(T)
    active_mask = h_idx_all <= i
    active_idx = h_idx_all[active_mask]
    num_active = len(active_idx)

    offsets_B = np.zeros((num_traces, 2))
    u_B_trace = np.zeros(num_traces)
    v_B_trace = np.zeros(num_traces)
    
    offsets_C = np.zeros((num_traces, 2))
    u_C_trace = np.zeros(num_traces)
    v_C_trace = np.zeros(num_traces)

    if num_active > 0:
        offsets_B[:num_active] = r_B[active_idx] / AU
        u_B_trace[:num_active] = v_B[active_idx, 0] / vector_scale
        v_B_trace[:num_active] = v_B[active_idx, 1] / vector_scale
        
        offsets_C[:num_active] = r_C[active_idx] / AU
        u_C_trace[:num_active] = v_C[active_idx, 0] / vector_scale
        v_C_trace[:num_active] = v_C[active_idx, 1] / vector_scale
        
    Q_B_trace.set_offsets(offsets_B)
    Q_B_trace.set_UVC(u_B_trace, v_B_trace)
    Q_C_trace.set_offsets(offsets_C)
    Q_C_trace.set_UVC(u_C_trace, v_C_trace)

    # 5. Update Current Heliocentric Velocity Vectors V(T)
    Q_A_vel.set_offsets(np.c_[x_A, y_A])
    Q_A_vel.set_UVC(v_A[i, 0]/vector_scale, v_A[i, 1]/vector_scale)

    Q_B_vel.set_offsets(np.c_[x_B, y_B])
    Q_B_vel.set_UVC(v_B[i, 0]/vector_scale, v_B[i, 1]/vector_scale)
    
    Q_C_vel.set_offsets(np.c_[x_C, y_C])
    Q_C_vel.set_UVC(v_C[i, 0]/vector_scale, v_C[i, 1]/vector_scale)

    # 6. Update Text Labels
    t_label_B.set_position((x_B, y_B + 1.2))
    t_label_B.set_text(
        f'R={r_mag_B:.1f} AU | V={V_B[i]/1000:.0f} km/s\n'
        f'Δτ - Δτ_Earth = {delta_tau_B_ms[i]:.1f} ms'
    )
    t_label_C.set_position((x_C, y_C - 2.8))
    t_label_C.set_text(
        f'R={r_mag_C:.1f} AU | V={V_C[i]/1000:.0f} km/s\n'
        f'Δτ - Δτ_Earth = {delta_tau_C_ms[i]:.1f} ms'
    )

    # 7. Update Rate and Cumulative Time Dilation Graphs
    line_rate_A.set_data(t_days[:i+1], rate_A[:i+1] * 1e8)
    line_rate_B.set_data(t_days[:i+1], rate_B[:i+1] * 1e8)
    line_rate_C.set_data(t_days[:i+1], rate_C[:i+1] * 1e8)

    line_dtau_A.set_data(t_days[:i+1], delta_tau_A_ms[:i+1])
    line_dtau_B.set_data(t_days[:i+1], delta_tau_B_ms[:i+1])
    line_dtau_C.set_data(t_days[:i+1], delta_tau_C_ms[:i+1])

    time_text.set_text(f'Coordinate Time T = {t_days[i]:.1f} days')
    
    return (line_A, line_B, line_C, circle_B, circle_C, pt_A, pt_B, pt_C,
            Q_A_vel, Q_B_vel, Q_C_vel, Q_B_trace, Q_C_trace,
            t_label_B, t_label_C, time_text,
            line_rate_A, line_rate_B, line_rate_C,
            line_dtau_A, line_dtau_B, line_dtau_C)

frame_step = 5 
frames = range(0, len(t), frame_step)

print("Generating 1365-day heliocentric time dilation animation...")
ani = animation.FuncAnimation(fig, animate, frames=frames, 
                              interval=30, blit=True, init_func=init)

writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='Simulation'), bitrate=1800)
ani.save('triplet_heliocentric_animation_30_90.mp4', writer=writer)

print("Saved animation to 'triplet_heliocentric_animation_30_90.mp4'")