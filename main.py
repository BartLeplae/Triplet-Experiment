import numpy as np
import matplotlib.pyplot as plt

# --- 1. Set Constants ---
G = 6.67430e-11           # Gravitational constant [m^3 kg^-1 s^-2]
M_sun = 1.989e30          # Mass of the Sun [kg]
AU = 1.495978707e11       # Astronomical Unit [m]
DAY_S = 86400             # Seconds in a day
YEAR_S = 365.25 * DAY_S   # Seconds in a year
HALF_YEAR_S = YEAR_S / 2

# Angular velocity of Earth for a perfect circular orbit
omega = np.sqrt(G * M_sun / AU**3)

# Simulation settings (e.g., 2 years to give C enough time to travel)
total_days = 365 * 3
t = np.arange(0, total_days * DAY_S, DAY_S)

# --- 2. Calculate Positions and Velocities ---

# Observer A (Earth)
# Starts at x=0, y=AU. Counter-clockwise means movement towards negative x.
# Angle theta at t=0 is pi/2.
theta_A = np.pi/2 + omega * t
r_A = np.column_stack((-AU * np.sin(omega * t), AU * np.cos(omega * t)))
# The derivative of position is velocity
v_A = np.column_stack((-AU * omega * np.cos(omega * t), -AU * omega * np.sin(omega * t)))

# Observer B (Leaves Earth at t=0 with 60 km/s along the x-axis)
v_const = 60000 # 60 km/s in m/s
r_B = np.column_stack((v_const * t, np.full_like(t, AU)))
v_B = np.column_stack((np.full_like(t, v_const), np.zeros_like(t)))

# Observer C (Leaves Earth after 0.5 years with 60 km/s along the x-axis)
# First, copy A's path (stays on Earth for half a year)
r_C = np.copy(r_A)
v_C = np.copy(v_A)

# Find the index in the time array where t >= half a year
idx_half_year = np.argmax(t >= HALF_YEAR_S)
t_after = t[idx_half_year:] - t[idx_half_year]

# From this moment on, C departs in a straight line from x=0, y=-AU
r_C[idx_half_year:, 0] = v_const * t_after
r_C[idx_half_year:, 1] = -AU
v_C[idx_half_year:, 0] = v_const
v_C[idx_half_year:, 1] = 0

# --- 3. Calculate Virtual Objects ---

def calc_virtual_velocity(r_vec):
    """Calculates the velocity vector of a perfect circular orbit for an array of positions."""
    r_mag = np.linalg.norm(r_vec, axis=1)
    v_circ = np.sqrt(G * M_sun / r_mag)
    # Direction is the tangent, counter-clockwise
    tangent = np.column_stack((-r_vec[:, 1]/r_mag, r_vec[:, 0]/r_mag))
    # Multiply magnitude by the direction vector
    return tangent * v_circ[:, np.newaxis]

v_virt_A = calc_virtual_velocity(r_A)
v_virt_B = calc_virtual_velocity(r_B)
v_virt_C = calc_virtual_velocity(r_C)

# --- 4. Vector Differences & Storage ---

diff_v_A = v_A - v_virt_A
diff_v_B = v_B - v_virt_B
diff_v_C = v_C - v_virt_C

# Store in a Numpy array (shape: 3 observers x N days x 2 components)
# Index 0 = A, Index 1 = B, Index 2 = C
vector_differences = np.stack([diff_v_A, diff_v_B, diff_v_C])
# (Optional) Save to file: np.save('vector_differences.npy', vector_differences)

# --- 5. Create Plot ---

# To visualize this properly, we plot the magnitude of the velocity difference
mag_diff_A = np.linalg.norm(vector_differences[0], axis=1) / 1000 # in km/s
mag_diff_B = np.linalg.norm(vector_differences[1], axis=1) / 1000
mag_diff_C = np.linalg.norm(vector_differences[2], axis=1) / 1000

plt.figure(figsize=(10, 6))
plt.plot(t / DAY_S, mag_diff_A, label='Observer A (Remains on Earth)', linestyle='--')
plt.plot(t / DAY_S, mag_diff_B, label='Observer B (Departs on day 0)')
plt.plot(t / DAY_S, mag_diff_C, label='Observer C (Departs around day 182)')

plt.xlabel('Time (days)')
plt.ylabel('Velocity difference magnitude (km/s)')
plt.title('Velocity Difference Relative to Virtual Object in Circular Orbit')
plt.legend()
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig('velocity_difference_plot.png', dpi=300, bbox_inches='tight')
plt.show()