import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- 1. Set Constants ---
G = 6.67430e-11           
AU = 1.495978707e11       
DAY_S = 86400             
YEAR_S = 365.25 * DAY_S   
HALF_YEAR_S = YEAR_S / 2  
c = 299792458.0           

# Adjust M_sun so that the perfect circular orbit speed at 1 AU is EXACTLY 30 km/s
M_sun = (30000**2) * AU / G  

omega = np.sqrt(G * M_sun / AU**3)

# Reduced simulation length to fit a 35 AU x-axis
total_days = 1000
t = np.arange(0, total_days * DAY_S, DAY_S)

# --- 2. Calculate Positions and Velocities ---

r_A = np.column_stack((-AU * np.sin(omega * t), AU * np.cos(omega * t)))
v_A = np.column_stack((-AU * omega * np.cos(omega * t), -AU * omega * np.sin(omega * t)))

v_const = 60000 
r_B = np.column_stack((v_const * t, np.full_like(t, AU)))
v_B = np.column_stack((np.full_like(t, v_const), np.zeros_like(t)))

r_C = np.copy(r_A)
v_C = np.copy(v_A)

idx_half_year = np.argmax(t >= HALF_YEAR_S)
t_after = t[idx_half_year:] - t[idx_half_year]

r_C[idx_half_year:, 0] = v_const * t_after
r_C[idx_half_year:, 1] = -AU
v_C[idx_half_year:, 0] = v_const
v_C[idx_half_year:, 1] = 0

# --- 3. Calculate Virtual Objects ---

def calc_virtual_velocity(r_vec):
    r_mag = np.linalg.norm(r_vec, axis=1)
    v_circ = np.sqrt(G * M_sun / r_mag)
    tangent = np.column_stack((-r_vec[:, 1]/r_mag, r_vec[:, 0]/r_mag))
    return tangent * v_circ[:, np.newaxis]

v_virt_B = calc_virtual_velocity(r_B)
v_virt_C = calc_virtual_velocity(r_C)

# --- 4. Vector Differences & Clock Rate Calculation ---

diff_v_B = v_B - v_virt_B 
diff_v_C = v_C - v_virt_C

mag_diff_B = np.linalg.norm(diff_v_B, axis=1) 
mag_diff_C = np.linalg.norm(diff_v_C, axis=1) 

rate_diff_B = (mag_diff_B**2 / (2 * c**2)) * 1e10
rate_diff_C = (mag_diff_C**2 / (2 * c**2)) * 1e10

# --- 5. Create Animation ---

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [2, 1]})

# Subplot 1: Spatial Map
ax1.set_aspect('equal') 
# Zoomed in axes
ax1.set_xlim(-3, 35)
ax1.set_ylim(-10, 10)
ax1.set_xlabel('X Position (AU)')
ax1.set_ylabel('Y Position (AU)')
ax1.set_title('Observer Positions & Physical Vector Subtraction (1 AU = 10,000 m/s)')
ax1.plot(0, 0, 'yo', markersize=10)

theta_circ = np.linspace(0, 2*np.pi, 200)
circle_B, = ax1.plot([], [], color='darkblue', linestyle='--', alpha=0.3, linewidth=1)
circle_C, = ax1.plot([], [], color='darkgreen', linestyle='--', alpha=0.3, linewidth=1)

line_A, = ax1.plot([], [], 'r-', alpha=0.5)
line_B, = ax1.plot([], [], 'b-', alpha=0.5)
line_C, = ax1.plot([], [], 'g-', alpha=0.5)

pt_A, = ax1.plot([], [], 'ro')
pt_B, = ax1.plot([], [], 'bo')
pt_C, = ax1.plot([], [], 'go')

vector_scale = 10000 
q_args = dict(scale=1, scale_units='xy', angles='xy')

# Historical Traces
trace_step = 30
h_idx_all = np.arange(0, len(t), trace_step)
num_traces = len(h_idx_all)

Q_B_trace = ax1.quiver(np.zeros(num_traces), np.zeros(num_traces), 
                       np.zeros(num_traces), np.zeros(num_traces), 
                       color='blue', alpha=0.2, width=0.0015, zorder=3, **q_args)
Q_C_trace = ax1.quiver(np.zeros(num_traces), np.zeros(num_traces), 
                       np.zeros(num_traces), np.zeros(num_traces), 
                       color='green', alpha=0.2, width=0.0015, zorder=3, **q_args)

# Current Vector Components
Q_B_obs = ax1.quiver([0], [0], [0], [0], color='blue', alpha=0.4, width=0.002, zorder=4, **q_args)
Q_B_sub_virt = ax1.quiver([0], [0], [0], [0], color='darkorange', alpha=0.5, width=0.002, zorder=4, **q_args)
Q_B = ax1.quiver([0], [0], [0], [0], color='darkblue', width=0.004, zorder=5, **q_args)

Q_C_obs = ax1.quiver([0], [0], [0], [0], color='green', alpha=0.4, width=0.002, zorder=4, **q_args)
Q_C_sub_virt = ax1.quiver([0], [0], [0], [0], color='darkorange', alpha=0.5, width=0.002, zorder=4, **q_args)
Q_C = ax1.quiver([0], [0], [0], [0], color='darkgreen', width=0.004, zorder=5, **q_args)

# Local Inertial Frame Vectors
Q_virt_B = ax1.quiver([0], [0], [0], [0], color='darkorange', width=0.003, zorder=4, **q_args)
Q_virt_C = ax1.quiver([0], [0], [0], [0], color='darkorange', width=0.003, zorder=4, **q_args)

t_label_B = ax1.text(0, 0, '', color='darkblue', fontsize=10, weight='bold')
t_label_C = ax1.text(0, 0, '', color='darkgreen', fontsize=10, weight='bold')

# Subplot 2: Clock Rate Difference
ax2.set_xlim(0, 36) # Zoomed to match maximum distance
ax2.set_ylim(0, max(np.max(rate_diff_B), np.max(rate_diff_C)) * 1.1)
ax2.set_xlabel('Distance from Sun (AU)')
ax2.set_ylabel('Clock Rate Diff. w/ A (x10^-10 s/s)')
ax2.set_title('Relative Kinematic Time Dilation')
ax2.grid(True, alpha=0.5)

line_diff_B, = ax2.plot([], [], 'b-', label='Observer B')
line_diff_C, = ax2.plot([], [], 'g-', label='Observer C')
ax2.legend(loc='upper right')

time_template = 'Time = %.1f days'
time_text = ax1.text(0.02, 0.93, '', transform=ax1.transAxes, fontsize=10, weight='bold')

def init():
    return (line_A, line_B, line_C, circle_B, circle_C, pt_A, pt_B, pt_C, 
            line_diff_B, line_diff_C, t_label_B, t_label_C, time_text, 
            Q_B, Q_B_obs, Q_B_sub_virt, Q_C, Q_C_obs, Q_C_sub_virt, Q_B_trace, Q_C_trace,
            Q_virt_B, Q_virt_C)

def animate(i):
    # Update Spatial Trajectories
    line_A.set_data(r_A[:i+1, 0]/AU, r_A[:i+1, 1]/AU)
    line_B.set_data(r_B[:i+1, 0]/AU, r_B[:i+1, 1]/AU)
    line_C.set_data(r_C[:i+1, 0]/AU, r_C[:i+1, 1]/AU)
    
    # Update Virtual Object Dashed Circles
    r_mag_B = np.linalg.norm(r_B[i]) / AU
    circle_B.set_data(r_mag_B * np.cos(theta_circ), r_mag_B * np.sin(theta_circ))
    r_mag_C = np.linalg.norm(r_C[i]) / AU
    circle_C.set_data(r_mag_C * np.cos(theta_circ), r_mag_C * np.sin(theta_circ))

    # Current Coordinates in AU
    x_B, y_B = r_B[i, 0]/AU, r_B[i, 1]/AU
    x_C, y_C = r_C[i, 0]/AU, r_C[i, 1]/AU
    
    pt_A.set_data([r_A[i, 0]/AU], [r_A[i, 1]/AU])
    pt_B.set_data([x_B], [y_B])
    pt_C.set_data([x_C], [y_C])

    # Update Historical Traces
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
        u_B_trace[:num_active] = diff_v_B[active_idx, 0] / vector_scale
        v_B_trace[:num_active] = diff_v_B[active_idx, 1] / vector_scale
        
        offsets_C[:num_active] = r_C[active_idx] / AU
        u_C_trace[:num_active] = diff_v_C[active_idx, 0] / vector_scale
        v_C_trace[:num_active] = diff_v_C[active_idx, 1] / vector_scale
        
    Q_B_trace.set_offsets(offsets_B)
    Q_B_trace.set_UVC(u_B_trace, v_B_trace)
    Q_C_trace.set_offsets(offsets_C)
    Q_C_trace.set_UVC(u_C_trace, v_C_trace)

    # Calculate absolute velocities scaled to AU
    u_B_obs, v_B_obs = v_B[i, 0]/vector_scale, v_B[i, 1]/vector_scale
    u_C_obs, v_C_obs = v_C[i, 0]/vector_scale, v_C[i, 1]/vector_scale
    
    u_B_virt, v_B_virt = v_virt_B[i, 0]/vector_scale, v_virt_B[i, 1]/vector_scale
    u_C_virt, v_C_virt = v_virt_C[i, 0]/vector_scale, v_virt_C[i, 1]/vector_scale

    # Update Current Velocity Vectors and Components for B
    Q_B_obs.set_offsets(np.c_[x_B, y_B])
    Q_B_obs.set_UVC(u_B_obs, v_B_obs)
    Q_B_sub_virt.set_offsets(np.c_[x_B + u_B_obs, y_B + v_B_obs]) 
    Q_B_sub_virt.set_UVC(-u_B_virt, -v_B_virt)
    Q_B.set_offsets(np.c_[x_B, y_B])
    Q_B.set_UVC(u_B_obs - u_B_virt, v_B_obs - v_B_virt)
    
    # Update Current Velocity Vectors and Components for C
    Q_C_obs.set_offsets(np.c_[x_C, y_C])
    Q_C_obs.set_UVC(u_C_obs, v_C_obs)
    Q_C_sub_virt.set_offsets(np.c_[x_C + u_C_obs, y_C + v_C_obs]) 
    Q_C_sub_virt.set_UVC(-u_C_virt, -v_C_virt)
    Q_C.set_offsets(np.c_[x_C, y_C])
    Q_C.set_UVC(u_C_obs - u_C_virt, v_C_obs - v_C_virt)

    # Update Local Inertial Frame (Virtual Object) Vectors
    Q_virt_B.set_offsets(np.c_[x_B, y_B])
    Q_virt_B.set_UVC(u_B_virt, v_B_virt)

    Q_virt_C.set_offsets(np.c_[x_C, y_C])
    Q_virt_C.set_UVC(u_C_virt, v_C_virt)

    # Update Text Labels
    t_label_B.set_position((x_B, y_B + 1.5))
    t_label_B.set_text(f'{mag_diff_B[i]/1000:.1f} km/s')
    t_label_C.set_position((x_C, y_C - 2.5))
    t_label_C.set_text(f'{mag_diff_C[i]/1000:.1f} km/s')

    # Update Clock Rate Difference Line Graphs using Distance (AU) for the X-axis
    dist_B = np.linalg.norm(r_B[:i+1], axis=1) / AU
    dist_C = np.linalg.norm(r_C[:i+1], axis=1) / AU
    line_diff_B.set_data(dist_B, rate_diff_B[:i+1])
    line_diff_C.set_data(dist_C, rate_diff_C[:i+1])

    time_text.set_text(time_template % (t[i] / DAY_S))
    
    return (line_A, line_B, line_C, circle_B, circle_C, pt_A, pt_B, pt_C, 
            line_diff_B, line_diff_C, t_label_B, t_label_C, time_text, 
            Q_B, Q_B_obs, Q_B_sub_virt, Q_C, Q_C_obs, Q_C_sub_virt, Q_B_trace, Q_C_trace,
            Q_virt_B, Q_virt_C)

frame_step = 5 
frames = range(0, len(t), frame_step)

print("Generating zoomed 1000-day animation...")
ani = animation.FuncAnimation(fig, animate, frames=frames, 
                              interval=30, blit=True, init_func=init)

writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='Simulation'), bitrate=1800)
ani.save('triplet_animation.mp4', writer=writer)

print("Saved animation to 'triplet_animation.mp4'")