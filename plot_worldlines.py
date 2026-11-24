import numpy as np
import matplotlib.pyplot as plt

# --- Physics Constants ---
G = 6.67430e-11
M_sun = 1.989e30
c = 299792458.0
AU = 1.495978707e11      # meters
YEAR = 365.25 * 86400.0  # seconds

# Simulation parameters matching README.md
v_earth = 30000.0        # 30 km/s orbital speed
omega = 2.0 * np.pi / YEAR
T_max_yr = 2.0
N = 4000

t_yr = np.linspace(0.0, T_max_yr, N)
t = t_yr * YEAR
dt = t[1] - t[0]

# Observer A (Earth): starts at (0, +1 AU), counter-clockwise circular orbit
xA = -AU * np.sin(omega * t)
yA = AU * np.cos(omega * t)
xA_au = xA / AU

def compute_scenario(vB_helio, vC_helio):
    """Compute x-trajectories (in AU) and cumulative proper time diffs (in ms) relative to Earth."""
    rateA = -G * M_sun / (AU * c**2) - v_earth**2 / (2.0 * c**2)
    tauA = np.cumsum(np.full_like(t, rateA)) * dt

    # Observer B: rides Earth until t = 1.0 yr (at x=0, y=+1 AU), then moves along +x at vB_helio
    xB = np.where(t_yr < 1.0, xA, vB_helio * (t - 1.0 * YEAR))
    yB = np.where(t_yr < 1.0, yA, AU)
    rB = np.sqrt(xB**2 + yB**2)
    vB = np.where(t_yr < 1.0, v_earth, vB_helio)
    rateB = -G * M_sun / (rB * c**2) - vB**2 / (2.0 * c**2)
    tauB = np.cumsum(rateB) * dt

    # Observer C: rides Earth until t = 0.5 yr (at x=0, y=-1 AU), then moves along +x at vC_helio
    xC = np.where(t_yr < 0.5, xA, vC_helio * (t - 0.5 * YEAR))
    yC = np.where(t_yr < 0.5, yA, -AU)
    rC = np.sqrt(xC**2 + yC**2)
    vC = np.where(t_yr < 0.5, v_earth, vC_helio)
    rateC = -G * M_sun / (rC * c**2) - vC**2 / (2.0 * c**2)
    tauC = np.cumsum(rateC) * dt

    dtauB_ms = (tauB - tauA) * 1e3
    dtauC_ms = (tauC - tauA) * 1e3
    return (xB / AU, dtauB_ms), (xC / AU, dtauC_ms)

# Scenario 1: 30/90 km/s relative launches -> both +60 km/s heliocentric
scen1_B, scen1_C = compute_scenario(60000.0, 60000.0)
# Scenario 2: 60/60 km/s relative launches -> B at +30 km/s, C at +90 km/s heliocentric
scen2_B, scen2_C = compute_scenario(30000.0, 90000.0)

# --- Build 2x1 Vertical Stacked 2D Spacetime Figure ---
fig, axes = plt.subplots(2, 1, figsize=(13.5, 10.8), dpi=150)
fig.patch.set_facecolor('white')

scenarios = [
    (
        "Scenario 1: Identical Heliocentric Speeds (60 km/s) — Relative Launches: B = 90 km/s, C = 30 km/s",
        scen1_B, scen1_C,
        "90 km/s", "60 km/s",
        "30 km/s", "60 km/s"
    ),
    (
        "Scenario 2: Identical Relative Launches (60 km/s vs. Earth) — Heliocentric Speeds: B = 30 km/s, C = 90 km/s",
        scen2_B, scen2_C,
        "60 km/s", "30 km/s",
        "60 km/s", "90 km/s"
    )
]

idx_05 = np.searchsorted(t_yr, 0.5)
idx_10 = np.searchsorted(t_yr, 1.0)

for row, (title, (xB_au, dtauB), (xC_au, dtauC), vB_rel, vB_helio, vC_rel, vC_helio) in enumerate(scenarios):
    ax = axes[row]

    ax.axvline(0, color='orange', linestyle='--', linewidth=1.7, alpha=0.85, label='Sun (x = 0)')
    ax.plot(xA_au, t_yr, color='#1f77b4', linewidth=2.5, label='Observer A (Earth: v_orb = 30 km/s)')
    ax.plot(
        xC_au[idx_05:], t_yr[idx_05:], color='#d62728', linewidth=2.7,
        label=f'Observer C (v_rel = {vC_rel} vs. Earth | v_helio = {vC_helio})'
    )
    ax.plot(
        xB_au[idx_10:], t_yr[idx_10:], color='#2ca02c', linewidth=2.7,
        label=f'Observer B (v_rel = {vB_rel} vs. Earth | v_helio = {vB_helio})'
    )

    # Mark launch events and annotate with relative launch speed
    ax.scatter([0, 0], [0.5, 1.0], color=['#d62728', '#2ca02c'], edgecolor='black', s=75, zorder=6)

    ax.annotate(
        f'Launch C (T = 0.5 yr)\nv_rel = {vC_rel} vs. Earth',
        xy=(0.12, 0.49), xytext=(2.6, 0.23),
        arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.4),
        fontsize=9.2, fontweight='bold', color='#d62728',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#d62728', alpha=0.95)
    )

    ax.annotate(
        f'Launch B (T = 1.0 yr)\nv_rel = {vB_rel} vs. Earth',
        xy=(0.15, 0.99), xytext=(1.15, 0.87),
        arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.4),
        fontsize=9.2, fontweight='bold', color='#2ca02c',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#2ca02c', alpha=0.95)
    )

    # Annotate final proper time differences at T = 2.0 yr
    ax.text(
        xB_au[-1], 2.04,
        f"Δτ_B - Δτ_A = {dtauB[-1]:+.1f} ms",
        color='#2ca02c', fontsize=10, fontweight='bold', ha='center'
    )
    ax.text(
        xC_au[-1], 2.04,
        f"Δτ_C - Δτ_A = {dtauC[-1]:+.1f} ms",
        color='#d62728', fontsize=10, fontweight='bold',
        ha=('center' if row == 0 else 'right')
    )

    ax.set_xlim(-1.8, 30.8)
    ax.set_ylim(0.0, 2.2)
    ax.set_ylabel('Heliocentric Coordinate Time T (years)', fontsize=10.5)
    ax.set_title(title, fontsize=11.5, fontweight='bold', pad=10)
    ax.grid(True, linestyle='--', alpha=0.45)
    ax.legend(loc='lower right', fontsize=9.2, framealpha=0.95)

axes[1].set_xlabel('Heliocentric x-Position (AU)  [1 AU/yr ≈ 4.74 km/s]', fontsize=11)

fig.suptitle(
    "Triplet Experiment: 2D Spacetime Worldlines (x, T) in the Heliocentric Frame (Rescaled Time Axis)",
    fontsize=14, fontweight='bold', y=0.985
)
fig.tight_layout(rect=[0, 0, 1, 0.955])
plt.savefig('triplet_worldlines.png', dpi=300)
plt.show()