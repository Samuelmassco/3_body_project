import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# =============================================================================
# Precise computation of Lagrange points via Newton-Raphson
# =============================================================================

def find_L1(mu, tol=1e-14, max_iter=200):
    """x position of L1 (between the two bodies) via Newton-Raphson."""
    x = 1.0 - mu - (mu / 3.0) ** (1.0 / 3.0)
    for _ in range(max_iter):
        r1 = abs(x + mu)
        r2 = abs(x - 1.0 + mu)
        f  = x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
        df = 1 - (1-mu)*(1/r1**3 - 3*(x+mu)**2/r1**5) \
               - mu*(1/r2**3 - 3*(x-1+mu)**2/r2**5)
        dx = f / df
        x -= dx
        if abs(dx) < tol:
            break
    return x

def find_L2(mu, tol=1e-14, max_iter=200):
    """x position of L2 (beyond the secondary body) via Newton-Raphson."""
    x = 1.0 - mu + (mu / 3.0) ** (1.0 / 3.0)
    for _ in range(max_iter):
        r1 = abs(x + mu)
        r2 = abs(x - 1.0 + mu)
        f  = x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
        df = 1 - (1-mu)*(1/r1**3 - 3*(x+mu)**2/r1**5) \
               - mu*(1/r2**3 - 3*(x-1+mu)**2/r2**5)
        dx = f / df
        x -= dx
        if abs(dx) < tol:
            break
    return x


def run_all():
    """Runs simulations for L1, L2, L4, L5 (equilibrium + perturbed)."""

    G = 1.0
    m1, m2 = 1.0, 0.012
    masses = [m1, m2, 1e-10]
    R = 1.0

    mu = m2 / (m1 + m2)
    omega = np.sqrt(G * (m1 + m2) / R**3)

    # Positions in the center of mass frame
    x1 = -mu * R          # Earth
    x2 = (1 - mu) * R     # Moon

    # Precise positions of Lagrange points
    x_L1 = find_L1(mu)
    x_L2 = find_L2(mu)
    # L4 and L5: vertices of the equilateral triangle (distance = R from both bodies)
    # In the CM frame: x = 0.5 - mu,  y = ± sqrt(3)/2
    x_L4 = 0.5 - mu
    y_L4 = np.sqrt(3.0) / 2.0
    x_L5 = 0.5 - mu
    y_L5 = -np.sqrt(3.0) / 2.0

    print("=" * 60)
    print("  EARTH-MOON SYSTEM – LAGRANGE POINTS")
    print("=" * 60)
    print(f"  mu    = {mu:.6f}")
    print(f"  omega = {omega:.6f}")
    print(f"  Earth = ({x1:.6f}, 0)")
    print(f"  Moon  = ({x2:.6f}, 0)")
    print(f"  L1    = ({x_L1:.8f}, 0)")
    print(f"  L2    = ({x_L2:.8f}, 0)")
    print(f"  L4    = ({x_L4:.6f}, {y_L4:.6f})")
    print(f"  L5    = ({x_L5:.6f}, {y_L5:.6f})")
    print("=" * 60)

    # ==================================================================
    #  L1: unstable point between Earth and Moon
    # ==================================================================
    L1_pos = [[x1, 0.0], [x2, 0.0], [x_L1, 0.0]]
    compute('L1_equilibrium.mp4', L1_pos, masses)

    L1p_pos = [[x1, 0.0], [x2, 0.0], [x_L1, 0.001]]
    compute('L1_perturbed.mp4', L1p_pos, masses)

    # ==================================================================
    #  L2: unstable point beyond the Moon
    # ==================================================================
    L2_pos = [[x1, 0.0], [x2, 0.0], [x_L2, 0.0]]
    compute('L2_equilibrium.mp4', L2_pos, masses)

    L2p_pos = [[x1, 0.0], [x2, 0.0], [x_L2 + 0.0005, 0.0]]
    compute('L2_perturbed.mp4', L2p_pos, masses)

def runL45():
    
    G = 1.0
    m1, m2 = 1.0, 0.012
    masses = [m1, m2, 1e-10]
    R = 1.0

    mu = m2 / (m1 + m2)
    omega = np.sqrt(G * (m1 + m2) / R**3)

    # Positions in the center of mass frame
    x1 = -mu * R          # Earth
    x2 = (1 - mu) * R     # Moon

    # Precise positions of Lagrange points
    x_L1 = find_L1(mu)
    x_L2 = find_L2(mu)
    # L4 and L5: vertices of the equilateral triangle (distance = R from both bodies)
    # In the CM frame: x = 0.5 - mu,  y = ± sqrt(3)/2
    x_L4 = 0.5 - mu
    y_L4 = np.sqrt(3.0) / 2.0
    x_L5 = 0.5 - mu
    y_L5 = -np.sqrt(3.0) / 2.0
    # ==================================================================
    #  L4: stable point (equilateral triangle, y > 0)
    # ==================================================================
    L4_pos = [[x1, 0.0], [x2, 0.0], [x_L4, y_L4]]
    compute('L4_equilibrium.mp4', L4_pos, masses)

    L4p_pos = [[x1, 0.0], [x2, 0.0], [x_L4, y_L4 + 0.02]]
    compute('L4_perturbed.mp4', L4p_pos, masses)

    # ==================================================================
    #  L5: stable point (equilateral triangle, y < 0)
    # ==================================================================
    L5_pos = [[x1, 0.0], [x2, 0.0], [x_L5, y_L5]]
    compute('L5_equilibrium.mp4', L5_pos, masses)

    L5p_pos = [[x1, 0.0], [x2, 0.0], [x_L5, y_L5 - 0.02]]
    compute('L5_perturbed.mp4', L5p_pos, masses)

    # ==================================================================
    #  L5_bis: mu > mu_Routh → L5 becomes UNSTABLE
    #  Routh criterion: mu_crit = (1 - sqrt(23/27)) / 2 ≈ 0.03852
    #  We choose mu_bis = 0.045 (comparable masses → L4/L5 unstable)
    # ==================================================================
    mu_bis = 0.045
    m1_bis = 1.0 - mu_bis
    m2_bis = mu_bis
    masses_bis = [m1_bis, m2_bis, 1e-10]

    x1_bis = -mu_bis * R
    x2_bis = (1 - mu_bis) * R
    x_L5_bis = 0.5 - mu_bis
    y_L5_bis = -np.sqrt(3.0) / 2.0

    print("\n" + "=" * 60)
    print("  L5_bis: mu = {:.4f} > mu_Routh ≈ 0.0385 → UNSTABLE".format(mu_bis))
    print(f"  Earth = ({x1_bis:.6f}, 0),  Moon = ({x2_bis:.6f}, 0)")
    print(f"  L5    = ({x_L5_bis:.6f}, {y_L5_bis:.6f})")
    print("=" * 60)

    L5bis_pos = [[x1_bis, 0.0], [x2_bis, 0.0], [x_L5_bis, y_L5_bis]]
    compute('L5_bis_equilibrium.mp4', L5bis_pos, masses_bis)

    L5bis_p_pos = [[x1_bis, 0.0], [x2_bis, 0.0], [x_L5_bis, y_L5_bis - 0.02]]
    compute('L5_bis_perturbed.mp4', L5bis_p_pos, masses_bis)


# ==========================================================
# N-BODY SIMULATION (inertial frame, video output)
#
# compute(output_name, init_pos, masses)
#   - automatically computes circular velocities
#   - integrates with RK4
#   - saves as .mp4 (ffmpeg) or .gif (pillow)
# ==========================================================


def _nbody_accel(positions, masses, G):
    """Gravitational accelerations for N bodies (numpy)."""
    N = len(masses)
    acc = [np.zeros(2) for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j:
                r_vec = positions[j] - positions[i]
                dist = np.linalg.norm(r_vec)
                if dist > 1e-8:
                    acc[i] += G * masses[j] * r_vec / dist**3
    return acc


def _rk4_step(positions, velocities, masses, G, dt):
    """One RK4 step for the N-body system."""
    N = len(masses)

    def derivs(pos_list, vel_list):
        acc = _nbody_accel(pos_list, masses, G)
        return vel_list, acc          # dx/dt = v,  dv/dt = a

    # k1
    dx1, dv1 = derivs(positions, velocities)

    # k2
    p2 = [positions[i] + 0.5 * dt * dx1[i] for i in range(N)]
    v2 = [velocities[i] + 0.5 * dt * dv1[i] for i in range(N)]
    dx2, dv2 = derivs(p2, v2)

    # k3
    p3 = [positions[i] + 0.5 * dt * dx2[i] for i in range(N)]
    v3 = [velocities[i] + 0.5 * dt * dv2[i] for i in range(N)]
    dx3, dv3 = derivs(p3, v3)

    # k4
    p4 = [positions[i] + dt * dx3[i] for i in range(N)]
    v4 = [velocities[i] + dt * dv3[i] for i in range(N)]
    dx4, dv4 = derivs(p4, v4)

    new_pos = [positions[i]  + (dt / 6) * (dx1[i] + 2*dx2[i] + 2*dx3[i] + dx4[i])
               for i in range(N)]
    new_vel = [velocities[i] + (dt / 6) * (dv1[i] + 2*dv2[i] + 2*dv3[i] + dv4[i])
               for i in range(N)]
    return new_pos, new_vel


def compute(output_name, init_pos, masses, dt=0.001, T_max=10.0):
    """N-body simulation.

    Parameters
    ----------
    output_name : str   – output file name (.mp4 or .gif)
    init_pos    : list  – [[x,y], ...] initial positions (3 bodies)
    masses      : list  – [m1, m2, m3]
    dt, T_max   : float – time step and duration
    """
    G_val = 1.0
    N = len(masses)
    collision_limit = 0.01

    # --- Center of mass → origin ---
    total_mass = sum(masses)
    cm = np.zeros(2)
    for i in range(N):
        cm += masses[i] * np.array(init_pos[i])
    cm /= total_mass

    pos = [np.array(init_pos[i], dtype=float) - cm for i in range(N)]

    # --- Automatic circular velocities ---
    # common omega = sqrt(G * M_total / a^3)  with a = |r1 − r2|
    r12 = np.linalg.norm(pos[1] - pos[0])
    omega = np.sqrt(G_val * total_mass / r12**3)

    vel = []
    for i in range(N):
        x, y = pos[i]
        vel.append(np.array([-omega * y, omega * x]))   # v = omega × r

    print(f"\n=== {output_name} ===")
    print(f"  CM recentered, omega = {omega:.6f}")
    for i in range(N):
        print(f"  Body {i}: pos=({pos[i][0]:+.6f}, {pos[i][1]:+.6f})  "
              f"vel=({vel[i][0]:+.6f}, {vel[i][1]:+.6f})")

    # --- Initial energy (diagnostic) ---
    def total_energy(p, v):
        E = 0.0
        for i in range(N):
            E += 0.5 * masses[i] * np.dot(v[i], v[i])
        for i in range(N):
            for j in range(i+1, N):
                r = np.linalg.norm(p[j] - p[i])
                if r > 1e-10:
                    E -= G_val * masses[i] * masses[j] / r
        return E

    E0 = total_energy(pos, vel)

    # --- RK4 integration loop ---
    num_steps = int(T_max / dt)
    history = [[p.copy() for p in pos]]   # stores at each step
    times = [0.0]
    collision_time = None

    for step in range(1, num_steps + 1):
        pos, vel = _rk4_step(pos, vel, masses, G_val, dt)

        # Collision detection
        broke = False
        for i in range(N):
            for j in range(i+1, N):
                if np.linalg.norm(pos[j] - pos[i]) < collision_limit:
                    collision_time = step * dt
                    print(f"  !!! COLLISION bodies {i}-{j} at t = {collision_time:.4f}")
                    broke = True
                    break
            if broke:
                break

        history.append([p.copy() for p in pos])
        times.append(step * dt)
        if broke:
            break

    Ef = total_energy(pos, vel)
    print(f"  E initial = {E0:.8e},  E final = {Ef:.8e},  "
          f"dE/E = {abs(Ef - E0) / (abs(E0) + 1e-30):.2e}")

    # --- Conversion for animation ---
    n_frames = len(times)
    hist_arr = [np.array([history[t][i] for t in range(n_frames)]) for i in range(N)]

    # --- Compute trajectories in the rotating frame ---
    # Rotate by -omega*t at each time step to switch to the co-rotating frame
    hist_rot = [np.zeros_like(h) for h in hist_arr]
    for k in range(n_frames):
        theta = -omega * times[k]
        c, s = np.cos(theta), np.sin(theta)
        for i in range(N):
            x, y = hist_arr[i][k]
            hist_rot[i][k, 0] = c * x - s * y
            hist_rot[i][k, 1] = s * x + c * y

    # ==========================================================
    #           ANIMATION (inertial + rotating side by side)
    # ==========================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # --- Adaptive limits ---
    margin = 0.3
    all_x = np.concatenate([h[:, 0] for h in hist_arr])
    all_y = np.concatenate([h[:, 1] for h in hist_arr])
    lim = max(np.max(np.abs(all_x)), np.max(np.abs(all_y))) + margin

    for ax, title in [(ax1, "Inertial frame"), (ax2, "Rotating frame")]:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_title(title, fontsize=12, weight='bold')
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    colors = ['blue', 'orange', 'green']
    labels_bodies = ['Earth (m1)', 'Moon (m2)', 'Satellite (m3)']
    sizes = [10, 7, 4]

    # Graphical elements for both panels
    trails_in, dots_in = [], []
    trails_rot, dots_rot = [], []
    for i in range(N):
        ln1, = ax1.plot([], [], '-', color=colors[i], linewidth=0.8, alpha=0.6)
        trails_in.append(ln1)
        d1, = ax1.plot([], [], 'o', color=colors[i], markersize=sizes[i],
                       label=labels_bodies[i], zorder=5)
        dots_in.append(d1)

        ln2, = ax2.plot([], [], '-', color=colors[i], linewidth=0.8, alpha=0.6)
        trails_rot.append(ln2)
        d2, = ax2.plot([], [], 'o', color=colors[i], markersize=sizes[i],
                       label=labels_bodies[i], zorder=5)
        dots_rot.append(d2)

    # L1 marker in the rotating frame (initial satellite position)
    ax2.plot(hist_rot[2][0, 0], hist_rot[2][0, 1], 'k+', markersize=14,
             markeredgewidth=2, label='Initial position', zorder=4)

    ax1.legend(loc='upper right', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    time_text = ax1.text(0.05, 0.95, '', transform=ax1.transAxes, weight='bold')

    plt.tight_layout()

    # Only draw every `skip`-th frame to keep the video lightweight
    skip = max(1, n_frames // 800)

    all_artists = trails_in + dots_in + trails_rot + dots_rot + [time_text]

    def update(frame):
        idx = frame * skip
        if idx >= n_frames:
            idx = n_frames - 1
        tail = max(0, idx - 400)

        for i in range(N):
            # Panneau inertiel
            trails_in[i].set_data(hist_arr[i][tail:idx, 0],
                                  hist_arr[i][tail:idx, 1])
            dots_in[i].set_data([hist_arr[i][idx, 0]], [hist_arr[i][idx, 1]])
            # Panneau tournant
            trails_rot[i].set_data(hist_rot[i][tail:idx, 0],
                                   hist_rot[i][tail:idx, 1])
            dots_rot[i].set_data([hist_rot[i][idx, 0]], [hist_rot[i][idx, 1]])

        t_now = times[idx]
        if collision_time and idx == n_frames - 1:
            time_text.set_text(f"COLLISION t = {t_now:.2f}")
            time_text.set_color('red')
        else:
            time_text.set_text(f"t = {t_now:.2f}")
        return all_artists

    total_anim_frames = n_frames // skip
    ani = FuncAnimation(fig, update, frames=total_anim_frames,
                        interval=20, blit=True)

    # Save: try ffmpeg (.mp4), otherwise fall back to pillow (.gif)
    print(f"  Saving {output_name} ...")
    if output_name.endswith('.mp4'):
        try:
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=30)
            ani.save(output_name, writer=writer, dpi=120)
        except Exception as e:
            fallback = output_name.replace('.mp4', '.gif')
            print(f"  ffmpeg unavailable ({e}), fallback → {fallback}")
            ani.save(fallback, writer='pillow', fps=30)
    else:
        ani.save(output_name, writer='pillow', fps=30)

    plt.close(fig)
    print(f"  → done.\n")






def main():
    run_all()

if __name__ == "__main__":
    main()
