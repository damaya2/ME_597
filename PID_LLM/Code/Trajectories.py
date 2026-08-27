import sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from CRTBP import crtbp_eom, second_partials, crtbp_var_eom, compute_lagrange_points, MU, DU, TU
from Controller import CRTBPController, controlled_crtbp_eom

# =============================================================================
# System Constants & Mission Descriptions
# =============================================================================
LU_TO_KM = DU                                 # Length unit to kilometers (384,400 km)
TU_TO_SEC = TU                                # Time unit to seconds (~375,700 s)
VU_TO_MS = (LU_TO_KM * 1000.0) / TU_TO_SEC    # Velocity conversion (~1023.15 m/s)
AU_TO_MS2 = (LU_TO_KM * 1000.0) / (TU_TO_SEC ** 2) # Acceleration conversion (~0.002723 m/s²)

EARTH_POS = np.array([-MU, 0.0, 0.0])
MOON_POS = np.array([1.0 - MU, 0.0, 0.0])

# Canonical Radii for continuous safety events
R_EARTH_CANONICAL = 6371.0 / LU_TO_KM  # ~0.01657 DU
R_MOON_CANONICAL = 1737.4 / LU_TO_KM   # ~0.00452 DU
MAX_ESCAPE_RADIUS = 3.0                # 3.0 DU (~1.15 million km)

MISSION_DESCRIPTIONS = {
    "earth_orbit": "Maintain an orbit trajectory in the Earth region.",
    "moon_orbit": "Maintain a cislunar trajectory near the Moon.",
    "transfer": "Execute an Earth-to-Moon transfer trajectory.",
    "halo_l1": "L1 Northern/Southern Halo Orbit Stationkeeping.",
    "halo_l2": "L2 Near-Rectilinear/Halo Orbit Stationkeeping.",
    "dro": "Lunar Distant Retrograde Orbit (DRO) Stationkeeping.",
}


# =============================================================================
# Physical Verification & Perturbations
# =============================================================================

def unmodeled_perturbations(t, state):
    """
    Simulates continuous real-time unmodeled disturbance forces acting on spacecraft:
    1. Solar Radiation Pressure (SRP ~ 1e-6 m/s^2) in canonical units
    2. Deterministic multi-frequency thruster noise (smooth for ODE solvers)
    """
    r_vec = np.asarray(state[:3], dtype=np.float64)
    r_norm = np.linalg.norm(r_vec)

    srp_dir = r_vec / (r_norm + 1e-12)
    
    # Realistic SRP acceleration: ~ 1.0e-6 m/s^2 converted to non-dimensional AU
    a_srp_ms2 = 1.0e-6
    a_srp_canon = (a_srp_ms2 / AU_TO_MS2) * srp_dir
    
    # Deterministic multi-frequency harmonic disturbance
    noise_amplitude_ms2 = 1.0e-7 / AU_TO_MS2
    w1, w2, w3 = 12.3, 45.6, 78.9
    a_noise_canon = noise_amplitude_ms2 * np.array([
        np.sin(w1 * t),
        np.cos(w2 * t),
        np.sin(w3 * t + 0.5)
    ])

    return a_srp_canon + a_noise_canon


def verify_physical_bounds(state, pos_err, vel_err, control_vec, u_max_m_s2=0.01):
    """Validates physical safety boundaries and active thruster acceleration limits."""
    pos_err_km = np.linalg.norm(pos_err) * LU_TO_KM
    vel_err_ms = np.linalg.norm(vel_err) * VU_TO_MS
    u_mag_ms2 = np.linalg.norm(control_vec) * AU_TO_MS2

    # Allow maximum tracking error limits
    if pos_err_km > 25000.0 or vel_err_ms > 2500.0:
        return False

    # Adaptive thruster limit check
    if u_mag_ms2 > (u_max_m_s2 * 1.05):  # 5% numerical threshold buffer
        return False

    return True


def verify_and_get_dynamic_metadata(state):
    """Determines mission regime and description based on position."""
    r_earth_km = np.linalg.norm(state[:3] - EARTH_POS) * LU_TO_KM
    r_moon_km = np.linalg.norm(state[:3] - MOON_POS) * LU_TO_KM

    if r_moon_km < 60000.0:
        mission_type = "moon_orbit"
    elif r_earth_km < 120000.0:
        mission_type = "earth_orbit"
    else:
        mission_type = "transfer"

    return mission_type, get_mission_description(mission_type)


def get_mission_description(mission_type):
    """Returns official description text for a mission type."""
    return MISSION_DESCRIPTIONS.get(mission_type, "Unknown mission profile.")


# =============================================================================
# Continuous Safety Events for solve_ivp
# =============================================================================

def earth_impact_event(t, x, *args):
    """Triggers zero-crossing when distance to Earth equals Earth radius."""
    mu_val = args[0] if args else MU
    r1 = np.sqrt((x[0] + mu_val) ** 2 + x[1] ** 2 + x[2] ** 2)
    return r1 - R_EARTH_CANONICAL

earth_impact_event.terminal = True
earth_impact_event.direction = -1


def moon_impact_event(t, x, *args):
    """Triggers zero-crossing when distance to Moon equals Moon radius."""
    mu_val = args[0] if args else MU
    r2 = np.sqrt((x[0] - 1.0 + mu_val) ** 2 + x[1] ** 2 + x[2] ** 2)
    return r2 - R_MOON_CANONICAL

moon_impact_event.terminal = True
moon_impact_event.direction = -1


def escape_event(t, x, *args):
    """Triggers zero-crossing when radial distance exceeds MAX_ESCAPE_RADIUS."""
    mu_val = args[0] if args else MU
    r1 = np.sqrt((x[0] + mu_val) ** 2 + x[1] ** 2 + x[2] ** 2)
    return MAX_ESCAPE_RADIUS - r1

escape_event.terminal = True
escape_event.direction = -1

SAFETY_EVENTS = [earth_impact_event, moon_impact_event, escape_event]


# =============================================================================
# Multiple Shooting Differential Corrector
# =============================================================================

def multiple_shooting_corrector(X_guess, N_nodes, T_total, mu=MU, max_iter=20, tol=1e-10):
    """
    Parallel Multiple Shooting Algorithm using full state variational equations.
    Enforces continuous trajectory nodes over fixed sub-intervals DT = T_total / N_nodes.
    """
    dt = T_total / N_nodes
    X_nodes = np.copy(X_guess)  # Shape: (N_nodes + 1, 6)
    
    for iteration in range(max_iter):
        F = np.zeros(N_nodes * 6)            # Continuity constraint vector
        DF = np.zeros((N_nodes * 6, (N_nodes + 1) * 6))  # Sparse Jacobian matrix
        
        for i in range(N_nodes):
            X_i = X_nodes[i]
            Phi0 = np.eye(6)
            Y0_var = np.hstack((X_i, Phi0.flatten()))
            
            # Propagate node state and State Transition Matrix (STM)
            sol = solve_ivp(
                lambda t, y: crtbp_var_eom(t, y, mu),
                [0, dt], Y0_var,
                method='DOP853', rtol=1e-11, atol=1e-12
            )
            
            X_next_integrated = sol.y[0:6, -1]
            Phi_i = sol.y[6:, -1].reshape((6, 6))
            
            # Continuity defect: F_i = X_integrated(t_i + dt) - X_node(i+1)
            F[i*6 : (i+1)*6] = X_next_integrated - X_nodes[i+1]
            
            # Sub-Jacobian blocks
            DF[i*6 : (i+1)*6, i*6 : (i+1)*6] = Phi_i
            DF[i*6 : (i+1)*6, (i+1)*6 : (i+2)*6] = -np.eye(6)
            
        norm_F = np.linalg.norm(F)
        if norm_F < tol:
            print(f"[Multiple Shooting] Converged in {iteration} iterations with residual {norm_F:.2e}")
            break
            
        # Minimum-norm Newton correction step
        dX = np.linalg.pinv(DF) @ (-F)
        X_nodes += dX.reshape((N_nodes + 1, 6))
        
    return X_nodes, dt


# =============================================================================
# Advanced Target Orbit Generators (Halos, DROs, Transfers)
# =============================================================================

def generate_halo_orbit(libration_point="L1", Ax_km=15000.0, mu=MU, Nnodes=20):
    """Generates an automated, numerically corrected Halo orbit around L1 or L2."""
    l_points = compute_lagrange_points(mu)
    l_pos = l_points[libration_point]
    
    # Non-dimensionalize amplitude
    Ax = Ax_km / LU_TO_KM
    P_approx = 2.0 * np.pi * 0.5  # Approximate halo period
    
    # Analytical seed initial guess
    if libration_point == "L1":
        x0_guess = np.array([l_pos[0] - Ax, 0.0, Ax * 0.8, 0.0, 0.18, 0.0])
    else:
        x0_guess = np.array([l_pos[0] + Ax, 0.0, Ax * 0.8, 0.0, -0.18, 0.0])
        
    t_nodes = np.linspace(0, P_approx, Nnodes + 1)
    X_nodes_guess = np.zeros((Nnodes + 1, 6))
    
    # Uncorrected linear node seed
    for i in range(Nnodes + 1):
        sol_seed = solve_ivp(
            lambda t, y: crtbp_eom(t, y, mu),
            [0, t_nodes[i]], x0_guess, method='RK45'
        )
        X_nodes_guess[i, :] = sol_seed.y[:, -1]
        
    X_nodes_corrected, dt = multiple_shooting_corrector(X_nodes_guess, Nnodes, P_approx, mu=mu)
    return f"halo_{libration_point.lower()}", X_nodes_corrected, P_approx


def generate_dro_orbit(x_amplitude_km=40000.0, mu=MU, Nnodes=20):
    """Generates a stable Distant Retrograde Orbit (DRO) near the Moon."""
    x_moon = 1.0 - mu
    Ax = x_amplitude_km / LU_TO_KM
    x0_guess = np.array([x_moon + Ax, 0.0, 0.0, 0.0, -0.45, 0.0])
    P_approx = 2.0 * np.pi * 0.4
    
    t_nodes = np.linspace(0, P_approx, Nnodes + 1)
    X_nodes_guess = np.zeros((Nnodes + 1, 6))
    for i in range(Nnodes + 1):
        sol_seed = solve_ivp(lambda t, y: crtbp_eom(t, y, mu), [0, t_nodes[i]], x0_guess)
        X_nodes_guess[i, :] = sol_seed.y[:, -1]
        
    X_nodes_corrected, dt = multiple_shooting_corrector(X_nodes_guess, Nnodes, P_approx, mu=mu)
    return "dro", X_nodes_corrected, P_approx


def generate_reference_trajectory(mission_type=None, max_retries=15):
    """Generates an unforced reference solution using RK45 integration."""
    if mission_type is None:
        mission_type = np.random.choice(["earth_orbit", "moon_orbit", "transfer"])

    if mission_type == "earth_orbit":
        base_x0 = [-0.30, 0.00, 0.00, 0.00, -1.85, 0.05]
        t_span = (0.0, 4.0 * np.pi)
    elif mission_type == "moon_orbit":
        x_moon = 1.0 - MU
        base_x0 = [x_moon + 0.10, 0.00, 0.010, 0.00, 0.25, 0.02]
        t_span = (0.0, 4.0 * np.pi)
    elif mission_type == "transfer":
        r_heo_du = 80000.0 / LU_TO_KM
        base_x0 = [-MU + r_heo_du, 0.00, 0.005, 0.00, 1.36, 0.01]
        t_span = (0.0, 2.2 * np.pi)
    else:
        raise ValueError(f"Unknown mission type: {mission_type}")

    sol = None
    for attempt in range(max_retries):
        if attempt == 0:
            x0_ref = np.array(base_x0, dtype=np.float64)
        else:
            pos_pert = np.random.uniform(-1.0 / LU_TO_KM, 1.0 / LU_TO_KM, size=3)
            vel_pert = np.random.uniform(-0.0001 / VU_TO_MS, 0.0001 / VU_TO_MS, size=3)
            x0_ref = np.array(base_x0, dtype=np.float64) + np.hstack([pos_pert, vel_pert])

        sol = solve_ivp(
            crtbp_eom,
            t_span,
            x0_ref,
            args=(MU,),
            method="RK45",
            rtol=1e-9,
            atol=1e-11,
            max_step=0.005,
        )

        if sol.status == 0 and len(sol.t) > 5:
            break

    # Continuous CubicSpline interpolator over reference time domain
    spline = CubicSpline(sol.t, sol.y.T, axis=0)
    t_min, t_max = sol.t[0], sol.t[-1]

    def ref_fn(t):
        t_arr = np.asarray(t, dtype=np.float64)
        t_eval = np.clip(t_arr, t_min, t_max)
        return spline(t_eval)

    return mission_type, sol, ref_fn

# =============================================================================
# Controlled Trajectory Simulation
# =============================================================================

def simulate_controlled_trajectory(
    mu_val,
    initial_state,
    reference_orbit_fn,
    t_span,
    u_max_m_s2=0.01,
    Kp=None,
    Kd=None,
    Ki=None,
    enable_perturbations=True,
    rtol=1e-7,
    atol=1e-9,
    print_stride=100,
):
    """
    Simulates closed-loop controlled state propagation under continuous disturbance forces.
    """
    ref_state_0 = np.squeeze(reference_orbit_fn(t_span[0]))

    if initial_state is None or np.allclose(initial_state, 0):
        initial_state = ref_state_0.copy()

    controller = CRTBPController(
        mu=mu_val,
        reference_state=ref_state_0,
        Kp=Kp,
        Kd=Kd,
        Ki=Ki,
        u_max_m_s2=u_max_m_s2,
    )

    Y0_aug = np.hstack([initial_state, np.zeros(3)])
    p_fn = unmodeled_perturbations if enable_perturbations else None

    step_counter = 0

    def rhs_wrapper(t, Y, *args):
        nonlocal step_counter
        step_counter += 1

        dY = controlled_crtbp_eom(t, Y, mu_val, reference_orbit_fn, controller, perturbations_fn=p_fn)

        if step_counter % print_stride == 0:
            state_curr = Y[:6]
            ref_curr = np.squeeze(reference_orbit_fn(t))
            err_pos_km = np.linalg.norm(state_curr[:3] - ref_curr[:3]) * LU_TO_KM
            err_vel_ms = np.linalg.norm(state_curr[3:6] - ref_curr[3:6]) * VU_TO_MS

            print(
                f"[Solver Step {step_counter:06d}] t={t:6.3f} | "
                f"Pos Err: {err_pos_km:7.2f} km | "
                f"Vel Err: {err_vel_ms:7.2f} m/s",
                flush=True,
            )

        return dY

    sol = solve_ivp(
        rhs_wrapper,
        t_span,
        Y0_aug,
        args=(mu_val,),
        events=SAFETY_EVENTS,
        method="RK45",
        rtol=rtol,
        atol=atol,
        max_step=0.005,
    )

    u_history = np.zeros((sol.t.size, 3))
    safety_flags = np.zeros(sol.t.size, dtype=bool)

    for idx in range(sol.t.size):
        t_i = sol.t[idx]
        state_i = sol.y[:6, idx]
        xc_i = sol.y[6:9, idx]

        ref_i = np.squeeze(reference_orbit_fn(t_i))
        err_i = state_i - ref_i

        u_i = controller.compute_control_action(err_i, xc_i, current_ref_state=ref_i)
        u_history[idx, :] = u_i

        safety_flags[idx] = verify_physical_bounds(
            state=state_i,
            pos_err=err_i[:3],
            vel_err=err_i[3:],
            control_vec=u_i,
            u_max_m_s2=u_max_m_s2,
        )

    return sol, u_history, safety_flags