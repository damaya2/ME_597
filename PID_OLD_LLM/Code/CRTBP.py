import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

# Standard Earth-Moon System Constants
MU = 0.012150585609262
DU = 384400.0          # Distance Unit (km) - Earth-Moon average distance
TU = 375700.0          # Time Unit (seconds)

# =============================================================================
# Helper Functions & Equations of Motion
# =============================================================================

def compute_jacobi_constant(state, mu=MU):
    """Computes the Jacobi Constant (C) for a given state [x, y, z, vx, vy, vz]."""
    x, y, z, vx, vy, vz = state[:6]
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1.0 + mu)**2 + y**2 + z**2)
    
    # Effective potential U(x, y, z)
    U = 0.5 * (x**2 + y**2) + (1.0 - mu) / r1 + mu / r2
    v_sq = vx**2 + vy**2 + vz**2
    return 2.0 * U - v_sq


def compute_lagrange_points(mu=MU):
    """
    Computes the dimensionless coordinates of the 5 Lagrange points (L1-L5).
    Returns a dictionary of 3D position vectors.
    """
    # L1, L2, L3 along the x-axis
    def dUdx(x):
        r1 = np.abs(x + mu)
        r2 = np.abs(x - 1.0 + mu)
        return x - (1.0 - mu) * (x + mu) / r1**3 - mu * (x - 1.0 + mu) / r2**3

    l1_x = root_scalar(dUdx, bracket=[-mu + 1e-4, 1.0 - mu - 1e-4]).root
    l2_x = root_scalar(dUdx, bracket=[1.0 - mu + 1e-4, 2.0]).root
    l3_x = root_scalar(dUdx, bracket=[-2.0, -mu - 1e-4]).root

    return {
        "L1": np.array([l1_x, 0.0, 0.0]),
        "L2": np.array([l2_x, 0.0, 0.0]),
        "L3": np.array([l3_x, 0.0, 0.0]),
        "L4": np.array([0.5 - mu, np.sqrt(3) / 2.0, 0.0]),
        "L5": np.array([0.5 - mu, -np.sqrt(3) / 2.0, 0.0]),
    }


def second_partials(x, y, z, mu=MU):
    """Computes second partial derivatives of the CRTBP effective potential."""
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1.0 + mu)**2 + y**2 + z**2)
    
    r1_3, r1_5 = r1**3, r1**5
    r2_3, r2_5 = r2**3, r2**5

    Uxx = 1.0 - (1.0 - mu)*(1.0/r1_3 - 3.0*(x + mu)**2/r1_5) - mu*(1.0/r2_3 - 3.0*(x - 1.0 + mu)**2/r2_5)
    Uyy = 1.0 - (1.0 - mu)*(1.0/r1_3 - 3.0*y**2/r1_5) - mu*(1.0/r2_3 - 3.0*y**2/r2_5)
    Uzz =     - (1.0 - mu)*(1.0/r1_3 - 3.0*z**2/r1_5) - mu*(1.0/r2_3 - 3.0*z**2/r2_5)
    
    Uxy = 3.0 * y * ((1.0 - mu)*(x + mu)/r1_5 + mu*(x - 1.0 + mu)/r2_5)
    Uxz = 3.0 * z * ((1.0 - mu)*(x + mu)/r1_5 + mu*(x - 1.0 + mu)/r2_5)
    Uyz = 3.0 * y * z * ((1.0 - mu)/r1_5 + mu/r2_5)

    return Uxx, Uxy, Uxz, Uyy, Uyz, Uzz


def crtbp_eom(t, y, mu=MU):
    """Rotating-frame CRTBP EOM."""
    x, yy, z, vx, vy, vz = y

    r1 = np.sqrt((x + mu)**2 + yy**2 + z**2)
    r2 = np.sqrt((x - 1.0 + mu)**2 + yy**2 + z**2)

    Ux = x  - (1.0 - mu)*(x + mu)/r1**3 - mu*(x - 1.0 + mu)/r2**3
    Uy = yy - (1.0 - mu)*yy/r1**3      - mu*yy/r2**3
    Uz =     - (1.0 - mu)*z/r1**3       - mu*z/r2**3

    ax =  2.0 * vy + Ux
    ay = -2.0 * vx + Uy
    az =  Uz

    return [vx, vy, vz, ax, ay, az]


def crtbp_var_eom(t, Y, mu=MU):
    """Variational equations for state propagation and State Transition Matrix (STM)."""
    x, y, z, vx, vy, vz = Y[0:6]

    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1.0 + mu)**2 + y**2 + z**2)

    Ux = x - (1.0 - mu)*(x + mu)/r1**3 - mu*(x - 1.0 + mu)/r2**3
    Uy = y - (1.0 - mu)*y/r1**3       - mu*y/r2**3
    Uz =   - (1.0 - mu)*z/r1**3       - mu*z/r2**3

    dxdt, dydt, dzdt = vx, vy, vz
    dvxdt =  2.0 * vy + Ux
    dvydt = -2.0 * vx + Uy
    dvzdt =  Uz

    # Jacobian Matrix A
    Uxx, Uxy, Uxz, Uyy, Uyz, Uzz = second_partials(x, y, z, mu)
    A = np.array([
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [Uxx, Uxy, Uxz, 0.0, 2.0, 0.0],
        [Uxy, Uyy, Uyz,-2.0, 0.0, 0.0],
        [Uxz, Uyz, Uzz, 0.0, 0.0, 0.0]
    ], dtype=np.float64)

    Phi = Y[6:].reshape((6, 6))
    dPhi = A @ Phi

    dY = np.zeros(42, dtype=np.float64)
    dY[0:6] = [dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt]
    dY[6:] = dPhi.flatten()

    return dY


# =============================================================================
# Compute Halo Manifolds
# =============================================================================

def compute_halo_manifolds(mu=MU, X0=None, P=None, Nnodes=40, eps0=1e-6, T_fwd_factor=4.0, T_bwd_factor=4.0, rtol=1e-12, atol=1e-12):
    """Computes stable and unstable manifolds for a CRTBP periodic orbit."""
    if X0 is None:
        X0 = np.array([1.118824382902157, 0.0, 0.014654873101278, 0.0, 0.180568501159703, 0.0])
    if P is None:
        P = 2.0 * 1.706067405636607

    T_fwd = T_fwd_factor * P
    T_bwd = T_bwd_factor * P

    # Step 1: Propagate state and STM over one full period
    Phi0 = np.eye(6)
    Y0_var = np.hstack((X0, Phi0.flatten()))
    t_nodes = np.linspace(0, P, Nnodes + 1)

    sol_period = solve_ivp(
        lambda t, y: crtbp_var_eom(t, y, mu),
        [0, P], Y0_var, t_eval=t_nodes,
        method='DOP853', rtol=rtol, atol=atol
    )

    Y_out = sol_period.y.T  # Shape: (Nnodes+1, 42)
    X_orb_full = Y_out[:, 0:6]

    # Step 2: Monodromy Matrix M = Phi(T, 0)
    M = Y_out[-1, 6:].reshape((6, 6))

    # Step 3: Compute Eigenvalues & Eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(M)

    # Unstable direction (|lambda| max)
    idx_u = np.argmax(np.abs(eigenvalues))
    v_u = np.real(eigenvectors[:, idx_u])
    v_u = v_u / np.linalg.norm(v_u)

    # Stable direction (|lambda| min / reciprocal)
    lambda_s = 1.0 / eigenvalues[idx_u]
    idx_s = np.argmin(np.abs(eigenvalues - lambda_s))
    v_s = np.real(eigenvectors[:, idx_s])
    v_s = v_s / np.linalg.norm(v_s)

    # Step 4 & 5: Map eigenvectors along orbit nodes via local STM
    eta_u = []
    eta_s = []
    X_nodes = np.zeros((Nnodes, 6))

    for i in range(Nnodes):
        Xi = Y_out[i, 0:6]
        Phi_i = Y_out[i, 6:].reshape((6, 6))
        X_nodes[i, :] = Xi

        vu_i = Phi_i @ v_u
        vs_i = Phi_i @ v_s

        eta_u.append(vu_i / np.linalg.norm(vu_i))
        eta_s.append(vs_i / np.linalg.norm(vs_i))

    # Step 6: Create local initial perturbations
    unstable_seeds = np.zeros((Nnodes, 2, 6))
    stable_seeds   = np.zeros((Nnodes, 2, 6))

    for i in range(Nnodes):
        Xi = X_nodes[i, :]
        unstable_seeds[i, 0, :] = Xi + eps0 * eta_u[i]
        unstable_seeds[i, 1, :] = Xi - eps0 * eta_u[i]
        stable_seeds[i, 0, :]   = Xi + eps0 * eta_s[i]
        stable_seeds[i, 1, :]   = Xi - eps0 * eta_s[i]

    # Step 7: Propagate Manifold Trajectories
    UM_traj = [[None, None] for _ in range(Nnodes)]
    SM_traj = [[None, None] for _ in range(Nnodes)]

    for i in range(Nnodes):
        for sgn in range(2):
            # Unstable manifold (forward in time)
            sol_u = solve_ivp(
                lambda t, x: crtbp_eom(t, x, mu),
                [0, T_fwd], unstable_seeds[i, sgn, :],
                method='DOP853', rtol=rtol, atol=atol
            )
            UM_traj[i][sgn] = {'t': sol_u.t, 'x': sol_u.y.T}

            # Stable manifold (backward in time)
            sol_s = solve_ivp(
                lambda t, x: crtbp_eom(t, x, mu),
                [0, -T_bwd], stable_seeds[i, sgn, :],
                method='DOP853', rtol=rtol, atol=atol
            )
            SM_traj[i][sgn] = {'t': sol_s.t, 'x': sol_s.y.T}

    return {
        'mu': mu,
        'P': P,
        'X_orb_full': X_orb_full,
        'M': M,
        'UM_traj': UM_traj,
        'SM_traj': SM_traj,
        'Nnodes': Nnodes
    }