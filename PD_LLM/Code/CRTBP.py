import numpy as np

# =========================================================
# SYSTEM PARAMETERS & CONVERSION CONSTANTS (EARTH-MOON)
# =========================================================
# Earth–Moon mass parameter
mu = 0.012150585609262  # m2 / (m1 + m2)
prim = {"x1": -mu, "y1": 0.0, "x2": 1.0 - mu, "y2": 0.0}

# Earth-Moon Canonical Units -> Metric Physical Units
DU = 384400.0  # Length Unit: Distance between Earth and Moon barycenter (km)
TU = 375190.26  # Time Unit: Earth-Moon orbital period / 2pi (seconds)
VU = (DU / TU) * 1000.0  # Velocity Unit: ~1024.55 m/s
AU = VU / TU  # Acceleration Unit: ~0.00273 m/s²


# =========================================================
# UNIT CONVERSION HELPERS
# =========================================================
def canonical_to_physical(state, control=None):
    """Converts canonical CRTBP state [x, y, z, vx, vy, vz] (and optional control
    vector [ux, uy, uz]) to physical metric units (km, m/s, m/s²).
    """
    pos_km = state[0:3] * DU
    vel_ms = state[3:6] * VU

    if control is not None:
        acc_ms2 = control * AU
        return pos_km, vel_ms, acc_ms2

    return pos_km, vel_ms


def physical_to_canonical(pos_km, vel_ms, acc_ms2=None):
    """Converts physical metric units (km, m/s, m/s²) to canonical CRTBP state."""
    pos_can = np.array(pos_km, dtype=np.float64) / DU
    vel_can = np.array(vel_ms, dtype=np.float64) / VU

    if acc_ms2 is not None:
        acc_can = np.array(acc_ms2, dtype=np.float64) / AU
        return pos_can, vel_can, acc_can

    return pos_can, vel_can


# =========================================================
# EQUATIONS OF MOTION
# =========================================================
def crtbp_eom(t, x, mu):
    X, Y, Z = x[0], x[1], x[2]
    dX, dY, dZ = x[3], x[4], x[5]

    # -----------------------------------------------------
    # FIX 1: BOUNDED SPATIAL SANITIZATION
    # Prevents runaway divergence at large t (e.g., t = 6pi)
    # If state escapes beyond 5 LU (~1.9M km), softly damp position
    # -----------------------------------------------------
    pos_norm = np.hypot(X, np.hypot(Y, Z))
    if pos_norm > 5.0:
        X, Y, Z = (X / pos_norm) * 5.0, (Y / pos_norm) * 5.0, (Z / pos_norm) * 5.0

    # Distances to primaries with singularity protection (eps = 1e-4)
    eps = 1e-4
    r1 = max(np.sqrt((X + mu) ** 2 + Y**2 + Z**2), eps)
    r2 = max(np.sqrt((X - 1 + mu) ** 2 + Y**2 + Z**2), eps)

    # Effective potential derivatives
    Ux = X - (1 - mu) * (X + mu) / r1**3 - mu * (X - 1 + mu) / r2**3
    Uy = Y - (1 - mu) * Y / r1**3 - mu * Y / r2**3
    Uz = -((1 - mu) * Z / r1**3 + mu * Z / r2**3)

    ddX = Ux + 2 * dY
    ddY = Uy - 2 * dX
    ddZ = Uz

    # -----------------------------------------------------
    # FIX 2: ACCELERATION SATURATION CLAMPING
    # Prevents extreme forces near primary singularities
    # -----------------------------------------------------
    max_acc = 50.0  # Max canonical acceleration
    ddX = np.clip(ddX, -max_acc, max_acc)
    ddY = np.clip(ddY, -max_acc, max_acc)
    ddZ = np.clip(ddZ, -max_acc, max_acc)

    return [dX, dY, dZ, ddX, ddY, ddZ]


def crtbp_controlled_eom(t, state, reference_sol, control_law_func, mu):
    """Closed-loop CRTBP dynamics incorporating control thrust accelerations."""
    desired_state = reference_sol.sol(t)
    dsdt = np.array(crtbp_eom(t, state, mu), dtype=np.float64)
    u = np.array(control_law_func(state, desired_state), dtype=np.float64)

    # Apply thrust acceleration
    dsdt[3:6] += u

    return dsdt