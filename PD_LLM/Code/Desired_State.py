import numpy as np
from scipy.integrate import solve_ivp
from CRTBP import crtbp_eom, mu, DU

MISSION_DESCRIPTIONS = {
    "earth_orbit": "Maintain an orbit trajectory in the Earth region.",
    "moon_orbit": "Maintain a cislunar trajectory near the Moon.",
    "transfer": "Execute an Earth-to-Moon transfer trajectory.",
}

# Physical primary radii converted to canonical units via DU (384,400 km)
R_EARTH_CANONICAL = 6371.0 / DU  # ~0.01657 canonical units
R_MOON_CANONICAL = 1737.4 / DU  # ~0.00452 canonical units
MAX_ESCAPE_RADIUS = 3.0  # 3.0 DU (~1.15 million km)


# =========================================================
# CONTINUOUS SAFETY EVENTS FOR SOLVE_IVP
# =========================================================
def earth_impact_event(t, x, *args):
    """Triggers zero-crossing when distance to Earth equals Earth radius."""
    # Extract mu if passed in args, otherwise import/use global mu
    mu_val = args[-1] if args else mu
    r1 = np.sqrt((x[0] + mu_val) ** 2 + x[1] ** 2 + x[2] ** 2)
    return r1 - R_EARTH_CANONICAL


earth_impact_event.terminal = True
earth_impact_event.direction = -1


def moon_impact_event(t, x, *args):
    """Triggers zero-crossing when distance to Moon equals Moon radius."""
    mu_val = args[-1] if args else mu
    r2 = np.sqrt((x[0] - 1 + mu_val) ** 2 + x[1] ** 2 + x[2] ** 2)
    return r2 - R_MOON_CANONICAL


moon_impact_event.terminal = True
moon_impact_event.direction = -1


def escape_event(t, x, *args):
    """Triggers zero-crossing when radial distance exceeds MAX_ESCAPE_RADIUS."""
    mu_val = args[-1] if args else mu
    r1 = np.sqrt((x[0] + mu_val) ** 2 + x[1] ** 2 + x[2] ** 2)
    return MAX_ESCAPE_RADIUS - r1


escape_event.terminal = True
escape_event.direction = -1


# Dynamic event list passed into RK45
SAFETY_EVENTS = [earth_impact_event, moon_impact_event, escape_event]


# =========================================================
# REFERENCE TRAJECTORY GENERATOR
# =========================================================
def generate_reference_trajectory(mission_type=None, max_retries=15):
    """Generates an unforced reference solution using RK45 integration.

    Uses continuous zero-crossing events to handle escape and surface impact.
    """
    if mission_type is None:
        mission_type = np.random.choice(
            ["earth_orbit", "moon_orbit", "transfer"]
        )

    # Quasi-stable baseline state vectors in canonical synodic frame
    if mission_type == "earth_orbit":
        # Medium Earth Orbit / High Geostationary-like synodic trajectory
        base_x0 = [-0.15, 0.00, 0.00, 0.00, 1.85, 0.05]
    elif mission_type == "moon_orbit":
        # Low Moon Orbit / Cislunar halo-like region
        base_x0 = [0.95, 0.00, 0.02, 0.00, 0.22, 0.00]
    elif mission_type == "transfer":
        # Earth-to-Moon direct transfer-like arc
        base_x0 = [-0.01215, 0.00, 0.00, 0.00, 3.12, 0.00]
    else:
        raise ValueError(f"Unknown mission type: {mission_type}")

    t_span = (0, 6 * np.pi)

    for attempt in range(max_retries):
        if attempt == 0:
            x0_ref = np.array(base_x0, dtype=np.float64)
        else:
            # Scaled perturbations to find a non-escaping trajectory window
            pos_pert = np.random.uniform(-0.002, 0.002, size=3)
            vel_pert = np.random.uniform(-0.01, 0.01, size=3)
            x0_ref = np.array(base_x0, dtype=np.float64) + np.hstack(
                [pos_pert, vel_pert]
            )

        sol = solve_ivp(
            crtbp_eom,
            t_span,
            x0_ref,
            args=(mu,),
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
            max_step=0.01,
            events=SAFETY_EVENTS,
            dense_output=True,
        )

        # Check if trajectory integrated without triggering early termination
        has_terminated = any(len(ev) > 0 for ev in sol.t_events)

        if sol.status == 0 and not has_terminated:
            return mission_type, sol

    # Fallback: if all retries hit an event boundary, return the longest valid run
    return mission_type, sol


def get_mission_description(mission_type):
    return MISSION_DESCRIPTIONS[mission_type]