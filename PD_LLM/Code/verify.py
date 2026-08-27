import numpy as np

# Canonical units constants for Earth-Moon system
MU = 0.01215058560962404  # CR3BP mass parameter
LU_TO_KM = 384400.0        # Length unit to kilometers
VU_TO_MS = 1024.54         # Velocity unit to meters per second

EARTH_POS = np.array([-MU, 0.0, 0.0])
MOON_POS = np.array([1.0 - MU, 0.0, 0.0])


def verify_physical_bounds(state, pos_err, vel_err, control_vec):

    pos_err_km = np.linalg.norm(pos_err) * LU_TO_KM
    vel_err_ms = np.linalg.norm(vel_err) * VU_TO_MS
    u_mag_ms2 = np.linalg.norm(control_vec) * (VU_TO_MS / 375190.0) # approx canonical acceleration conversion

    # Discard step if position error > 2,000 km or velocity error > 100 m/s
    if pos_err_km > 2000.0 or vel_err_ms > 100.0:
        return False

    # Discard step if requested control acceleration is unrealistically high (> 0.01 m/s^2)
    if u_mag_ms2 > 0.01:
        return False

    return True


def verify_and_get_dynamic_metadata(state):

    r_earth_km = np.linalg.norm(state[:3] - EARTH_POS) * LU_TO_KM
    r_moon_km = np.linalg.norm(state[:3] - MOON_POS) * LU_TO_KM

    if r_moon_km < 60000.0:
        mission_type = "Moon Orbit"
        desc = "Maintain a cislunar trajectory near the Moon."
    elif r_earth_km < 120000.0:
        mission_type = "Earth Orbit"
        desc = "Maintain an orbit trajectory in the Earth region."
    else:
        mission_type = "Cislunar Transfer"
        desc = "Execute a transfer trajectory through Earth-Moon synodic space."

    return mission_type, desc