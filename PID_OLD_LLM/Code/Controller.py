import numpy as np
from scipy.linalg import inv
from CRTBP import crtbp_eom, second_partials, DU, TU

# Acceleration unit conversion: AU = DU / TU^2 (km/s^2) -> convert to m/s^2
AU_M_S2 = (DU * 1000.0) / (TU ** 2)  # ~ 0.002723 m/s^2


class CRTBPController:
    """
    Continuous LTI/Gain-Scheduled Controller and Closed-Loop Model for CRTBP Spacecraft.
    Supports dynamic adaptive thrust limits and gain rescheduling.
    """
    def __init__(self, mu, reference_state, Kp=None, Kd=None, Ki=None, u_max_m_s2=0.01):
        self.mu = mu
        self.x_ref = np.array(reference_state, dtype=np.float64).flatten()
        
        # Max thrust limit handling (m/s^2 to non-dimensional acceleration)
        self.u_max_m_s2 = u_max_m_s2
        self.u_max = u_max_m_s2 / AU_M_S2 if u_max_m_s2 is not None else None
        
        self.is_saturated = False  # Saturation flag for anti-windup integration
        self.sat_ratio = 1.0       # Continuous saturation measure
        
        # Gain matrices updated for canonical non-dimensional state feedback
        self.Kp = Kp if Kp is not None else np.diag([2500.0, 2500.0, 2500.0])
        self.Kd = Kd if Kd is not None else np.diag([300.0, 300.0, 300.0])
        self.Ki = Ki if Ki is not None else np.diag([10.0, 10.0, 10.0])
        
        # Build Plant, Controller, and Closed-Loop State Space Models
        self._build_plant_ss()
        self._build_controller_ss()
        self._build_closed_loop_ss()

    def set_thrust_limit(self, u_max_m_s2):
        """Dynamically updates the thrust saturation bound (Adaptive Bounds)."""
        self.u_max_m_s2 = u_max_m_s2
        self.u_max = u_max_m_s2 / AU_M_S2 if u_max_m_s2 is not None else None

    def set_gains(self, Kp=None, Kd=None, Ki=None):
        """Dynamically reschedules controller gains."""
        if Kp is not None: self.Kp = Kp
        if Kd is not None: self.Kd = Kd
        if Ki is not None: self.Ki = Ki
        self._build_controller_ss()
        self._build_closed_loop_ss()

    def _build_plant_ss(self, ref_state=None):
        """Linearizes CRTBP dynamics around a given reference state vector."""
        if ref_state is not None:
            self.x_ref = np.array(ref_state, dtype=np.float64).flatten()

        x, y, z = self.x_ref[:3]
        
        # Compute second partial derivatives using imported helper function
        Uxx, Uxy, Uxz, Uyy, Uyz, Uzz = second_partials(x, y, z, self.mu)

        # 6x6 Plant A matrix (Jacobian)
        self.Ap = np.array([
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [Uxx, Uxy, Uxz, 0.0, 2.0, 0.0],
            [Uxy, Uyy, Uyz,-2.0, 0.0, 0.0],
            [Uxz, Uyz, Uzz, 0.0, 0.0, 0.0]
        ], dtype=np.float64)

        # 6x3 Plant B matrix
        self.Bp = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        self.Cp = np.eye(6)
        self.Dp = np.zeros((6, 3))

    def _build_controller_ss(self):
        """Builds State-Space Controller matrices."""
        self.Ac = np.zeros((3, 3))
        self.Bc1 = np.hstack([-np.eye(3), np.zeros((3, 3))])
        self.Bc2 = np.hstack([np.eye(3), np.zeros((3, 3))])
        
        self.Cc = self.Ki
        
        K_fb = np.hstack([self.Kp, self.Kd])
        self.Dc1 = -K_fb
        self.Dc2 = K_fb

    def _build_closed_loop_ss(self):
        """Forms closed-loop state space system."""
        I_dp = np.eye(self.Dc1.shape[0])
        Z = inv(I_dp - self.Dc1 @ self.Dp)

        A11 = self.Ap + self.Bp @ Z @ self.Dc1 @ self.Cp
        A12 = self.Bp @ Z @ self.Cc
        A21 = self.Bc1 @ (self.Cp + self.Dp @ Z @ self.Dc1 @ self.Cp)
        A22 = self.Ac + self.Bc1 @ self.Dp @ Z @ self.Cc

        self.Acl = np.block([[A11, A12], [A21, A22]])
        
        B1 = self.Bp @ Z @ self.Dc2
        B2 = self.Bc2 + self.Bc1 @ self.Dp @ Z @ self.Dc2
        self.Bcl = np.vstack([B1, B2])

        self.Ccl = np.hstack([self.Cp + self.Dp @ Z @ self.Dc1 @ self.Cp, self.Dp @ Z @ self.Cc])
        self.Dcl = self.Dp @ Z @ self.Dc2

    def check_stability(self):
        """Closed-loop stability check."""
        eigenvalues = np.linalg.eigvals(self.Acl)
        max_real = np.max(np.real(eigenvalues))
        return max_real <= 0

    def compute_control_action(self, state_error, xc, current_ref_state=None):
        """
        Calculates control acceleration vector u = [ux, uy, uz]^T with smooth adaptive saturation.
        """
        if current_ref_state is not None:
            self._build_plant_ss(current_ref_state)

        state_err = np.asarray(state_error, dtype=np.float64).flatten()
        pos_err = state_err[:3]
        vel_err = state_err[3:6]
        xc_vec = np.asarray(xc, dtype=np.float64).flatten()
        
        # Raw linear control acceleration output
        u_raw = self.Ki @ xc_vec - self.Kp @ pos_err - self.Kd @ vel_err
        
        # Smooth continuous saturation using hyperbolic tangent (tanh)
        if self.u_max is not None and self.u_max > 0:
            u_norm = np.linalg.norm(u_raw)
            if u_norm > 1e-12:
                # Smooth saturation preserving vector direction
                u_mag_sat = self.u_max * np.tanh(u_norm / self.u_max)
                u = u_raw * (u_mag_sat / u_norm)
                self.sat_ratio = u_mag_sat / u_norm
                self.is_saturated = self.sat_ratio < 0.95
            else:
                u = u_raw
                self.sat_ratio = 1.0
                self.is_saturated = False
        else:
            u = u_raw
            self.sat_ratio = 1.0
            self.is_saturated = False
                
        return u


# =============================================================================
# CRTBP Integration Function with Active Controller & Anti-Windup
# =============================================================================

def controlled_crtbp_eom(t, Y_aug, mu, reference_orbit_fn, controller, perturbations_fn=None):
    """
    Augmented system EOM for solve_ivp with smooth anti-windup integration:
    Y_aug = [x, y, z, vx, vy, vz, xc_x, xc_y, xc_z]
    """
    Y_aug = np.asarray(Y_aug, dtype=np.float64).flatten()
    state = Y_aug[:6]
    xc = Y_aug[6:9]

    # Target reference state
    ref_state = np.squeeze(reference_orbit_fn(t)).flatten()
    state_error = state - ref_state

    # Control acceleration output
    u = controller.compute_control_action(state_error, xc, current_ref_state=ref_state)

    # Natural CRTBP dynamics
    natural_derivs = crtbp_eom(t, state, mu)

    # Disturbance accelerations (e.g. SRP)
    a_dist = perturbations_fn(t, state) if perturbations_fn is not None else np.zeros(3)

    # Combine natural acceleration, control, and perturbations
    ax = natural_derivs[3] + u[0] + a_dist[0]
    ay = natural_derivs[4] + u[1] + a_dist[1]
    az = natural_derivs[5] + u[2] + a_dist[2]

    # Smooth anti-windup integrator dynamics: scale down integration when saturated
    dxc_dt = -state_error[:3] * controller.sat_ratio

    return [natural_derivs[0], natural_derivs[1], natural_derivs[2], ax, ay, az, dxc_dt[0], dxc_dt[1], dxc_dt[2]]