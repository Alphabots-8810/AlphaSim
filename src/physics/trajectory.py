"""2D projectile trajectory with quadratic drag. Magnus is a stub for stage 2."""
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class TrajectoryResult:
    times: np.ndarray
    positions: np.ndarray      # shape (N, 2): columns (x, y)
    velocities: np.ndarray     # shape (N, 2): columns (vx, vy)
    hit: bool
    hit_x: Optional[float] = None
    hit_t: Optional[float] = None
    hit_vx: Optional[float] = None
    hit_vy: Optional[float] = None

    @property
    def entry_angle_deg(self) -> Optional[float]:
        # 90° = straight down; 0° = grazing horizontal.
        if not self.hit:
            return None
        return float(np.degrees(np.arctan2(-self.hit_vy, abs(self.hit_vx))))

    @property
    def entry_speed_ms(self) -> Optional[float]:
        if not self.hit:
            return None
        return float(np.hypot(self.hit_vx, self.hit_vy))


def simulate_trajectory(
    v_exit: float,
    theta_deg: float,
    exit_height: float,
    target_height: float,
    ball_mass: float,
    ball_radius: float,
    drag_coefficient: float,
    air_density: float = 1.225,
    gravity: float = 9.81,
    spin_rate: float = 0.0,
    magnus_cl_fn: Optional[Callable[[float], float]] = None,
    t_max: float = 6.0,
    max_step: float = 0.02,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> TrajectoryResult:
    """Integrate ball trajectory until it crosses y=target_height descending, or t_max.

    spin_rate (rad/s, positive=backspin) and magnus_cl_fn(S)→Cl are stage-2 hooks.
    Default Magnus off when spin_rate=0 or magnus_cl_fn is None.
    """
    theta = np.radians(theta_deg)
    vx0 = v_exit * np.cos(theta)
    vy0 = v_exit * np.sin(theta)

    area = np.pi * ball_radius ** 2
    k_drag = 0.5 * air_density * drag_coefficient * area / ball_mass
    magnus_active = (spin_rate != 0.0) and (magnus_cl_fn is not None)
    k_mag_base = 0.5 * air_density * area / ball_mass

    def rhs(t, s):
        x, y, vx, vy = s
        v = np.hypot(vx, vy)
        ax = -k_drag * v * vx
        ay = -gravity - k_drag * v * vy
        if magnus_active and v > 1e-9:
            # Spin parameter S = ω r / v; lift force perpendicular to velocity.
            # For backspin (ω>0) in 2D, lift rotates v 90° CCW → (-vy, +vx) direction.
            S = spin_rate * ball_radius / v
            Cl = magnus_cl_fn(S)
            mag_acc = k_mag_base * Cl * v
            ax += -mag_acc * vy
            ay += mag_acc * vx
        return [vx, vy, ax, ay]

    def hit_target_plane(t, s):
        return s[1] - target_height
    hit_target_plane.terminal = True
    hit_target_plane.direction = -1  # only catch downward crossings

    sol = solve_ivp(
        rhs,
        t_span=(0.0, t_max),
        y0=[0.0, exit_height, vx0, vy0],
        events=[hit_target_plane],
        max_step=max_step,
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )

    positions = sol.y[0:2, :].T
    velocities = sol.y[2:4, :].T

    if len(sol.t_events[0]) > 0:
        t_hit = float(sol.t_events[0][0])
        s_hit = sol.y_events[0][0]
        return TrajectoryResult(
            times=sol.t, positions=positions, velocities=velocities,
            hit=True,
            hit_x=float(s_hit[0]), hit_t=t_hit,
            hit_vx=float(s_hit[2]), hit_vy=float(s_hit[3]),
        )
    return TrajectoryResult(
        times=sol.t, positions=positions, velocities=velocities, hit=False,
    )
