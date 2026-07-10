"""4414-style robust optimizer: find (v, θ) maximizing distance to valid-region boundary.

Unlike the heuristic optimizer (cost = entry + v² + TOF² weighted sum), this asks
the right physical question directly:

    "Among all (v, θ) that score, which one tolerates the largest noise in either
     direction before missing?"

Output: best (v*, θ*) + symmetric tolerance ±Δv, ±Δθ (the largest noise box you
can take and still score). No σ_v / σ_θ measurements needed.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import distance_transform_cdt
from scipy.optimize import minimize

from src.geometry.target import HexTarget
from src.physics.shooter import DualRoller
from src.physics.trajectory import TrajectoryResult, simulate_trajectory


def is_scoring(v, theta_deg, distance, exit_h, ball_mass, ball_r, Cd, target,
               air_density=1.225, gravity=9.81, extra_margin=0.0):
    """True if (v, θ) trajectory lands within effective hex window.

    extra_margin shrinks the window further — e.g. the 4-ball clump half-width,
    so all balls of a dumper volley fit, matching solve_shot's margin convention."""
    if v <= 0 or theta_deg <= 0 or theta_deg >= 90:
        return False
    traj = simulate_trajectory(
        v, theta_deg, exit_h, target.height_m,
        ball_mass, ball_r, Cd,
        air_density=air_density, gravity=gravity,
    )
    return traj.hit and abs(traj.hit_x - distance) <= (target.apothem - ball_r - extra_margin)


def _direction_margin(v0, th0, dv_unit, dth_unit, args,
                      initial_step=0.05, max_expand=14, bisect_iters=10):
    """Largest k > 0 such that (v0 + k·dv_unit, th0 + k·dth_unit) still scores.
    Returns the bisected margin in the (dv_unit, dth_unit) direction."""
    if not is_scoring(v0, th0, *args):
        return 0.0
    k_hi = initial_step
    expanded = 0
    while expanded < max_expand and is_scoring(
            v0 + k_hi * dv_unit, th0 + k_hi * dth_unit, *args):
        k_hi *= 1.6
        expanded += 1
    if expanded >= max_expand:
        return k_hi
    k_lo = k_hi / 1.6
    for _ in range(bisect_iters):
        k_mid = 0.5 * (k_lo + k_hi)
        if is_scoring(v0 + k_mid * dv_unit, th0 + k_mid * dth_unit, *args):
            k_lo = k_mid
        else:
            k_hi = k_mid
    return k_lo


def continuous_axis_margins(v, theta, args):
    """Return (Δv_pos, Δv_neg, Δθ_pos, Δθ_neg) continuously via bisection."""
    return (
        _direction_margin(v, theta,  1.0,  0.0, args),
        _direction_margin(v, theta, -1.0,  0.0, args),
        _direction_margin(v, theta,  0.0,  1.0, args),
        _direction_margin(v, theta,  0.0, -1.0, args),
    )


def refine_robust_center(
    v0, theta0, scoring_args, theta_range, v_range,
    v_scale=1.0, theta_scale=2.0,
    local_radius_v=0.8, local_radius_theta=6.0,
    maxiter=25,
):
    """Refine (v0, θ0) by Nelder-Mead maximizing min(margins) in local box.

    v_scale and theta_scale normalize axes so 1 m/s of v noise ≈ theta_scale°
    of θ noise in the objective. Default (1, 2): 1 m/s ≈ 2°.
    """
    v_lo = max(v_range[0], v0 - local_radius_v)
    v_hi = min(v_range[1], v0 + local_radius_v)
    t_lo = max(theta_range[0], theta0 - local_radius_theta)
    t_hi = min(theta_range[1], theta0 + local_radius_theta)

    def neg_margin(params):
        v, theta = params
        if v < v_lo or v > v_hi or theta < t_lo or theta > t_hi:
            return 1e6
        if not is_scoring(v, theta, *scoring_args):
            return 1e3
        dvp, dvn, dtp, dtn = continuous_axis_margins(v, theta, scoring_args)
        norm = [dvp / v_scale, dvn / v_scale, dtp / theta_scale, dtn / theta_scale]
        # Tiebreak among equal-margin points: prefer lower v_exit (save battery).
        # Weight is tiny so it only matters when margins are within ~1e-4 of each other.
        return -min(norm) + 1e-4 * v

    res = minimize(
        neg_margin, x0=[v0, theta0],
        method='Nelder-Mead',
        bounds=[(v_lo, v_hi), (t_lo, t_hi)],
        options={'maxiter': maxiter, 'xatol': 0.02, 'fatol': 0.003},
    )
    if res.fun >= 1e3:
        return float(v0), float(theta0)
    return float(res.x[0]), float(res.x[1])


@dataclass
class RobustShotSolution:
    success: bool
    v_exit_ms: float
    hood_angle_deg: float
    flywheel_rpm: float
    hood_roller_rpm: float
    entry_angle_deg: float
    entry_speed_ms: float
    time_of_flight_s: float
    hit_x_m: float
    x_offset_m: float
    margin_v_ms: float            # ±Δv tolerance at chosen θ
    margin_theta_deg: float       # ±Δθ tolerance at chosen v
    boundary_distance_grid: float # Euclidean distance to boundary in grid cells
    valid_mask: Optional[np.ndarray] = None
    v_grid: Optional[np.ndarray] = None
    theta_grid: Optional[np.ndarray] = None
    trajectory: Optional[TrajectoryResult] = None
    note: str = ""


def compute_valid_mask(
    distance: float,
    exit_height: float,
    ball_mass: float,
    ball_radius: float,
    drag_coefficient: float,
    target: HexTarget,
    theta_range_deg: tuple[float, float],
    v_range_ms: tuple[float, float],
    grid_n: int,
    air_density: float = 1.225,
    gravity: float = 9.81,
    ball_clump_half_width_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mask, vs, thetas). mask[i, j] = True if (vs[i], thetas[j]) scores."""
    thetas = np.linspace(*theta_range_deg, grid_n)
    vs = np.linspace(*v_range_ms, grid_n)
    mask = np.zeros((grid_n, grid_n), dtype=bool)
    eff_radius = target.apothem - ball_radius - ball_clump_half_width_m

    for i, v in enumerate(vs):
        for j, th in enumerate(thetas):
            traj = simulate_trajectory(
                float(v), float(th), exit_height, target.height_m,
                ball_mass, ball_radius, drag_coefficient,
                air_density=air_density, gravity=gravity,
            )
            if traj.hit and abs(traj.hit_x - distance) <= eff_radius:
                mask[i, j] = True
    return mask, vs, thetas


def find_robust_center(
    mask: np.ndarray, near_max_cells: int = 1,
) -> Optional[tuple[int, int, float]]:
    """Return (i, j, dist) — grid indices of the most-robust cell, with energy
    tiebreak: among all cells whose Chebyshev distance is within
    `near_max_cells` of the global max (i.e. "about equally" robust), prefer the
    LOWEST v_exit (lowest row index, since vs is ascending).

    Why ≤1 cell tolerance: when two plateaus (high-arc vs low-arc) have margins
    differing by < 1 grid cell, the optimizer's pick is essentially noise; user
    prefers lower v_exit for battery management. This also smooths the
    distance-sweep (no more flipping between equal-quality plateaus).
    """
    if not mask.any():
        return None
    dist = distance_transform_cdt(mask, metric='chessboard')
    max_dist = dist.max()
    if max_dist == 0:
        return None
    threshold = max(1, max_dist - near_max_cells)
    candidates = np.argwhere(dist >= threshold)
    # Sort by (i_v ascending → lowest v first, then j_theta ascending).
    sorted_idx = candidates[np.lexsort((candidates[:, 1], candidates[:, 0]))]
    i, j = int(sorted_idx[0, 0]), int(sorted_idx[0, 1])
    return i, j, float(dist[i, j])


def compute_axis_margins(
    mask: np.ndarray, i: int, j: int,
    vs: np.ndarray, thetas: np.ndarray,
) -> tuple[float, float]:
    """Symmetric ±Δv (along v axis at fixed θ_j) and ±Δθ (at fixed v_i)
    such that the box (i±di, j±dj) is entirely valid."""
    n_v, n_th = mask.shape
    di = 0
    while (i - di - 1 >= 0 and i + di + 1 < n_v
           and mask[i - di - 1, j] and mask[i + di + 1, j]):
        di += 1
    dj = 0
    while (j - dj - 1 >= 0 and j + dj + 1 < n_th
           and mask[i, j - dj - 1] and mask[i, j + dj + 1]):
        dj += 1
    dv = vs[1] - vs[0] if len(vs) > 1 else 0
    dth = thetas[1] - thetas[0] if len(thetas) > 1 else 0
    return di * dv, dj * dth


def solve_shot_robust(
    distance: float,
    exit_height: float,
    ball_mass: float,
    ball_radius: float,
    drag_coefficient: float,
    target: HexTarget,
    shooter: DualRoller,
    theta_range_deg: tuple[float, float] = (25.0, 75.0),
    v_range_ms: Optional[tuple[float, float]] = None,
    grid_n: int = 60,
    refine: bool = True,
    v_scale: float = 1.0,
    theta_scale: float = 2.0,
    seed: Optional[tuple[float, float]] = None,
    air_density: float = 1.225,
    gravity: float = 9.81,
    ball_clump_half_width_m: float = 0.0,
) -> RobustShotSolution:
    """Find the (v, θ) point most robust to noise, in the valid scoring region.

    ball_clump_half_width_m shrinks the scoring window like solve_shot's 4-ball
    clump margin, so both solvers agree on what counts as "all balls fit"."""
    v_top_max = shooter.top.surface_speed_ms(shooter.top.max_flywheel_rpm)
    v_bot_max = shooter.bottom.surface_speed_ms(shooter.bottom.max_flywheel_rpm)
    v_mech_max = min(v_top_max, v_bot_max)
    if v_range_ms is None:
        v_range_ms = (1.0, v_mech_max)
    else:
        v_range_ms = (v_range_ms[0], min(v_range_ms[1], v_mech_max))

    mask, vs, thetas = compute_valid_mask(
        distance, exit_height, ball_mass, ball_radius, drag_coefficient, target,
        theta_range_deg, v_range_ms, grid_n,
        air_density=air_density, gravity=gravity,
        ball_clump_half_width_m=ball_clump_half_width_m,
    )

    if not mask.any():
        return RobustShotSolution(
            success=False, v_exit_ms=0, hood_angle_deg=0,
            flywheel_rpm=0, hood_roller_rpm=0,
            entry_angle_deg=0, entry_speed_ms=0, time_of_flight_s=0,
            hit_x_m=0, x_offset_m=0,
            margin_v_ms=0, margin_theta_deg=0, boundary_distance_grid=0,
            valid_mask=mask, v_grid=vs, theta_grid=thetas,
            note=(f"no (v, θ) in grid (v≤{v_mech_max:.1f} m/s, "
                  f"θ∈{theta_range_deg}) scores at distance {distance:.2f} m"),
        )

    i, j, dist_grid = find_robust_center(mask)
    v_opt, theta_opt = float(vs[i]), float(thetas[j])

    scoring_args = (distance, exit_height, ball_mass, ball_radius,
                    drag_coefficient, target, air_density, gravity,
                    ball_clump_half_width_m)

    # NOTE on `seed` param: continuity seeding was tried but found to lock the
    # optimizer into suboptimal high-arc plateaus across distance sweeps (scipy
    # refines greedily; seed prevents jumping to better low-v plateaus). The
    # parameter is kept for API compatibility but not currently used. To smooth
    # the sweep, prefer (a) finer grid_n or (b) polynomial post-fit on the table.
    _ = seed  # silence unused-argument lint

    if refine:
        v_opt, theta_opt = refine_robust_center(
            v_opt, theta_opt, scoring_args,
            theta_range_deg, v_range_ms,
            v_scale=v_scale, theta_scale=theta_scale,
        )
        dvp, dvn, dtp, dtn = continuous_axis_margins(v_opt, theta_opt, scoring_args)
        margin_v = min(dvp, dvn)
        margin_theta = min(dtp, dtn)
    else:
        margin_v, margin_theta = compute_axis_margins(mask, i, j, vs, thetas)

    traj = simulate_trajectory(
        v_opt, theta_opt, exit_height, target.height_m,
        ball_mass, ball_radius, drag_coefficient,
        air_density=air_density, gravity=gravity,
    )
    rpm_top, rpm_bot = shooter.rpms_for_exit_velocity(v_opt, omega_spin=0.0)

    note = ""
    if rpm_top > shooter.top.max_flywheel_rpm or rpm_bot > shooter.bottom.max_flywheel_rpm:
        note = "RPM exceeds motor free speed"

    return RobustShotSolution(
        success=True,
        v_exit_ms=v_opt, hood_angle_deg=theta_opt,
        flywheel_rpm=float(rpm_top), hood_roller_rpm=float(rpm_bot),
        entry_angle_deg=float(traj.entry_angle_deg) if traj.hit else 0,
        entry_speed_ms=float(traj.entry_speed_ms) if traj.hit else 0,
        time_of_flight_s=float(traj.hit_t) if traj.hit else 0,
        hit_x_m=float(traj.hit_x) if traj.hit else 0,
        x_offset_m=float(traj.hit_x - distance) if traj.hit else 0,
        margin_v_ms=float(margin_v), margin_theta_deg=float(margin_theta),
        boundary_distance_grid=float(dist_grid),
        valid_mask=mask, v_grid=vs, theta_grid=thetas, trajectory=traj,
        note=note,
    )
