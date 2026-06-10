"""Distance → optimal (v_exit, hood angle) → flywheel/hood-roller RPMs.

Strategy: for each hood angle θ in the mechanical range, find the v_exit that
lands the ball exactly at hub center (1D root-find seeded with the no-drag
analytical estimate). Then optimize θ to maximize entry quality. This avoids
the degenerate "overshoot to the hex far edge" behavior of unconstrained 2D
optimization.
"""
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from src.geometry.target import HexTarget
from src.physics.shooter import DualRoller
from src.physics.trajectory import TrajectoryResult, simulate_trajectory


@dataclass
class ShotSolution:
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
    boundary_margin_m: float    # apothem − ball_r − |x_offset| − clump_half_width
    cost: float
    trajectory: Optional[TrajectoryResult]
    note: str = ""


def _v_no_drag_for_centered_landing(
    distance: float, theta_deg: float, exit_height: float, target_height: float,
    gravity: float,
) -> Optional[float]:
    """Closed-form v (no drag) such that ball lands at x=distance at y=target_height.
    Returns None if not feasible (target too high for this θ at any v)."""
    th = math.radians(theta_deg)
    dh = target_height - exit_height
    if math.sin(th) < 1e-6 or math.cos(th) < 1e-6:
        return None
    # exit_h + d·tan θ - 0.5g·d²/(v² cos² θ) = target_h
    denom = distance * math.tan(th) - dh
    if denom <= 0:
        return None  # no trajectory reaches target_h at horizontal distance d for this θ
    v_sq = 0.5 * gravity * distance ** 2 / (math.cos(th) ** 2 * denom)
    return math.sqrt(v_sq)


def _v_for_centered_landing(
    distance: float,
    theta_deg: float,
    exit_height: float,
    target_height: float,
    ball_mass: float,
    ball_radius: float,
    drag_coefficient: float,
    v_min: float,
    v_max: float,
    air_density: float,
    gravity: float,
) -> Optional[float]:
    """At hood angle θ, find v ∈ [v_min, v_max] such that traj hits at x=distance.

    Seeds brentq with the no-drag analytical estimate, then brackets ±50% around it
    to stay in the smooth/continuous region of offset(v) (i.e. where traj.hit=True).
    """
    v_no_drag = _v_no_drag_for_centered_landing(
        distance, theta_deg, exit_height, target_height, gravity,
    )
    if v_no_drag is None:
        return None
    # With drag, real v is always >= no-drag v (drag eats horizontal energy).
    # Bracket from v_no_drag (likely undershoots with drag) to v_no_drag * 1.6 (overshoots).
    v_lo = max(v_no_drag * 0.95, v_min)
    v_hi = min(v_no_drag * 2.0, v_max)
    if v_lo >= v_hi:
        return None

    def offset(v):
        traj = simulate_trajectory(
            v, theta_deg, exit_height, target_height,
            ball_mass, ball_radius, drag_coefficient,
            air_density=air_density, gravity=gravity,
        )
        if not traj.hit:
            return -distance
        return traj.hit_x - distance

    o_lo = offset(v_lo)
    o_hi = offset(v_hi)

    # Expand v_hi until overshoot, capped at v_max.
    while o_hi < 0 and v_hi < v_max:
        v_hi = min(v_hi * 1.3, v_max)
        o_hi = offset(v_hi)
    if o_hi < 0:
        return None  # even v_max undershoots

    # Shrink v_lo if it overshoots (rare with this seed but possible at low arc).
    while o_lo > 0 and v_lo > v_min:
        v_lo = max(v_lo * 0.7, v_min)
        o_lo = offset(v_lo)
    if o_lo > 0:
        return None

    if o_lo == 0:
        return v_lo
    if o_hi == 0:
        return v_hi
    try:
        return brentq(offset, v_lo, v_hi, xtol=1e-3, maxiter=30)
    except (RuntimeError, ValueError):
        return None


def solve_shot(
    distance: float,
    exit_height: float,
    ball_mass: float,
    ball_radius: float,
    drag_coefficient: float,
    target: HexTarget,
    shooter: DualRoller,
    theta_range_deg: tuple[float, float] = (30.0, 75.0),
    v_range_ms: Optional[tuple[float, float]] = None,
    theta_resolution: int = 60,
    air_density: float = 1.225,
    gravity: float = 9.81,
    w_entry_angle: float = 1.0,
    w_v_exit: float = 3.0,
    w_tof: float = 1.1,
    ball_clump_half_width_m: float = 0.0,
) -> ShotSolution:
    """Find (v_exit, θ) such that ball lands at hub center with best entry quality.

    Cost: w_entry_angle·(90°-entry) + w_v_exit·v_exit² + w_tof·TOF²
      - entry term (linear): less rim-bounce risk
      - v_exit term: penalize unnecessary motor stress
      - TOF² term: dispersion proxy — longer flight amplifies input errors → wider scatter

    Defaults are the V3.1 physics-calibrated weights (see config/robot.yaml for the
    derivation): entry nearly off (foam ball doesn't bounce off the rim), v² for
    energy, w_tof from vortex-shedding dispersion (σ_x ≈ 0.04·T²; Darbois-Texier,
    NJP 2016). Beware large w_tof values: they push the solution into the low-arc
    region where θ-noise scatter explodes (the original w_tof=200 mistake).
    """
    v_top_max = shooter.top.surface_speed_ms(shooter.top.max_flywheel_rpm)
    v_bot_max = shooter.bottom.surface_speed_ms(shooter.bottom.max_flywheel_rpm)
    v_mech_max = min(v_top_max, v_bot_max)

    if v_range_ms is None:
        v_lo, v_hi = 0.5, v_mech_max
    else:
        v_lo = v_range_ms[0]
        v_hi = min(v_range_ms[1], v_mech_max)

    theta_min, theta_max = theta_range_deg
    thetas = np.linspace(theta_min, theta_max, theta_resolution)

    feasible: list[tuple[float, float, TrajectoryResult]] = []
    for th in thetas:
        v = _v_for_centered_landing(
            distance, float(th), exit_height, target.height_m,
            ball_mass, ball_radius, drag_coefficient,
            v_lo, v_hi, air_density, gravity,
        )
        if v is None:
            continue
        traj = simulate_trajectory(
            v, float(th), exit_height, target.height_m,
            ball_mass, ball_radius, drag_coefficient,
            air_density=air_density, gravity=gravity,
        )
        if not traj.hit or abs(traj.hit_x - distance) > 0.05:
            continue  # brentq false root, drop
        feasible.append((float(th), float(v), traj))

    if not feasible:
        return ShotSolution(
            success=False, v_exit_ms=0.0, hood_angle_deg=0.0,
            flywheel_rpm=0.0, hood_roller_rpm=0.0,
            entry_angle_deg=0.0, entry_speed_ms=0.0, time_of_flight_s=0.0,
            hit_x_m=0.0, x_offset_m=0.0, boundary_margin_m=0.0,
            cost=float('inf'), trajectory=None,
            note=(f"no θ ∈ {theta_range_deg} can land at distance {distance:.2f} m "
                  f"within v_exit ≤ {v_mech_max:.1f} m/s"),
        )

    def cost(theta, v, traj):
        return (
            w_entry_angle * (90.0 - traj.entry_angle_deg)
            + w_v_exit * v ** 2
            + w_tof * traj.hit_t ** 2
        )

    th_best, v_best, traj_best = min(feasible, key=lambda x: cost(*x))

    # Optional local refine on θ (continuous 1D)
    def cost_of_theta(th):
        v = _v_for_centered_landing(
            distance, float(th), exit_height, target.height_m,
            ball_mass, ball_radius, drag_coefficient,
            v_lo, v_hi, air_density, gravity,
        )
        if v is None:
            return 1e9
        traj = simulate_trajectory(
            v, float(th), exit_height, target.height_m,
            ball_mass, ball_radius, drag_coefficient,
            air_density=air_density, gravity=gravity,
        )
        return cost(th, v, traj)

    # Refine within a small bracket around grid winner
    span = (thetas[1] - thetas[0]) if len(thetas) > 1 else 0.5
    th_lo_refine = max(theta_min, th_best - span)
    th_hi_refine = min(theta_max, th_best + span)
    if th_hi_refine > th_lo_refine:
        res = minimize_scalar(
            cost_of_theta,
            bracket=None,
            bounds=(th_lo_refine, th_hi_refine),
            method='bounded',
            options={'xatol': 1e-3},
        )
        if res.fun < cost(th_best, v_best, traj_best):
            v_refined = _v_for_centered_landing(
                distance, float(res.x), exit_height, target.height_m,
                ball_mass, ball_radius, drag_coefficient,
                v_lo, v_hi, air_density, gravity,
            )
            if v_refined is not None:   # keep the grid winner if the re-solve falls through
                th_best, v_best = float(res.x), v_refined
                traj_best = simulate_trajectory(
                    v_best, th_best, exit_height, target.height_m,
                    ball_mass, ball_radius, drag_coefficient,
                    air_density=air_density, gravity=gravity,
                )

    rpm_top, rpm_bottom = shooter.rpms_for_exit_velocity(v_best, omega_spin=0.0)
    over_limit = (
        rpm_top > shooter.top.max_flywheel_rpm
        or rpm_bottom > shooter.bottom.max_flywheel_rpm
    )
    margin = target.apothem - ball_radius - abs(traj_best.hit_x - distance) - ball_clump_half_width_m
    notes = []
    if over_limit:
        notes.append("RPM exceeds motor free speed")
    if margin < 0:
        notes.append(f"4-ball clump margin {margin:+.2f} m < 0 (some balls miss)")
    elif margin < 0.05:
        notes.append(f"tight clump margin {margin:.2f} m (< 5 cm)")

    return ShotSolution(
        success=True,
        v_exit_ms=float(v_best),
        hood_angle_deg=float(th_best),
        flywheel_rpm=float(rpm_top),
        hood_roller_rpm=float(rpm_bottom),
        entry_angle_deg=float(traj_best.entry_angle_deg),
        entry_speed_ms=float(traj_best.entry_speed_ms),
        time_of_flight_s=float(traj_best.hit_t),
        hit_x_m=float(traj_best.hit_x),
        x_offset_m=float(traj_best.hit_x - distance),
        boundary_margin_m=float(margin),
        cost=float(cost(th_best, v_best, traj_best)),
        trajectory=traj_best,
        note="; ".join(notes),
    )
