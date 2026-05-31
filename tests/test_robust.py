"""Sanity tests for the robust optimizer."""
import numpy as np

from src.geometry.target import HexTarget
from src.optimizer.robust import (
    compute_axis_margins,
    compute_valid_mask,
    find_robust_center,
    solve_shot_robust,
)
from src.physics.shooter import DualRoller, Roller


def _shooter():
    return DualRoller(
        top=Roller(diameter_m=0.1016, slip_factor=0.18, motor_free_rpm=6000),
        bottom=Roller(diameter_m=0.0508, slip_factor=0.18, motor_free_rpm=6000),
        ball_radius_m=0.075,
    )


def _target():
    return HexTarget(across_flats_m=1.060, height_m=1.829)


def test_valid_mask_has_some_hits():
    """At a reachable distance, some (v, θ) combos must score."""
    mask, vs, thetas = compute_valid_mask(
        distance=3.5, exit_height=0.46,
        ball_mass=0.215, ball_radius=0.075, drag_coefficient=0.5,
        target=_target(),
        theta_range_deg=(25, 75), v_range_ms=(3, 14),
        grid_n=20,
    )
    assert mask.any(), "should have some valid (v, θ) at reachable distance"
    assert mask.shape == (20, 20)


def test_find_robust_center_returns_interior_point():
    """Robust center must be inside the valid region."""
    mask, vs, thetas = compute_valid_mask(
        3.5, 0.46, 0.215, 0.075, 0.5, _target(), (25, 75), (3, 14), 20,
    )
    i, j, dist = find_robust_center(mask)
    assert mask[i, j], "robust center should be a True cell"
    assert dist >= 1.0, "expect at least 1 cell from boundary"


def test_robust_solver_finds_solution_at_reachable_distance():
    sol = solve_shot_robust(
        distance=3.5, exit_height=0.46,
        ball_mass=0.215, ball_radius=0.075, drag_coefficient=0.5,
        target=_target(), shooter=_shooter(),
        theta_range_deg=(25, 75), grid_n=60,
    )
    assert sol.success
    assert 25 <= sol.hood_angle_deg <= 75
    assert sol.v_exit_ms > 0
    assert sol.margin_v_ms > 0
    assert sol.margin_theta_deg > 0


def test_unreachable_distance_returns_failure():
    sol = solve_shot_robust(
        distance=100.0, exit_height=0.46,
        ball_mass=0.215, ball_radius=0.075, drag_coefficient=0.5,
        target=_target(), shooter=_shooter(),
        theta_range_deg=(25, 75), grid_n=20,
    )
    assert not sol.success


def test_axis_margins_zero_at_boundary():
    """Single-True cell has zero margins (no symmetric box fits)."""
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    dv, dth = compute_axis_margins(mask, 2, 2, np.arange(5), np.arange(5))
    assert dv == 0 and dth == 0


def test_axis_margins_grows_with_interior():
    """A 3x3 True block centered at (2, 2) has margin = 1 cell each side."""
    mask = np.zeros((7, 7), dtype=bool)
    mask[1:6, 1:6] = True
    dv, dth = compute_axis_margins(mask, 3, 3, np.arange(7), np.arange(7))
    assert dv == 2 and dth == 2
