"""Stage 1 baseline: no-drag trajectory must match closed-form solution."""
import math

import numpy as np
import pytest

from src.physics.trajectory import simulate_trajectory


def _analytic_landing(v, theta_deg, h0, target_h, g=9.81):
    """Closed-form (no drag): when does y(t) = target_h on the descending leg?"""
    th = math.radians(theta_deg)
    vx = v * math.cos(th)
    vy0 = v * math.sin(th)
    # h0 + vy0 t - 0.5 g t² = target_h
    # 0.5 g t² - vy0 t + (target_h - h0) = 0
    a, b, c = 0.5 * g, -vy0, target_h - h0
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    t = (-b + math.sqrt(disc)) / (2 * a)  # later root = descending crossing
    x = vx * t
    vy_at = vy0 - g * t
    return t, x, vx, vy_at


def test_no_drag_matches_analytical_45deg():
    v, theta, h0, target_h = 18.0, 45.0, 1.0, 1.8
    expect = _analytic_landing(v, theta, h0, target_h)
    assert expect is not None
    t_a, x_a, vx_a, vy_a = expect

    traj = simulate_trajectory(
        v_exit=v, theta_deg=theta, exit_height=h0,
        target_height=target_h, ball_mass=0.215, ball_radius=0.075,
        drag_coefficient=0.0,
    )
    assert traj.hit
    assert traj.hit_t == pytest.approx(t_a, abs=1e-4)
    assert traj.hit_x == pytest.approx(x_a, abs=1e-3)
    assert traj.hit_vx == pytest.approx(vx_a, abs=1e-4)
    assert traj.hit_vy == pytest.approx(vy_a, abs=1e-3)


def test_no_drag_high_arc():
    v, theta, h0, target_h = 22.0, 70.0, 0.5, 1.83
    expect = _analytic_landing(v, theta, h0, target_h)
    assert expect is not None
    _, x_a, _, _ = expect

    traj = simulate_trajectory(v, theta, h0, target_h, 0.215, 0.075, 0.0)
    assert traj.hit
    assert traj.hit_x == pytest.approx(x_a, abs=1e-3)


def test_drag_reduces_range():
    v, theta, h0, target_h = 22.0, 45.0, 0.6, 0.6
    no_drag = simulate_trajectory(v, theta, h0, target_h, 0.215, 0.075, 0.0)
    with_drag = simulate_trajectory(v, theta, h0, target_h, 0.215, 0.075, 0.5)
    assert no_drag.hit and with_drag.hit
    assert with_drag.hit_x < no_drag.hit_x


def test_too_weak_shot_misses_high_target():
    traj = simulate_trajectory(
        v_exit=2.0, theta_deg=45.0, exit_height=0.5,
        target_height=10.0, ball_mass=0.215, ball_radius=0.075,
        drag_coefficient=0.5,
    )
    assert not traj.hit


def test_entry_angle_increases_with_arc_height():
    """Higher launch angle → steeper entry angle."""
    args = dict(exit_height=0.5, target_height=1.83,
                ball_mass=0.215, ball_radius=0.075, drag_coefficient=0.0)
    low = simulate_trajectory(v_exit=15.0, theta_deg=40.0, **args)
    high = simulate_trajectory(v_exit=15.0, theta_deg=70.0, **args)
    assert low.hit and high.hit
    assert high.entry_angle_deg > low.entry_angle_deg
