import pytest

from src.geometry.target import HexTarget
from src.optimizer.solve import solve_shot
from src.physics.shooter import DualRoller, Roller


def _shooter():
    return DualRoller(
        top=Roller(diameter_m=0.1016, slip_factor=0.18, motor_free_rpm=6000),
        bottom=Roller(diameter_m=0.0508, slip_factor=0.18, motor_free_rpm=6000),
        ball_radius_m=0.075,
    )


def _target():
    return HexTarget(across_flats_m=1.060, height_m=1.829)


@pytest.mark.parametrize("distance_m", [2.0, 3.5, 5.0])
def test_solver_finds_feasible_shot(distance_m):
    sol = solve_shot(
        distance=distance_m, exit_height=0.6,
        ball_mass=0.215, ball_radius=0.075, drag_coefficient=0.5,
        target=_target(), shooter=_shooter(),
        theta_range_deg=(25, 75), v_range_ms=(5, 30),
    )
    assert sol.success, f"{distance_m}m should be reachable; note={sol.note}"
    assert sol.entry_angle_deg > 0
    assert sol.v_exit_ms > 0
    assert sol.flywheel_rpm > 0


def test_unreachable_distance_returns_failure():
    sol = solve_shot(
        distance=100.0, exit_height=0.6,
        ball_mass=0.215, ball_radius=0.075, drag_coefficient=0.5,
        target=_target(), shooter=_shooter(),
        theta_range_deg=(25, 75), v_range_ms=(5, 30),
    )
    assert not sol.success


def test_farther_distance_needs_more_exit_velocity():
    """Closer shots need less v_exit than farther ones (monotonic)."""
    short = solve_shot(
        distance=2.5, exit_height=0.6,
        ball_mass=0.215, ball_radius=0.075, drag_coefficient=0.5,
        target=_target(), shooter=_shooter(),
        theta_range_deg=(25, 75), v_range_ms=(5, 30),
    )
    long_ = solve_shot(
        distance=5.5, exit_height=0.6,
        ball_mass=0.215, ball_radius=0.075, drag_coefficient=0.5,
        target=_target(), shooter=_shooter(),
        theta_range_deg=(25, 75), v_range_ms=(5, 30),
    )
    assert short.success and long_.success
    assert long_.v_exit_ms > short.v_exit_ms
