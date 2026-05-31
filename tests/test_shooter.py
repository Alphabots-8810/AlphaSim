import math

import pytest

from src.physics.shooter import DualRoller, Roller


def make_dual(slip=0.0):
    r = Roller(diameter_m=0.1, slip_factor=slip)
    return DualRoller(top=r, bottom=r, ball_radius_m=0.075)


def test_equal_speeds_give_zero_spin():
    dual = make_dual(slip=0.0)
    rpm = 3000.0
    assert dual.spin_rate_rad_s(rpm, rpm) == pytest.approx(0.0, abs=1e-9)


def test_equal_speeds_exit_velocity_matches_omega_r():
    dual = make_dual(slip=0.0)
    rpm = 3000.0
    omega = rpm * 2 * math.pi / 60
    expected_v = omega * 0.05
    assert dual.exit_velocity_ms(rpm, rpm) == pytest.approx(expected_v, abs=1e-9)


def test_bottom_faster_than_top_gives_positive_backspin():
    dual = make_dual(slip=0.0)
    assert dual.spin_rate_rad_s(rpm_top=2000, rpm_bottom=4000) > 0


def test_inverse_kinematics_round_trip():
    dual = make_dual(slip=0.15)
    v_target = 15.0
    rpm_t, rpm_b = dual.rpms_for_exit_velocity(v_target, omega_spin=0.0)
    assert dual.exit_velocity_ms(rpm_t, rpm_b) == pytest.approx(v_target, rel=1e-9)
    assert dual.spin_rate_rad_s(rpm_t, rpm_b) == pytest.approx(0.0, abs=1e-9)


def test_inverse_with_spin():
    dual = make_dual(slip=0.0)
    v_target, spin_target = 18.0, 80.0  # rad/s backspin
    rpm_t, rpm_b = dual.rpms_for_exit_velocity(v_target, omega_spin=spin_target)
    assert dual.exit_velocity_ms(rpm_t, rpm_b) == pytest.approx(v_target, rel=1e-9)
    assert dual.spin_rate_rad_s(rpm_t, rpm_b) == pytest.approx(spin_target, rel=1e-9)


def test_slip_reduces_surface_speed():
    r_no_slip = Roller(diameter_m=0.1, slip_factor=0.0)
    r_slip = Roller(diameter_m=0.1, slip_factor=0.2)
    assert r_slip.surface_speed_ms(3000) == pytest.approx(
        r_no_slip.surface_speed_ms(3000) * 0.8, rel=1e-9,
    )
