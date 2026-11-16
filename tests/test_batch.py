"""The vectorised integrator must reproduce the scipy reference and closed forms."""
import math

import numpy as np
import pytest

from src.physics.batch import Aero, no_drag_speed, simulate_batch, speed_for_distance
from src.physics.trajectory import simulate_trajectory

BALL = dict(mass_kg=0.215, radius_m=0.075)
AERO = Aero.from_ball(drag_coefficient=0.5, **BALL)


def _reference(v, th, h0, ht, cd=0.5, cl=0.0):
    kw = dict(rtol=1e-11, atol=1e-12, max_step=0.005)
    if cl:
        kw.update(spin_rate=1.0, magnus_cl_fn=lambda S: cl)
    return simulate_trajectory(v, th, h0, ht, BALL["mass_kg"], BALL["radius_m"], cd, **kw)


@pytest.mark.parametrize("cl", [0.0, 0.25])
def test_matches_scipy_reference(cl):
    rng = np.random.default_rng(7)
    v, th = rng.uniform(5.5, 12, 40), rng.uniform(30, 80, 40)
    aero = Aero.from_ball(drag_coefficient=0.5, lift_coefficient=cl, **BALL)
    got = simulate_batch(v, th, 0.5, 1.829, aero)
    compared = 0
    for i in range(v.size):
        ref = _reference(v[i], th[i], 0.5, 1.829, cl=cl)
        assert bool(got.hit[i]) == ref.hit
        if ref.hit:
            assert got.x[i] == pytest.approx(ref.hit_x, abs=1e-7)
            assert got.t[i] == pytest.approx(ref.hit_t, abs=1e-7)
            assert got.vx[i] == pytest.approx(ref.hit_vx, abs=1e-6)
            assert got.vy[i] == pytest.approx(ref.hit_vy, abs=1e-6)
            compared += 1
    assert compared > 20


def test_no_drag_matches_closed_form():
    vacuum = Aero(k_drag=0.0)
    v, th, h0, ht, g = 9.0, 60.0, 0.5, 1.829, 9.81
    r = simulate_batch(v, th, h0, ht, vacuum)
    vy0, vx0 = v * math.sin(math.radians(th)), v * math.cos(math.radians(th))
    t = (vy0 + math.sqrt(vy0 ** 2 - 2 * g * (ht - h0))) / g
    assert r.hit
    assert float(r.t) == pytest.approx(t, abs=1e-9)
    assert float(r.x) == pytest.approx(vx0 * t, abs=1e-9)
    assert float(r.apex_y) == pytest.approx(h0 + vy0 ** 2 / (2 * g), abs=1e-9)
    assert float(r.entry_angle_deg) == pytest.approx(
        math.degrees(math.atan2(g * t - vy0, vx0)), abs=1e-7)


def test_too_slow_never_crosses_and_reports_nan():
    r = simulate_batch([2.0, 9.0], [60.0, 60.0], 0.5, 1.829, AERO)
    assert list(r.hit) == [False, True]
    assert np.isnan(r.x[0]) and np.isfinite(r.x[1])


def test_broadcasting_shapes():
    r = simulate_batch(np.full((3, 1), 8.0), np.array([[50.0, 60.0, 70.0, 75.0]]),
                       0.5, 1.829, AERO)
    assert r.x.shape == (3, 4)
    assert np.allclose(r.x[0], r.x[1])


def test_speed_for_distance_lands_on_target():
    th = np.array([45.0, 55.0, 65.0, 75.0])
    v = speed_for_distance(3.0, th, 0.5, 1.829, AERO)
    assert np.all(np.isfinite(v))
    landed = simulate_batch(v, th, 0.5, 1.829, AERO)
    assert np.allclose(landed.x, 3.0, atol=1e-7)
    # And against the independent scipy integrator:
    assert _reference(float(v[1]), 55.0, 0.5, 1.829).hit_x == pytest.approx(3.0, abs=1e-6)


def test_speed_for_distance_rejects_impossible_shots():
    # 30 deg at 1.5 m: the ball could only pass the target on the way up.
    assert np.isnan(speed_for_distance(1.5, 30.0, 0.5, 1.829, AERO))
    # A straight line at 20 deg never gets as high as the opening by 1.5 m.
    assert np.isnan(no_drag_speed(1.5, 20.0, 0.5, 1.829))
    # Beyond the speed cap.
    assert np.isnan(speed_for_distance(60.0, 45.0, 0.5, 1.829, AERO, v_max=13.0))


def test_drag_needs_more_speed_than_vacuum():
    v_air = speed_for_distance(4.0, 55.0, 0.5, 1.829, AERO)
    assert v_air > no_drag_speed(4.0, 55.0, 0.5, 1.829)
