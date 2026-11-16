"""Opening geometry and the analytic scoring probability."""
import math

import numpy as np
import pytest
from scipy.special import ndtr

from src.geometry.target import HexTarget
from src.scoring import (ErrorBudget, HexOpening, Spread, landing_spread,
                         miss_probability, score_probability, sigma_margin)

A, R = 0.53, 0.075


def _spread(along, lateral=0.0):
    z = np.zeros(())
    return Spread(speed=z, angle=z, vortex=z, range=z, yaw=z,
                  along=np.asarray(float(along)), lateral=np.asarray(float(lateral)))


def test_centre_line_window_is_apothem_minus_ball_shadow():
    hexo = HexOpening(A, R)
    for phi in (30.0, 47.0, 90.0):
        lo, hi = hexo.along_track_limits(0.0, phi)
        expect = A - R / math.sin(math.radians(phi))
        assert hi == pytest.approx(expect) and lo == pytest.approx(-expect)
        assert hexo.half_width(phi) == pytest.approx(expect)


def test_shallow_entry_needs_more_room():
    hexo = HexOpening(A, R)
    assert hexo.half_width(35.0) < hexo.half_width(60.0) < hexo.half_width(90.0)
    assert hexo.half_width(90.0) == pytest.approx(A - R)


def test_vertex_on_is_wider_along_the_shot_line():
    flat, vertex = HexOpening(A, R, 0.0), HexOpening(A, R, 30.0)
    assert vertex.along_track_limits(0.0, 50.0)[1] > flat.along_track_limits(0.0, 50.0)[1]


def test_point_ball_agrees_with_hexagon_containment():
    """With a vanishing ball the clearance test is plain point-in-hexagon."""
    hexo, ref = HexOpening(A, 1e-9), HexTarget(across_flats_m=2 * A, height_m=1.0)
    rng = np.random.default_rng(3)
    pts = rng.uniform(-0.7, 0.7, size=(4000, 2))
    got = hexo.clears(pts[:, 0], pts[:, 1], 90.0)
    want = np.array([ref.contains_point_3d_flat_top(x, z) for x, z in pts])
    assert np.array_equal(got, want)


def test_far_lateral_offset_is_blocked():
    hexo = HexOpening(A, R)
    lo, hi = hexo.along_track_limits(0.7, 60.0)      # beyond the vertex
    assert lo > hi
    assert not hexo.clears(0.0, 0.7, 60.0)
    assert hexo.clears(0.0, 0.0, 60.0)


def test_planar_probability_is_the_two_normal_tails():
    hexo, sigma, phi = HexOpening(A, R), 0.15, 50.0
    w = float(hexo.half_width(phi))
    miss = float(miss_probability(hexo, ErrorBudget.planar(), _spread(sigma), phi))
    assert miss == pytest.approx(2 * ndtr(-w / sigma), rel=1e-12)
    assert float(sigma_margin(miss)) == pytest.approx(w / sigma, rel=1e-9)
    assert float(score_probability(hexo, ErrorBudget.planar(), _spread(sigma), phi)) == \
        pytest.approx(1 - miss)


def test_bias_shifts_probability_and_no_noise_is_exact():
    hexo, phi = HexOpening(A, R), 55.0
    bud = ErrorBudget.planar()
    centred = float(miss_probability(hexo, bud, _spread(0.15), phi))
    biased = float(miss_probability(hexo, bud, _spread(0.15), phi, along_bias=0.2))
    assert biased > centred
    assert float(miss_probability(hexo, bud, _spread(0.0), phi)) == 0.0
    assert float(miss_probability(hexo, bud, _spread(0.0), phi, along_bias=0.6)) == pytest.approx(1.0)


def test_lateral_error_can_only_hurt():
    hexo, phi = HexOpening(A, R), 50.0
    narrow = ErrorBudget.planar()
    wide = ErrorBudget(lateral_half_width_m=0.25)
    assert (float(miss_probability(hexo, wide, _spread(0.15, 0.15), phi))
            > float(miss_probability(hexo, narrow, _spread(0.15, 0.0), phi)))


def test_lateral_quadrature_against_brute_force():
    hexo, phi, bud = HexOpening(A, R), 48.0, ErrorBudget(lateral_half_width_m=0.2)
    s_along, s_lat = 0.18, 0.12
    rng = np.random.default_rng(11)
    n = 400_000
    x = s_along * rng.standard_normal(n)
    z = bud.lateral_half_width_m * rng.uniform(-1, 1, n) + s_lat * rng.standard_normal(n)
    brute = 1.0 - hexo.clears(x, z, phi).mean()
    model = float(miss_probability(hexo, bud, _spread(s_along, s_lat), phi))
    assert model == pytest.approx(brute, abs=4 * math.sqrt(brute * (1 - brute) / n) + 2e-4)


def test_spread_adds_in_quadrature_and_scales():
    bud = ErrorBudget()
    s = landing_spread(bud, np.array(8.0), np.array(-0.05), np.array(1.0), np.array(3.0))
    assert float(s.speed) == pytest.approx(8.0 * bud.speed_sigma)
    assert float(s.angle) == pytest.approx(0.05 * bud.angle_sigma_deg)
    assert float(s.along) == pytest.approx(math.sqrt(
        float(s.speed) ** 2 + float(s.angle) ** 2 + bud.vortex_coeff ** 2 + bud.range_sigma_m ** 2))
    doubled = landing_spread(bud.scaled(2.0), np.array(8.0), np.array(-0.05),
                             np.array(1.0), np.array(3.0))
    assert float(doubled.along) == pytest.approx(2 * float(s.along))
    assert bud.scaled(2.0).lateral_half_width_m == bud.lateral_half_width_m


def test_sigma_margin_known_values():
    assert float(sigma_margin(0.0455002638964)) == pytest.approx(2.0, abs=1e-6)
    assert float(sigma_margin(0.0026997960633)) == pytest.approx(3.0, abs=1e-6)
    assert np.isfinite(sigma_margin(0.0))
