import math

import pytest

from src.geometry.target import HexTarget


def test_apothem_and_side_length():
    t = HexTarget(across_flats_m=1.060, height_m=1.829)
    assert t.apothem == pytest.approx(0.530, rel=1e-9)
    assert t.side_length == pytest.approx(1.060 / math.sqrt(3), rel=1e-9)


def test_contains_point_2d_within_apothem():
    t = HexTarget(across_flats_m=1.0, height_m=1.0)
    assert t.contains_point_2d(0.0)
    assert t.contains_point_2d(0.49)
    assert t.contains_point_2d(-0.49)
    assert not t.contains_point_2d(0.51)


def test_ball_radius_reduces_effective_window():
    t = HexTarget(across_flats_m=1.0, height_m=1.0)
    assert t.contains_ball_2d(0.35, ball_radius=0.075)
    assert not t.contains_ball_2d(0.45, ball_radius=0.075)


def test_ball_larger_than_window_never_fits():
    t = HexTarget(across_flats_m=0.1, height_m=1.0)
    assert not t.contains_ball_2d(0.0, ball_radius=0.075)


def test_3d_hex_containment_center_in():
    t = HexTarget(across_flats_m=1.0, height_m=1.0)
    assert t.contains_point_3d_flat_top(0.0, 0.0)


def test_3d_hex_containment_outside():
    t = HexTarget(across_flats_m=1.0, height_m=1.0)
    s = t.side_length
    # Just outside a vertex (vertex at (-s, 0))
    assert not t.contains_point_3d_flat_top(-s - 0.01, 0.0)
