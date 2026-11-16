"""V5 solver: the shot it returns, and the rule it uses to pick it."""
import numpy as np
import pytest

from src.montecarlo import simulate_burst
from src.optimizer.maxprob import (aim_at_angle, centered_family, choose_angle,
                                   evaluate_shot, solve_aim, sweep_aim)
from src.physics.trajectory import simulate_trajectory
from src.scoring import ErrorBudget

DISTANCES = [1.5, 2.5, 3.5, 5.0]


@pytest.fixture(scope="module")
def sweep(model, budget):
    return sweep_aim(model, budget, DISTANCES, margin_concession=0.03)


def test_every_distance_solves_inside_the_hood_range(sweep, model):
    for s in sweep:
        assert s.success
        assert model.theta_min <= s.hood_angle_deg <= model.theta_max
        assert 0.5 < s.p_score <= 1.0
        assert s.sigma_margin > 1.5


def test_solution_lands_on_centre_per_the_independent_integrator(sweep, model):
    """Cross-check against scipy RK45, which shares no code with the batch path."""
    for s in sweep:
        ref = simulate_trajectory(s.v_exit_ms, s.hood_angle_deg, model.exit_height,
                                  model.target_height, 0.215, 0.075, 0.5,
                                  rtol=1e-10, atol=1e-12)
        assert ref.hit
        assert ref.hit_x == pytest.approx(s.distance_m, abs=1e-5)
        assert ref.hit_t == pytest.approx(s.time_of_flight_s, abs=1e-5)
        assert ref.entry_angle_deg == pytest.approx(s.entry_angle_deg, abs=1e-3)


def test_farther_needs_more_speed_and_a_flatter_hood(sweep):
    v = [s.v_exit_ms for s in sweep]
    hood = [s.hood_angle_deg for s in sweep]
    p = [s.p_score for s in sweep]
    assert v == sorted(v)
    assert hood == sorted(hood, reverse=True)
    assert p == sorted(p, reverse=True)          # closer is always easier


def test_concession_trades_margin_for_speed(model, budget):
    best = solve_aim(model, budget, 4.0, margin_concession=0.0)
    lean = solve_aim(model, budget, 4.0, margin_concession=0.03)
    assert best.hood_angle_deg == pytest.approx(best.best_angle_deg, abs=1e-9)
    assert lean.v_exit_ms < best.v_exit_ms
    assert lean.sigma_margin <= best.sigma_margin + 1e-9
    assert lean.sigma_margin >= 0.97 * best.sigma_margin - 5e-3
    # The pick sits between the lowest-speed angle and the best angle.
    assert lean.min_speed_angle_deg <= lean.hood_angle_deg <= lean.best_angle_deg


def test_no_other_angle_on_the_grid_beats_the_best(model, budget):
    best = solve_aim(model, budget, 3.0, margin_concession=0.0)
    thetas = np.arange(40.0, 85.0, 0.5)
    fam = centered_family(model, budget, [3.0], thetas)
    assert np.nanmax(fam.margin) <= best.sigma_margin + 2e-3


def test_pick_is_insensitive_to_overall_noise_level(model, budget):
    """The concession rule compares margins as a ratio. In the vertical plane
    that makes the pick exactly independent of how big the noise is; with the
    lateral terms switched on it still barely moves."""
    planar = ErrorBudget.planar()
    a = solve_aim(model, planar, 3.5, margin_concession=0.03)
    b = solve_aim(model, planar.scaled(2.0), 3.5, margin_concession=0.03)
    assert a.hood_angle_deg == pytest.approx(b.hood_angle_deg, abs=1e-6)
    assert b.sigma_margin == pytest.approx(a.sigma_margin / 2.0, rel=1e-6)
    c = solve_aim(model, budget, 3.5, margin_concession=0.03)
    d = solve_aim(model, budget.scaled(2.0), 3.5, margin_concession=0.03)
    assert c.hood_angle_deg == pytest.approx(d.hood_angle_deg, abs=1.0)


def test_speed_scatter_dominates_the_budget(sweep):
    """The finding the report leans on: speed error, not angle error, sets the spread."""
    for s in sweep:
        assert s.spread_speed_m > 2.0 * s.spread_angle_m
        assert s.spread_speed_m > s.spread_vortex_m


def test_steeper_shots_are_less_sensitive_to_speed(model, budget):
    fam = centered_family(model, budget, [3.0], [50.0, 60.0, 70.0])
    sens = fam.dx_dlnv[0]
    assert sens[0] > sens[1] > sens[2] > 0
    # ... and angle sensitivity changes sign through the lowest-speed angle.
    assert fam.dx_dtheta[0, 0] > 0 > fam.dx_dtheta[0, 2]


def test_unreachable_distance_fails_cleanly(model, budget):
    s = solve_aim(model, budget, 100.0)
    assert not s.success and "no hood angle" in s.note


def test_choose_angle_on_synthetic_curves():
    th = np.linspace(40, 80, 81)
    margin = 3.0 - ((th - 66.0) / 20.0) ** 2          # peak at 66
    speed = 7.0 + ((th - 56.0) / 30.0) ** 2           # slowest at 56
    pick, best, m_best, slow, v_slow = choose_angle(th, speed, margin, 0.0)
    assert pick == pytest.approx(66.0, abs=1e-6) and best == pytest.approx(66.0, abs=1e-6)
    assert m_best == pytest.approx(3.0, abs=1e-9)
    assert slow == pytest.approx(56.0, abs=1e-6) and v_slow == pytest.approx(7.0, abs=1e-9)
    pick, *_ = choose_angle(th, speed, margin, 0.03)   # margin >= 2.91 -> |th-66| <= 6
    assert pick == pytest.approx(60.0, abs=0.02)
    pick, *_ = choose_angle(th, speed, margin, 0.50)   # band reaches past the slowest angle
    assert pick == pytest.approx(56.0, abs=1e-6)
    assert choose_angle(th, np.full_like(th, np.nan), margin, 0.03) is None


def test_analytic_probability_matches_monte_carlo(model, budget):
    """The optimizer's fast model against the nonlinear, non-normal ground truth."""
    for d in (3.0, 5.0):
        s = solve_aim(model, budget, d)
        mc = simulate_burst(model, budget, d, s.hood_angle_deg, s.v_exit_ms, n=300_000, seed=5)
        miss_model, miss_mc = 1 - s.p_score, 1 - mc.p_score
        assert miss_model == pytest.approx(miss_mc, rel=0.15, abs=5 * mc.stderr)
        assert mc.along_std == pytest.approx(s.spread_along_m, rel=0.02)
        assert abs(mc.along_mean) < 0.01


def test_evaluate_shot_sees_an_off_centre_aim(model, budget):
    s = solve_aim(model, budget, 3.0)
    on = evaluate_shot(model, budget, 3.0, s.hood_angle_deg, s.v_exit_ms)
    hot = evaluate_shot(model, budget, 3.0, s.hood_angle_deg, s.v_exit_ms * 1.03)
    assert abs(on["offset_m"]) < 1e-6
    assert on["p_score"] == pytest.approx(s.p_score, abs=1e-9)
    assert hot["offset_m"] > 0.15 and hot["p_score"] < on["p_score"]
    assert not evaluate_shot(model, budget, 3.0, s.hood_angle_deg, 1.0)["hit"]


def test_aim_at_angle_grades_any_curve_on_the_same_terms(model, budget):
    s = solve_aim(model, budget, 3.0)
    again = aim_at_angle(model, budget, 3.0, s.hood_angle_deg)
    assert again.v_exit_ms == pytest.approx(s.v_exit_ms, abs=1e-9)
    assert again.p_score == pytest.approx(s.p_score, abs=1e-12)
    assert not aim_at_angle(model, budget, 1.5, 30.0).success
