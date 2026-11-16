"""Monte Carlo burst: reproducibility and the two noise laws."""
import numpy as np
import pytest

from src.montecarlo import simulate_burst
from src.optimizer.maxprob import solve_aim
from src.scoring import ErrorBudget


def test_same_seed_same_burst(model, budget):
    a = simulate_burst(model, budget, 3.0, 62.5, 7.13, n=2000, seed=9)
    b = simulate_burst(model, budget, 3.0, 62.5, 7.13, n=2000, seed=9)
    c = simulate_burst(model, budget, 3.0, 62.5, 7.13, n=2000, seed=10)
    assert np.array_equal(a.along, b.along) and a.p_score == b.p_score
    assert not np.array_equal(a.along, c.along)


def test_bounded_uniform_noise_never_misses_but_normal_noise_does(model):
    """Why the old '100 % at every distance' figure was an artefact: it drew
    speed and angle errors from a bounded box. Same variance, normal law, and
    the tails reach the rim."""
    spec = ErrorBudget.planar()                 # +/-3 % speed, +/-1 deg, as variance
    s = solve_aim(model, spec, 5.0)
    box = simulate_burst(model, spec, 5.0, s.hood_angle_deg, s.v_exit_ms,
                         n=50_000, seed=1, distribution="uniform")
    bell = simulate_burst(model, spec, 5.0, s.hood_angle_deg, s.v_exit_ms, n=50_000, seed=1)
    assert box.p_score == 1.0
    assert 0.95 < bell.p_score < 0.985
    assert box.along_std == pytest.approx(bell.along_std, rel=0.03)


def test_wildly_slow_ball_counts_as_a_miss(model, budget):
    r = simulate_burst(model, budget, 3.0, 62.5, 3.0, n=500)
    assert r.p_score == 0.0 and r.no_crossing == 500


def test_unknown_distribution_is_rejected(model, budget):
    with pytest.raises(ValueError):
        simulate_burst(model, budget, 3.0, 62.5, 7.1, n=10, distribution="triangular")
