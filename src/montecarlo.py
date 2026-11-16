"""Monte Carlo ground truth for the analytic scoring probability.

`src.scoring` linearises the trajectory around the aim point and assumes the
landing scatter is normal. This module makes neither assumption: it draws
every error source, flies each ball through the full nonlinear integrator,
and applies the exact rim-clearance test. Agreement between the two is what
justifies using the fast analytic model inside the optimizer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.physics.batch import simulate_batch
from src.scoring import ErrorBudget


@dataclass
class BurstResult:
    n: int
    p_score: float
    stderr: float           # binomial standard error of p_score
    no_crossing: int        # balls that never came down through the opening plane
    along: np.ndarray       # landing position relative to hub centre, along the shot (m)
    lateral: np.ndarray     # ... across the shot (m)
    entry_angle: np.ndarray
    tof: np.ndarray
    scored: np.ndarray      # bool per ball

    @property
    def along_std(self) -> float:
        return float(np.nanstd(self.along))

    @property
    def along_mean(self) -> float:
        return float(np.nanmean(self.along))

    @property
    def lateral_std(self) -> float:
        return float(np.nanstd(self.lateral))


def simulate_burst(model, budget: ErrorBudget, distance: float, hood_angle_deg: float,
                   v_exit_ms: float, n: int = 100_000, seed: int = 0,
                   distribution: str = "normal") -> BurstResult:
    """Fire `n` independent balls at one aim point.

    distribution="normal" draws speed/angle errors from a normal law;
    "uniform" draws them from a uniform law of the same variance (the bounded
    +/-sqrt(3) sigma box used by the original 100-ball burst spec).
    """
    rng = np.random.default_rng(seed)

    def draw(size):
        if distribution == "uniform":
            return rng.uniform(-math.sqrt(3.0), math.sqrt(3.0), size)
        if distribution == "normal":
            return rng.standard_normal(size)
        raise ValueError(f"unknown distribution {distribution!r}")

    v = v_exit_ms * (1.0 + budget.speed_sigma * draw(n))
    th = hood_angle_deg + budget.angle_sigma_deg * draw(n)
    r = simulate_batch(v, th, model.exit_height, model.target_height, model.aero)

    yaw = math.radians(budget.yaw_sigma_deg) * rng.standard_normal(n)
    wander = budget.vortex_coeff * np.where(r.hit, r.t, 0.0) ** 2
    # A range error means the aim was solved for the wrong distance; seen from
    # the hub that is a rigid along-track shift of the landing point.
    along = (r.x * np.cos(yaw) - distance
             + budget.range_sigma_m * rng.standard_normal(n)
             + wander * rng.standard_normal(n))
    lateral = (r.x * np.sin(yaw)
               + budget.lateral_half_width_m * rng.uniform(-1.0, 1.0, n)
               + wander * rng.standard_normal(n))
    entry = r.entry_angle_deg
    scored = r.hit & model.opening.clears(
        np.where(r.hit, along, 0.0), np.where(r.hit, lateral, 0.0),
        np.where(r.hit, entry, 90.0))
    p = float(scored.mean())
    return BurstResult(
        n=n, p_score=p, stderr=math.sqrt(max(p * (1 - p), 1e-12) / n),
        no_crossing=int((~r.hit).sum()),
        along=np.where(r.hit, along, np.nan), lateral=np.where(r.hit, lateral, np.nan),
        entry_angle=np.where(r.hit, entry, np.nan), tof=r.t, scored=scored,
    )
