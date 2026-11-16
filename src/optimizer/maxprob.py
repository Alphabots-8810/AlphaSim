"""V5 aim solver: pick the shot most likely to score under an explicit error budget.

For one distance there is a whole family of (hood angle, speed) pairs that land
the ball on the hub centre - one speed per angle. V5 walks that family and, for
each member, works out

  1. how far the landing point moves per unit of each error source
     (finite-difference sensitivities from the batch integrator),
  2. the resulting landing spread (sources add in quadrature),
  3. how much room the opening leaves at that entry angle, and
  4. the probability of a clean pass.

The answer is the member with the highest scoring probability - or, when
`margin_concession` > 0, the slowest member that keeps at least that fraction
of the best achievable sigma margin (trading a sliver of margin for flywheel
energy; see `solve_aim`).

Nothing here is a tuning weight: every input is a measurable physical quantity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.physics.batch import Aero, simulate_batch, speed_for_distance
from src.scoring import (ErrorBudget, HexOpening, Spread, landing_spread,
                         miss_probability, sigma_margin)


@dataclass(frozen=True)
class ShotModel:
    """Everything about the robot + field the ballistics need."""
    exit_height: float
    target_height: float
    aero: Aero
    opening: HexOpening
    theta_min: float = 25.0
    theta_max: float = 85.0
    v_max: float = 40.0

    @classmethod
    def from_configs(cls, game, robot, exit_height: Optional[float] = None,
                     hex_bearing_deg: float = 0.0,
                     drag_coefficient: Optional[float] = None,
                     lift_coefficient: Optional[float] = None) -> "ShotModel":
        """Build from the loaded YAML configs (src.config)."""
        from src.physics.shooter import Roller
        v_top = Roller(robot.flywheel.diameter_m, robot.flywheel.gear_ratio,
                       robot.flywheel.motor_free_rpm, robot.flywheel.slip_factor)
        v_bot = Roller(robot.hood_roller.diameter_m, robot.hood_roller.gear_ratio,
                       robot.hood_roller.motor_free_rpm, robot.hood_roller.slip_factor)
        v_max = min(v_top.surface_speed_ms(v_top.max_flywheel_rpm),
                    v_bot.surface_speed_ms(v_bot.max_flywheel_rpm))
        return cls(
            exit_height=robot.shooter.exit_height_m if exit_height is None else exit_height,
            target_height=game.hub.target_height_m,
            aero=Aero.from_ball(
                game.ball.mass_kg, game.ball.radius_m,
                game.ball.drag_coefficient if drag_coefficient is None else drag_coefficient,
                game.ball.magnus_cl if lift_coefficient is None else lift_coefficient,
                game.environment.air_density_kg_m3, game.environment.gravity_m_s2),
            opening=HexOpening(game.hub.target_across_flats_m / 2.0, game.ball.radius_m,
                               hex_bearing_deg),
            theta_min=robot.shooter.hood_angle_min_deg,
            theta_max=robot.shooter.hood_angle_max_deg,
            v_max=v_max,
        )


@dataclass
class ShotFamily:
    """Every centred-landing shot for a set of distances.

    Arrays are shaped (n_distance, n_theta); entries are NaN where no
    descending shot exists at that angle.
    """
    distance: np.ndarray        # (n_distance, 1)
    theta: np.ndarray           # (1, n_theta)
    speed: np.ndarray
    tof: np.ndarray
    entry_angle: np.ndarray
    entry_speed: np.ndarray
    apex: np.ndarray
    dx_dlnv: np.ndarray         # m per unit fractional speed change
    dx_dtheta: np.ndarray       # m per degree
    half_width: np.ndarray      # centre-line usable half-width along the shot, m
    spread: Spread
    miss: np.ndarray
    margin: np.ndarray          # equivalent sigma margin


def centered_family(model: ShotModel, budget: ErrorBudget, distance, theta_deg,
                    dt: float = 0.01) -> ShotFamily:
    """Evaluate the centred-landing shot at every (distance, angle) pair."""
    d = np.atleast_1d(np.asarray(distance, dtype=float))[:, None]
    th = np.atleast_1d(np.asarray(theta_deg, dtype=float))[None, :]
    h0, ht, aero = model.exit_height, model.target_height, model.aero

    v = speed_for_distance(d, th, h0, ht, aero, v_max=model.v_max, dt=dt)
    nominal = simulate_batch(v, th, h0, ht, aero, dt=dt)

    # Central differences. Steps are small enough to be linear, large enough
    # to sit far above the integrator's ~1e-10 m noise floor.
    ev, et = 2e-3, 0.05
    x_vp = simulate_batch(v * (1 + ev), th, h0, ht, aero, dt=dt).x
    x_vm = simulate_batch(v * (1 - ev), th, h0, ht, aero, dt=dt).x
    x_tp = simulate_batch(v, th + et, h0, ht, aero, dt=dt).x
    x_tm = simulate_batch(v, th - et, h0, ht, aero, dt=dt).x
    dx_dlnv = (x_vp - x_vm) / (2 * ev)
    dx_dtheta = (x_tp - x_tm) / (2 * et)

    entry = nominal.entry_angle_deg
    ok = np.isfinite(v) & np.isfinite(dx_dlnv) & np.isfinite(dx_dtheta)
    safe = lambda a, fill: np.where(ok, a, fill)   # noqa: E731
    spread = landing_spread(budget, safe(dx_dlnv, 0.0), safe(dx_dtheta, 0.0),
                            safe(nominal.t, 0.0), d)
    miss = miss_probability(model.opening, budget, spread, safe(entry, 90.0))
    miss = np.where(ok, miss, np.nan)
    nan = lambda a: np.where(ok, a, np.nan)        # noqa: E731
    return ShotFamily(
        distance=d, theta=th, speed=nan(v), tof=nan(nominal.t),
        entry_angle=nan(entry), entry_speed=nan(nominal.entry_speed),
        apex=nan(nominal.apex_y), dx_dlnv=nan(dx_dlnv), dx_dtheta=nan(dx_dtheta),
        half_width=nan(model.opening.half_width(safe(entry, 90.0))),
        spread=spread, miss=miss, margin=np.where(ok, sigma_margin(safe(miss, 1.0)), np.nan),
    )


@dataclass
class AimSolution:
    success: bool
    distance_m: float
    hood_angle_deg: float = float("nan")
    v_exit_ms: float = float("nan")
    time_of_flight_s: float = float("nan")
    entry_angle_deg: float = float("nan")
    entry_speed_ms: float = float("nan")
    apex_height_m: float = float("nan")
    p_score: float = float("nan")
    sigma_margin: float = float("nan")
    half_width_m: float = float("nan")
    # landing spread (1 sigma, m) by source
    spread_speed_m: float = float("nan")
    spread_angle_m: float = float("nan")
    spread_vortex_m: float = float("nan")
    spread_range_m: float = float("nan")
    spread_along_m: float = float("nan")
    spread_lateral_m: float = float("nan")
    dx_dlnv_m: float = float("nan")
    dx_dtheta_m_per_deg: float = float("nan")
    # context: where the unconstrained optima sit
    best_angle_deg: float = float("nan")        # highest scoring probability
    best_sigma_margin: float = float("nan")
    min_speed_angle_deg: float = float("nan")   # lowest launch speed
    min_speed_ms: float = float("nan")
    note: str = ""


def _parabolic_peak(x, y, i):
    """Refine the location/value of an interior extremum by a 3-point parabola."""
    if i <= 0 or i >= len(x) - 1 or not np.all(np.isfinite(y[i - 1:i + 2])):
        return float(x[i]), float(y[i])
    x0, x1, x2 = x[i - 1:i + 2]
    y0, y1, y2 = y[i - 1:i + 2]
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom
    if a == 0:
        return float(x1), float(y1)
    xv = -b / (2 * a)
    if not (x0 <= xv <= x2):
        return float(x1), float(y1)
    c = y1 - a * x1 * x1 - b * x1
    return float(xv), float(a * xv * xv + b * xv + c)


def choose_angle(theta, speed, margin, margin_concession: float = 0.0):
    """Pick the launch angle from one distance's family.

    Returns (theta_pick, theta_best, margin_best, theta_min_speed, min_speed).

    With margin_concession = 0 the pick is the angle of highest sigma margin.
    Otherwise it is the lowest-speed angle whose margin is still at least
    (1 - margin_concession) of the best. Because that condition is a ratio of
    margins, the pick barely moves if every noise source turns out to be, say,
    twice what the budget assumed (in the vertical plane it does not move at
    all) - so a wrong guess at the overall noise level does not misplace it.
    """
    ok = np.isfinite(margin) & np.isfinite(speed)
    if not ok.any():
        return None
    idx = np.flatnonzero(ok)
    th, v, m = theta[idx], speed[idx], margin[idx]
    i_best = int(np.argmax(m))
    th_best, m_best = _parabolic_peak(th, m, i_best)
    i_slow = int(np.argmin(v))
    th_slow, v_slow = _parabolic_peak(th, v, i_slow)
    if margin_concession <= 0 or not np.isfinite(m_best):
        return th_best, th_best, m_best, th_slow, v_slow

    # Walk outward from the best angle to the edges of the acceptable band.
    floor = (1.0 - margin_concession) * m_best

    def edge(step):
        j = i_best
        while 0 <= j + step < len(th) and m[j + step] >= floor:
            j += step
        k = j + step
        if not (0 <= k < len(th)) or m[j] == m[k]:
            return float(th[j])
        return float(th[j] + (floor - m[j]) * (th[k] - th[j]) / (m[k] - m[j]))

    lo, hi = edge(-1), edge(+1)
    return float(np.clip(th_slow, lo, hi)), th_best, m_best, th_slow, v_slow


def solve_aim(model: ShotModel, budget: ErrorBudget, distance: float,
              margin_concession: float = 0.03, theta_step: float = 0.25) -> AimSolution:
    """Best (hood angle, exit speed) for one distance. See module docstring."""
    return sweep_aim(model, budget, [distance], margin_concession, theta_step)[0]


def sweep_aim(model: ShotModel, budget: ErrorBudget, distances,
              margin_concession: float = 0.03, theta_step: float = 0.25) -> list[AimSolution]:
    """Solve many distances at once (one vectorised pass over the whole grid)."""
    ds = np.atleast_1d(np.asarray(distances, dtype=float))
    thetas = np.arange(model.theta_min, model.theta_max + 1e-9, theta_step)
    fam = centered_family(model, budget, ds, thetas)

    picks = [choose_angle(thetas, fam.speed[i], fam.margin[i], margin_concession)
             for i in range(len(ds))]
    out: list[Optional[AimSolution]] = [None] * len(ds)
    live = [i for i, p in enumerate(picks) if p is not None]
    for i, p in enumerate(picks):
        if p is None:
            out[i] = AimSolution(
                success=False, distance_m=float(ds[i]),
                note=(f"no hood angle in [{model.theta_min:g}, {model.theta_max:g}] deg lands a "
                      f"descending ball at {ds[i]:.2f} m within {model.v_max:.1f} m/s"))
    if not live:
        return out  # type: ignore[return-value]

    # Exact re-evaluation at the picked angles (one diagonal batch).
    th_pick = np.array([picks[i][0] for i in live])
    final = _evaluate_points(model, budget, ds[live], th_pick)
    for k, i in enumerate(live):
        _, th_best, m_best, th_slow, v_slow = picks[i]
        out[i] = AimSolution(
            success=True, distance_m=float(ds[i]),
            best_angle_deg=th_best, best_sigma_margin=m_best,
            min_speed_angle_deg=th_slow, min_speed_ms=v_slow,
            **{key: float(val[k]) for key, val in final.items()},
        )
    return out  # type: ignore[return-value]


def aim_at_angle(model: ShotModel, budget: ErrorBudget, distance: float,
                 hood_angle_deg: float) -> AimSolution:
    """The centred-landing shot at a hood angle you choose, fully graded.
    Lets any other rule for picking the angle (an old solver, a hand-tuned
    table) be scored on exactly the same terms as V5."""
    final = _evaluate_points(model, budget, [distance], [hood_angle_deg])
    if not np.isfinite(final["v_exit_ms"][0]):
        return AimSolution(success=False, distance_m=float(distance),
                           note=f"no descending shot at {hood_angle_deg:.2f} deg")
    return AimSolution(success=True, distance_m=float(distance),
                       **{key: float(val[0]) for key, val in final.items()})


def _evaluate_points(model: ShotModel, budget: ErrorBudget, distances, thetas) -> dict:
    """Full diagnostics for paired (distance_i, theta_i) points."""
    rows = [centered_family(model, budget, [d], [t]) for d, t in zip(distances, thetas)]
    get = lambda f: np.array([f(r).ravel()[0] for r in rows])     # noqa: E731
    miss = get(lambda r: r.miss)
    return dict(
        hood_angle_deg=np.asarray(thetas, dtype=float),
        v_exit_ms=get(lambda r: r.speed),
        time_of_flight_s=get(lambda r: r.tof),
        entry_angle_deg=get(lambda r: r.entry_angle),
        entry_speed_ms=get(lambda r: r.entry_speed),
        apex_height_m=get(lambda r: r.apex),
        p_score=1.0 - miss,
        sigma_margin=get(lambda r: r.margin),
        half_width_m=get(lambda r: r.half_width),
        spread_speed_m=get(lambda r: r.spread.speed),
        spread_angle_m=get(lambda r: r.spread.angle),
        spread_vortex_m=get(lambda r: r.spread.vortex),
        spread_range_m=get(lambda r: r.spread.range),
        spread_along_m=get(lambda r: r.spread.along),
        spread_lateral_m=get(lambda r: r.spread.lateral),
        dx_dlnv_m=get(lambda r: r.dx_dlnv),
        dx_dtheta_m_per_deg=get(lambda r: r.dx_dtheta),
    )


def evaluate_shot(model: ShotModel, budget: ErrorBudget, distance: float,
                  hood_angle_deg: float, v_exit_ms: float) -> dict:
    """Score an arbitrary (angle, speed) - it need not land centred.

    Used to grade lookup tables and polynomial fits end to end: the returned
    `offset_m` is where the nominal ball lands relative to the hub centre, and
    `p_score` accounts for that bias.
    """
    h0, ht, aero = model.exit_height, model.target_height, model.aero
    v = np.array([v_exit_ms, v_exit_ms * (1 + 2e-3), v_exit_ms * (1 - 2e-3),
                  v_exit_ms, v_exit_ms])
    th = np.array([hood_angle_deg, hood_angle_deg, hood_angle_deg,
                   hood_angle_deg + 0.05, hood_angle_deg - 0.05])
    r = simulate_batch(v, th, h0, ht, aero)
    if not r.hit[0]:
        return dict(hit=False, offset_m=float("nan"), p_score=0.0, sigma_margin=0.0)
    dx_dlnv = (r.x[1] - r.x[2]) / 4e-3
    dx_dth = (r.x[3] - r.x[4]) / 0.1
    spread = landing_spread(budget, np.array(dx_dlnv), np.array(dx_dth),
                            np.array(r.t[0]), np.array(distance))
    entry = float(r.entry_angle_deg[0])
    offset = float(r.x[0] - distance)
    miss = float(miss_probability(model.opening, budget, spread, entry, offset))
    return dict(
        hit=True, offset_m=offset, p_score=1.0 - miss,
        sigma_margin=float(sigma_margin(miss)),
        time_of_flight_s=float(r.t[0]), entry_angle_deg=entry,
        apex_height_m=float(r.apex_y[0]),
        spread_along_m=float(spread.along), spread_lateral_m=float(spread.lateral),
    )
