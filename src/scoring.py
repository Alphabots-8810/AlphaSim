"""What counts as a score, and how likely a shot is to be one.

Two pieces:

* `HexOpening` - the geometry test. The hub opening is a regular hexagon; a
  ball scores if it passes through without touching the rim. A ball descending
  at entry angle phi sweeps a cylinder whose cut through the opening plane is
  an ellipse: ball radius r across the shot line, r / sin(phi) along it. So a
  shallow entry needs more room along the shot line than a steep one - this is
  where entry angle enters the physics, with no tuning weight attached.

* `ErrorBudget` + `score_probability` - the noise model. Each physical error
  source (flywheel speed, hood angle, heading, range estimate, wake turbulence,
  where across the dumper the ball started) contributes a landing-point spread;
  they add in quadrature, and the probability of landing inside the opening
  follows from the normal distribution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np
from scipy.special import ndtr, ndtri


@dataclass(frozen=True)
class HexOpening:
    """Regular hexagonal opening in a horizontal plane.

    apothem     centre-to-flat distance (half the across-flats width), m
    ball_radius m
    bearing_deg angle between the shot line and the nearest flat's normal.
                0 = a flat faces the robot (narrowest along the shot line, the
                conservative default); 30 = a vertex faces the robot.
    """
    apothem: float
    ball_radius: float
    bearing_deg: float = 0.0

    @property
    def side_length(self) -> float:
        return 2.0 * self.apothem / math.sqrt(3.0)

    def _normals(self):
        ang = np.radians(self.bearing_deg + 60.0 * np.arange(6))
        return np.cos(ang), np.sin(ang)

    def half_width(self, entry_angle_deg) -> np.ndarray:
        """Usable half-width along the shot line for a ball on the centre line
        (lateral offset 0) with a flat facing the robot: a - r / sin(phi)."""
        phi = np.radians(np.asarray(entry_angle_deg, dtype=float))
        with np.errstate(divide="ignore"):
            return self.apothem - self.ball_radius / np.sin(phi)

    def along_track_limits(self, lateral, entry_angle_deg):
        """Interval [lo, hi] of along-track ball-centre positions (relative to
        the hub centre) that clear every rim edge, for a ball at the given
        lateral offset. lo > hi means no clean passage at that offset."""
        z = np.asarray(lateral, dtype=float)
        phi = np.radians(np.asarray(entry_angle_deg, dtype=float))
        z, phi = np.broadcast_arrays(z, phi)
        with np.errstate(divide="ignore"):
            a_long = self.ball_radius / np.sin(phi)    # ellipse semi-axis along the shot
        lo = np.full(z.shape, -np.inf)
        hi = np.full(z.shape, np.inf)
        blocked = np.zeros(z.shape, dtype=bool)
        for nx, nz in zip(*self._normals()):
            reach = np.hypot(a_long * nx, self.ball_radius * nz)   # ellipse support
            room = self.apothem - nz * z - reach
            if abs(nx) < 1e-12:
                blocked |= room < 0
            elif nx > 0:
                hi = np.minimum(hi, room / nx)
            else:
                lo = np.maximum(lo, room / nx)
        lo = np.where(blocked, np.inf, lo)
        hi = np.where(blocked, -np.inf, hi)
        return lo, hi

    def clears(self, along, lateral, entry_angle_deg) -> np.ndarray:
        """True where a ball centred at (along, lateral) passes cleanly."""
        lo, hi = self.along_track_limits(lateral, entry_angle_deg)
        along = np.asarray(along, dtype=float)
        return (along >= lo) & (along <= hi)


@dataclass(frozen=True)
class ErrorBudget:
    """One-sigma size of every modelled error source.

    speed_sigma           shot-to-shot ball speed scatter, fraction of speed
    angle_sigma_deg       shot-to-shot launch angle scatter
    yaw_sigma_deg         heading error of the shot line
    range_sigma_m         error in the distance fed to the aim function
    vortex_coeff          wake-turbulence wander: sigma = coeff * T^2 metres per
                          horizontal axis, T = time of flight (s)
    lateral_half_width_m  balls leave the dumper spread uniformly across
                          +/- this much (centre-to-centre) from the shot line
    """
    speed_sigma: float = 0.017
    angle_sigma_deg: float = 0.6
    yaw_sigma_deg: float = 1.0
    range_sigma_m: float = 0.03
    vortex_coeff: float = 0.04
    lateral_half_width_m: float = 0.15

    @classmethod
    def planar(cls, speed_sigma: float = 0.03 / math.sqrt(3.0),
               angle_sigma_deg: float = 1.0 / math.sqrt(3.0),
               vortex_coeff: float = 0.0) -> "ErrorBudget":
        """Speed and angle scatter only, everything in the vertical plane.
        Defaults match the variance of the original burst spec (+/-3 % speed,
        +/-1 deg angle, uniform)."""
        return cls(speed_sigma=speed_sigma, angle_sigma_deg=angle_sigma_deg,
                   yaw_sigma_deg=0.0, range_sigma_m=0.0, vortex_coeff=vortex_coeff,
                   lateral_half_width_m=0.0)

    def scaled(self, factor: float) -> "ErrorBudget":
        """Every random source multiplied by `factor` (the dumper width, being
        geometry rather than noise, is left alone)."""
        kw = {f.name: getattr(self, f.name) * factor for f in fields(self)}
        kw["lateral_half_width_m"] = self.lateral_half_width_m
        return ErrorBudget(**kw)


@dataclass
class Spread:
    """Landing-point standard deviations (m) at the opening, by source."""
    speed: np.ndarray
    angle: np.ndarray
    vortex: np.ndarray
    range: np.ndarray
    yaw: np.ndarray          # lateral
    along: np.ndarray        # total along the shot line
    lateral: np.ndarray      # total Gaussian part across it (excludes dumper width)


def landing_spread(budget: ErrorBudget, dx_dlnv, dx_dtheta_deg, tof, distance) -> Spread:
    """Combine the budget with the shot's sensitivities.

    dx_dlnv        metres of landing shift per unit fractional speed change
    dx_dtheta_deg  metres of landing shift per degree of launch angle
    """
    tof = np.asarray(tof, dtype=float)
    speed = np.abs(dx_dlnv) * budget.speed_sigma
    angle = np.abs(dx_dtheta_deg) * budget.angle_sigma_deg
    vortex = budget.vortex_coeff * tof ** 2
    rng = np.full(tof.shape, budget.range_sigma_m)
    yaw = np.asarray(distance, dtype=float) * math.radians(budget.yaw_sigma_deg) * np.ones(tof.shape)
    return Spread(
        speed=speed, angle=angle, vortex=vortex, range=rng, yaw=yaw,
        along=np.sqrt(speed ** 2 + angle ** 2 + vortex ** 2 + rng ** 2),
        lateral=np.sqrt(yaw ** 2 + vortex ** 2),
    )


# Lateral quadrature: uniform dumper offset (Gauss-Legendre) x Gaussian wander
# (Gauss-Hermite). Fixed node counts keep the result a smooth function of the
# inputs, which the optimizer and the polynomial fit both rely on.
_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(9)
_GH_W = _GH_W / _GH_W.sum()
_GL_W = _GL_W / _GL_W.sum()


def miss_probability(opening: HexOpening, budget: ErrorBudget, spread: Spread,
                     entry_angle_deg, along_bias=0.0) -> np.ndarray:
    """Probability that one ball does NOT pass cleanly through the opening.

    Along the shot line the landing point is normal(along_bias, spread.along);
    across it, a uniform dumper offset plus normal(0, spread.lateral). The
    hexagon couples the two, so the lateral part is integrated numerically and
    the along-track part in closed form. Working with the miss (the two normal
    tails) rather than the hit keeps full precision when misses are rare.
    """
    phi = np.asarray(entry_angle_deg, dtype=float)
    shape = np.broadcast(phi, spread.along, spread.lateral, along_bias).shape
    phi = np.broadcast_to(phi, shape)[..., None, None]
    s_along = np.broadcast_to(spread.along, shape)[..., None, None]
    s_lat = np.broadcast_to(spread.lateral, shape)[..., None, None]
    bias = np.broadcast_to(np.asarray(along_bias, dtype=float), shape)[..., None, None]

    z = (budget.lateral_half_width_m * _GL_X[:, None]
         + s_lat * _GH_X[None, :])
    w = _GL_W[:, None] * _GH_W[None, :]
    lo, hi = opening.along_track_limits(z, phi)
    with np.errstate(divide="ignore", invalid="ignore"):
        tails = np.where(
            s_along > 0,
            ndtr((lo - bias) / s_along) + ndtr((bias - hi) / s_along),
            1.0 - ((lo <= bias) & (bias <= hi)),
        )
    tails = np.where(hi > lo, np.clip(tails, 0.0, 1.0), 1.0)
    return (tails * w).sum(axis=(-2, -1))


def score_probability(opening: HexOpening, budget: ErrorBudget, spread: Spread,
                      entry_angle_deg, along_bias=0.0) -> np.ndarray:
    """Probability that one ball passes cleanly through the opening."""
    return 1.0 - miss_probability(opening, budget, spread, entry_angle_deg, along_bias)


def sigma_margin(miss) -> np.ndarray:
    """Express a miss probability as the equivalent two-sided normal margin:
    the z for which P(|N(0,1)| > z) equals the miss. 2 sigma = 4.6 % missed,
    3 sigma = 0.27 %. Unlike the hit rate it does not saturate, so it stays a
    usable figure of merit when nearly everything scores."""
    m = np.clip(np.asarray(miss, dtype=float), 1e-300, 1.0)
    return -ndtri(0.5 * m)
