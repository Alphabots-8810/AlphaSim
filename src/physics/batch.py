"""Vectorized ballistic integrator: thousands of trajectories per call.

`simulate_trajectory` (scipy RK45, one shot per call) stays the readable
reference. This module is the workhorse: a fixed-step RK4 over NumPy arrays,
so an entire (speed, angle) grid or a 100k-shot Monte Carlo burst is one call.
Roughly 300x faster per trajectory, and it agrees with the scipy reference to
well under a millimetre (see tests/test_batch.py).

Model: point mass, quadratic drag, optional constant-C_L backspin lift, in the
vertical plane through the shooter and the hub centre. Positive x is toward
the hub, y is up.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Aero:
    """Per-unit-mass aerodynamic constants (both in 1/m).

    k_drag = rho * Cd * A / (2 m);  drag  acceleration = k_drag * |v| * v, opposing v
    k_lift = rho * Cl * A / (2 m);  lift  acceleration = k_lift * |v|^2, 90 deg CCW
             from v (backspin lifts a ball travelling toward +x)
    """
    k_drag: float
    k_lift: float = 0.0
    gravity: float = 9.81

    @classmethod
    def from_ball(cls, mass_kg: float, radius_m: float, drag_coefficient: float,
                  lift_coefficient: float = 0.0, air_density: float = 1.225,
                  gravity: float = 9.81) -> "Aero":
        k = 0.5 * air_density * math.pi * radius_m ** 2 / mass_kg
        return cls(k_drag=k * drag_coefficient, k_lift=k * lift_coefficient,
                   gravity=gravity)


@dataclass
class BatchResult:
    """Per-shot outcome arrays, all shaped like the broadcast inputs.

    hit   True where the ball crossed the target plane while descending.
    x, t, vx, vy   state at that crossing (NaN where hit is False).
    apex_x, apex_y highest point of the flight.
    """
    hit: np.ndarray
    x: np.ndarray
    t: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    apex_x: np.ndarray
    apex_y: np.ndarray

    @property
    def entry_angle_deg(self) -> np.ndarray:
        """Descent angle below horizontal at the target plane (90 = straight down)."""
        return np.degrees(np.arctan2(-self.vy, np.abs(self.vx)))

    @property
    def entry_speed(self) -> np.ndarray:
        return np.hypot(self.vx, self.vy)


def simulate_batch(
    v_exit,
    theta_deg,
    exit_height,
    target_height: float,
    aero: Aero,
    dt: float = 0.005,
    t_max: float = 6.0,
) -> BatchResult:
    """Fly every (v_exit, theta_deg, exit_height) combination at once.

    Inputs broadcast against each other. Integration is classical RK4 with a
    fixed step; the target-plane crossing and the apex are located inside the
    bracketing step with a cubic Hermite interpolant, so accuracy does not
    depend on where the step boundaries happen to fall.
    """
    v_in, th_in, h_in = np.broadcast_arrays(
        np.asarray(v_exit, dtype=float),
        np.radians(np.asarray(theta_deg, dtype=float)),
        np.asarray(exit_height, dtype=float),
    )
    shape = v_in.shape
    n = v_in.size
    kd, kl, g = aero.k_drag, aero.k_lift, aero.gravity
    H = float(target_height)

    out_hit = np.zeros(n, dtype=bool)
    out = {k: np.full(n, np.nan) for k in ("x", "t", "vx", "vy")}
    out_apex_x = np.zeros(n)
    out_apex_y = h_in.ravel().astype(float).copy()

    # Live state: only shots still in flight are carried forward.
    idx = np.arange(n)
    x = np.zeros(n)
    y = h_in.ravel().astype(float).copy()
    vx = (v_in * np.cos(th_in)).ravel()
    vy = (v_in * np.sin(th_in)).ravel()
    t = 0.0

    def accel(ux, uy):
        sp = np.hypot(ux, uy)
        return -kd * sp * ux - kl * sp * uy, -g - kd * sp * uy + kl * sp * ux

    h = dt
    n_steps = int(math.ceil(t_max / dt))
    for _ in range(n_steps):
        if idx.size == 0:
            break
        # RK4. Acceleration depends on velocity only, so the position update
        # collapses to x1 = x0 + h v0 + h^2/6 (k1 + k2 + k3).
        k1x, k1y = accel(vx, vy)
        k2x, k2y = accel(vx + 0.5 * h * k1x, vy + 0.5 * h * k1y)
        k3x, k3y = accel(vx + 0.5 * h * k2x, vy + 0.5 * h * k2y)
        k4x, k4y = accel(vx + h * k3x, vy + h * k3y)
        vx1 = vx + h / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
        vy1 = vy + h / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y)
        x1 = x + h * vx + h * h / 6.0 * (k1x + k2x + k3x)
        y1 = y + h * vy + h * h / 6.0 * (k1y + k2y + k3y)

        # Apex: vy changes sign inside this step.
        top = (vy > 0.0) & (vy1 <= 0.0)
        if top.any():
            s = vy[top] / (vy[top] - vy1[top])
            out_apex_y[idx[top]] = _hermite(s, y[top], y1[top], vy[top], vy1[top], h)
            out_apex_x[idx[top]] = _hermite(s, x[top], x1[top], vx[top], vx1[top], h)

        # Descending crossing of the target plane inside this step.
        cross = (y > H) & (y1 <= H)
        if cross.any():
            y0c, y1c, vy0c, vy1c = y[cross], y1[cross], vy[cross], vy1[cross]
            s = (y0c - H) / (y0c - y1c)             # linear first guess
            for _newton in range(4):
                f = _hermite(s, y0c, y1c, vy0c, vy1c, h) - H
                df = _hermite_d(s, y0c, y1c, vy0c, vy1c, h)
                s = np.clip(s - f / np.where(df == 0.0, -1e-12, df), 0.0, 1.0)
            ci = idx[cross]
            out_hit[ci] = True
            out["t"][ci] = t + s * h
            out["x"][ci] = _hermite(s, x[cross], x1[cross], vx[cross], vx1[cross], h)
            out["vx"][ci] = _hermite_d(s, x[cross], x1[cross], vx[cross], vx1[cross], h) / h
            out["vy"][ci] = _hermite_d(s, y0c, y1c, vy0c, vy1c, h) / h

        # A ball below the plane and falling can never cross it going down.
        dead = (~cross) & (vy1 < 0.0) & (y1 < H)
        keep = ~(cross | dead)
        if not keep.all():
            idx, x, y, vx, vy = idx[keep], x1[keep], y1[keep], vx1[keep], vy1[keep]
        else:
            x, y, vx, vy = x1, y1, vx1, vy1
        t += h

    return BatchResult(
        hit=out_hit.reshape(shape),
        x=out["x"].reshape(shape), t=out["t"].reshape(shape),
        vx=out["vx"].reshape(shape), vy=out["vy"].reshape(shape),
        apex_x=out_apex_x.reshape(shape), apex_y=out_apex_y.reshape(shape),
    )


def _hermite(s, p0, p1, d0, d1, h):
    """Cubic Hermite position at fraction s of a step of length h."""
    s2 = s * s
    s3 = s2 * s
    return ((2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * h * d0
            + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * h * d1)


def _hermite_d(s, p0, p1, d0, d1, h):
    """d/ds of the cubic Hermite position."""
    s2 = s * s
    return ((6 * s2 - 6 * s) * p0 + (3 * s2 - 4 * s + 1) * h * d0
            + (-6 * s2 + 6 * s) * p1 + (3 * s2 - 2 * s) * h * d1)


def no_drag_speed(distance, theta_deg, exit_height, target_height, gravity=9.81):
    """Closed-form launch speed that lands at `distance` with no air (NaN if the
    angle is too flat to ever reach the target height at that distance)."""
    d = np.asarray(distance, dtype=float)
    th = np.radians(np.asarray(theta_deg, dtype=float))
    rise = d * np.tan(th) - (target_height - np.asarray(exit_height, dtype=float))
    with np.errstate(invalid="ignore", divide="ignore"):
        v2 = 0.5 * gravity * d ** 2 / (np.cos(th) ** 2 * rise)
        return np.where(rise > 0, np.sqrt(v2), np.nan)


def speed_for_distance(
    distance,
    theta_deg,
    exit_height,
    target_height: float,
    aero: Aero,
    v_max: float = 40.0,
    xtol: float = 1e-9,
    dt: float = 0.005,
) -> np.ndarray:
    """Launch speed that drops the ball exactly `distance` away, per element.

    Range grows monotonically with speed at a fixed angle, so each element is a
    bracketed 1-D root-find; they all advance in lock-step (Illinois regula
    falsi with a bisection safeguard) and finished elements drop out, so a
    whole array costs roughly ten batch simulations.

    Returns NaN where no descending shot exists: the angle is too flat for the
    ball to come *down* through the target height at that distance, or the
    speed needed exceeds v_max.
    """
    d, th, h0 = np.broadcast_arrays(
        np.asarray(distance, dtype=float),
        np.asarray(theta_deg, dtype=float),
        np.asarray(exit_height, dtype=float),
    )
    shape = d.shape
    d, th, h0 = d.ravel(), th.ravel(), h0.ravel()

    def miss(v, sel):
        r = simulate_batch(v, th[sel], h0[sel], target_height, aero, dt=dt)
        return np.where(r.hit, r.x - d[sel], -d[sel])

    result = np.full(d.size, np.nan)
    seed = no_drag_speed(d, th, h0, target_height, aero.gravity)
    sel = np.flatnonzero(np.isfinite(seed) & (seed < v_max))
    if sel.size == 0:
        return result.reshape(shape)

    lo = np.maximum(0.6 * seed[sel], 0.3)
    hi = np.minimum(1.25 * seed[sel], v_max)
    f_lo = miss(lo, sel)
    f_hi = miss(hi, sel)
    for _ in range(12):                      # widen until the root is bracketed
        grow = (f_hi < 0) & (hi < v_max)
        shrink = (f_lo > 0) & (lo > 0.3)
        if not (grow.any() or shrink.any()):
            break
        hi = np.where(grow, np.minimum(hi * 1.3, v_max), hi)
        lo = np.where(shrink, np.maximum(lo * 0.7, 0.3), lo)
        f_hi = np.where(grow, miss(hi, sel), f_hi)
        f_lo = np.where(shrink, miss(lo, sel), f_lo)
    good = (f_lo <= 0) & (f_hi >= 0)
    sel, lo, hi, f_lo, f_hi = sel[good], lo[good], hi[good], f_lo[good], f_hi[good]

    side = np.zeros(sel.size, dtype=int)
    bisect = np.zeros(sel.size, dtype=bool)
    for _ in range(100):
        if sel.size == 0:
            break
        width = hi - lo
        denom = f_hi - f_lo
        secant = (lo * f_hi - hi * f_lo) / np.where(denom != 0, denom, 1.0)
        mid = np.where(bisect | (denom == 0), 0.5 * (lo + hi), secant)
        mid = np.clip(mid, lo + 1e-3 * width, hi - 1e-3 * width)
        f_mid = miss(mid, sel)
        neg = f_mid < 0
        # Illinois: halve the stale endpoint's f when the same side moves twice.
        f_hi = np.where(neg & (side == -1), 0.5 * f_hi, f_hi)
        f_lo = np.where(~neg & (side == 1), 0.5 * f_lo, f_lo)
        lo, f_lo = np.where(neg, mid, lo), np.where(neg, f_mid, f_lo)
        hi, f_hi = np.where(neg, hi, mid), np.where(neg, f_hi, f_mid)
        side = np.where(neg, -1, 1)
        bisect = (hi - lo) > 0.5 * width     # secant stalled: bisect next time

        # A bracket that has closed while both ends still miss by centimetres
        # sits on the hit / no-hit jump, not on a root: the ball would have to
        # pass the target on the way up. Give up on it early.
        jump = ((hi - lo) < 1e-4) & (np.minimum(np.abs(f_lo), np.abs(f_hi)) > 1e-2)
        done = ((hi - lo) < xtol) | (np.abs(f_mid) < 1e-10) | jump
        if done.any():
            v_done = np.where(np.abs(f_mid) < 1e-10, mid, 0.5 * (lo + hi))[done]
            f_done = np.where(np.abs(f_mid) < 1e-10, f_mid,
                              np.where(np.abs(f_lo) < np.abs(f_hi), f_lo, f_hi))[done]
            result[sel[done]] = np.where(np.abs(f_done) < 1e-4, v_done, np.nan)
            keep = ~done
            sel, lo, hi, f_lo, f_hi = sel[keep], lo[keep], hi[keep], f_lo[keep], f_hi[keep]
            side, bisect = side[keep], bisect[keep]
    return result.reshape(shape)
