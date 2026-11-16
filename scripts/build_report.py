#!/usr/bin/env python3
"""Compute every number, table and figure in the research report.

The report template (web/report.html) holds the prose with {{slots}}; this
module fills them from the solver, so the text cannot drift away from the
code. Run through scripts/build_site.py, or directly:

    .venv/bin/python scripts/build_report.py      # rebuilds research_report.html
"""
from __future__ import annotations

import base64
import html
import io
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from svgplot import Fig, legend, polygon_px  # noqa: E402
from src.deploy import Deployment  # noqa: E402
from src.montecarlo import simulate_burst  # noqa: E402
from src.optimizer.maxprob import (ShotModel, aim_at_angle, centered_family,  # noqa: E402
                                   evaluate_shot, sweep_aim)
from src.physics.batch import Aero, simulate_batch, speed_for_distance  # noqa: E402
from src.physics.trajectory import simulate_trajectory  # noqa: E402
from src.scoring import ErrorBudget, HexOpening  # noqa: E402

# The previously shipped V3.1 hood curve, frozen here as the baseline to compare against.
V31_HOOD = (0.887787, -10.549949, 86.441900)

C_TEXT, C_T2, C_T3 = "var(--text)", "var(--text-2)", "var(--text-3)"
C_FUEL, C_SCORE, C_MISS, C_ACC = "var(--fuel)", "var(--score)", "var(--miss)", "var(--accent)"
C_SPEED, C_ANGLE, C_WAKE, C_RANGE = ("var(--src-speed)", "var(--src-angle)",
                                     "var(--src-wake)", "var(--src-range)")


def v31_hood(d):
    a, b, c = V31_HOOD
    return a * d * d + b * d + c


def pct(p: float, digits: int = 1) -> str:
    return f"{p * 100:.{digits}f}"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def compute(game, robot, dep: Deployment) -> dict:
    model = ShotModel.from_configs(game, robot, hex_bearing_deg=robot.aim.hex_bearing_deg)
    budget, eps = robot.error_budget, dep.margin_concession
    D = dict(model=model, budget=budget, eps=eps, dep=dep, game=game, robot=robot)

    # --- sweeps over distance: the four ways of picking an angle ---
    ds = np.round(np.arange(dep.d_min, dep.d_max + 1e-9, 0.25), 3)
    pick = sweep_aim(model, budget, ds, eps)
    best = sweep_aim(model, budget, ds, 0.0)
    slow = [aim_at_angle(model, budget, float(d), s.min_speed_angle_deg) for d, s in zip(ds, pick)]
    old = [aim_at_angle(model, budget, float(d), float(v31_hood(d))) for d in ds]
    D.update(ds=ds, pick=pick, best=best, slow=slow, old=old)

    # --- one distance in depth ---
    d0 = 3.0
    th = np.arange(model.theta_min, model.theta_max + 1e-9, 0.25)
    D.update(d0=d0, th=th, fam=centered_family(model, budget, [d0], th),
             pick0=sweep_aim(model, budget, [d0], eps)[0])

    # --- margin map over (distance, angle) ---
    md = np.round(np.arange(dep.d_min, dep.d_max + 1e-9, 0.1), 3)
    mt = np.arange(model.theta_min, model.theta_max + 1e-9, 0.5)
    D.update(map_d=md, map_t=mt, map=centered_family(model, budget, md, mt))

    # --- trade-off curves: probability against speed, slowest angle up to best odds ---
    trade = {}
    for d in (3.0, 4.0, 5.0):
        s = sweep_aim(model, budget, [d], eps)[0]
        b = sweep_aim(model, budget, [d], 0.0)[0]
        angles = np.linspace(s.min_speed_angle_deg, b.best_angle_deg + 3.0, 60)
        f = centered_family(model, budget, [d], angles)
        trade[d] = dict(extra=(f.speed[0] / s.min_speed_ms - 1) * 100, p=(1 - f.miss[0]) * 100,
                        pick=((s.v_exit_ms / s.min_speed_ms - 1) * 100, s.p_score * 100),
                        best=((b.v_exit_ms / s.min_speed_ms - 1) * 100, b.p_score * 100),
                        slow_p=float((1 - f.miss[0, 0]) * 100))
    D["trade"] = trade

    # --- Monte Carlo cross-check of the picks ---
    mc = []
    for d in (2.0, 3.0, 4.0, 5.0):
        s = sweep_aim(model, budget, [d], eps)[0]
        r = simulate_burst(model, budget, d, s.hood_angle_deg, s.v_exit_ms, n=400_000, seed=int(d * 10))
        mc.append(dict(d=d, s=s, r=r))
    D["mc"] = mc

    # --- the old "100 %" figure: bounded uniform noise versus a normal law ---
    spec = ErrorBudget.planar()
    s5 = sweep_aim(model, spec, [5.0], eps)[0]
    D["box5"] = simulate_burst(model, spec, 5.0, s5.hood_angle_deg, s5.v_exit_ms, n=200_000,
                               seed=1, distribution="uniform")
    D["bell5"] = simulate_burst(model, spec, 5.0, s5.hood_angle_deg, s5.v_exit_ms, n=200_000, seed=1)

    # --- how good does the flywheel have to be? ---
    scatters = [0.005, 0.010, 0.015, 0.017, 0.020, 0.025, 0.030, 0.035, 0.040]
    req_d = np.round(np.arange(dep.d_min, dep.d_max + 1e-9, 0.5), 3)
    grid = np.array([[s.p_score for s in sweep_aim(model, replace(budget, speed_sigma=sv), req_d, eps)]
                     for sv in scatters])
    D.update(req_scatter=np.array(scatters), req_d=req_d, req_grid=grid)
    need = []
    for j in range(len(req_d)):          # scatter at which P falls through 99 %
        col = grid[:, j]
        need.append(float(np.interp(0.99, col[::-1], np.array(scatters)[::-1]))
                    if col.max() >= 0.99 >= col.min() else float("nan"))
    D["req_need99"] = np.array(need)

    # --- the shipped function, graded end to end ---
    gd = np.round(np.arange(dep.d_min, dep.d_max + 1e-9, 0.05), 3)
    ev = [evaluate_shot(model, budget, float(d), float(dep.hood(d)), float(dep.v_ball(d))) for d in gd]
    D.update(grade_d=gd, grade_off=np.array([e["offset_m"] for e in ev]) * 100,
             grade_p=np.array([e["p_score"] for e in ev]))

    # --- calibration when the physics is wrong ---
    D["calib"] = calibration_study(model, dep, game)

    # --- three trajectories for the opening figure ---
    dg = 3.5
    pg = sweep_aim(model, budget, [dg], eps)[0]
    shots = []
    for name, ang in (("flat", 46.0), ("pick", pg.hood_angle_deg), ("steep", 70.0)):
        s = aim_at_angle(model, budget, dg, ang)
        tr = simulate_trajectory(s.v_exit_ms, ang, model.exit_height, model.target_height,
                                 game.ball.mass_kg, game.ball.radius_m, game.ball.drag_coefficient,
                                 air_density=game.environment.air_density_kg_m3,
                                 gravity=game.environment.gravity_m_s2, max_step=0.01)
        shots.append(dict(name=name, s=s, xy=tr.positions))
    D.update(geo_d=dg, geo=shots)

    # --- landing clouds for the top-view figure ---
    D["clouds"] = [dict(d=m["d"], s=m["s"], r=simulate_burst(
        model, budget, m["d"], m["s"].hood_angle_deg, m["s"].v_exit_ms, n=900, seed=77))
        for m in mc if m["d"] in (3.0, 5.0)]
    return D


def calibration_study(model: ShotModel, dep: Deployment, game) -> dict:
    """Robot runs the shipped function; the real air is different. Tune kSpeed
    at one distance, see what is left at the others. Then try two knobs."""
    ds = np.round(np.arange(dep.d_min, dep.d_max + 1e-9, 0.25), 3)
    hood, v_cmd = dep.hood(ds), dep.v_ball(ds)
    h0, ht = model.exit_height, model.target_height
    cases = []
    for label, cd, cl in (("drag 0.40 instead of 0.50", 0.40, 0.0),
                          ("drag 0.60", 0.60, 0.0),
                          ("backspin lift 0.15", 0.50, 0.15),
                          ("backspin lift 0.30", 0.50, 0.30),
                          ("drag 0.40 and lift 0.30", 0.40, 0.30)):
        true = Aero.from_ball(game.ball.mass_kg, game.ball.radius_m, cd, cl,
                              game.environment.air_density_kg_m3, game.environment.gravity_m_s2)
        need = speed_for_distance(ds, hood, h0, ht, true) / v_cmd         # gain that would centre each d
        k1 = float(np.interp(dep.d_ref, ds, need))
        one = (simulate_batch(v_cmd * k1, hood, h0, ht, true).x - ds) * 100
        # two-point calibration: centre a near and a far shot, gain varies linearly between
        near, far = 2.0, 4.5
        slope = float((np.interp(far, ds, need) - np.interp(near, ds, need)) / (far - near))
        k0 = float(np.interp(near, ds, need) - slope * (near - dep.d_ref))
        two = (simulate_batch(v_cmd * (k0 + slope * (ds - dep.d_ref)), hood, h0, ht, true).x - ds) * 100
        raw = (simulate_batch(v_cmd, hood, h0, ht, true).x - ds) * 100
        cases.append(dict(label=label, cd=cd, cl=cl, k1=k1, k0=k0, slope=slope,
                          raw=raw, one=one, two=two))
    return dict(ds=ds, cases=cases)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def hub_outline(X, Y, d, a, H, box_top=1.42, base_half=0.597, neck=0.37):
    return [(X(d - base_half), Y(0)), (X(d - base_half), Y(box_top)), (X(d - neck), Y(box_top)),
            (X(d - a), Y(H)), (X(d + a), Y(H)), (X(d + neck), Y(box_top)),
            (X(d + base_half), Y(box_top)), (X(d + base_half), Y(0))]


def fig_geometry(D) -> str:
    m, d = D["model"], D["geo_d"]
    a, H = m.opening.apothem, m.target_height
    f = Fig(width=780, height=540, left=14, right=14, top=34, bottom=34,
            xlim=(-0.95, 5.05), ylim=(0.0, 3.05))
    # equal scale on both axes
    sx = (f.w - f.l - f.r) / 6.0
    f.ylim = (0.0, (f.h - f.t - f.b) / sx)
    X, Y = (lambda x: float(f.X(x))), (lambda y: float(f.Y(y)))
    for ym in (1, 2, 3):
        if ym < f.ylim[1]:
            f.hline(ym, color="var(--line-soft)", dash=None, width=1)
            f.text_px(f.l + 4, Y(ym) - 6, f"{ym} m", C_T3)
    f.raw(f'<line x1="{f.l}" x2="{f.w - f.r}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" stroke="var(--line)" stroke-width="1.5"/>')
    f.raw(polygon_px(hub_outline(X, Y, d, a, H), fill="var(--ink-2)", stroke="var(--line)"))
    # robot
    f.raw(f'<rect x="{X(-0.82):.1f}" y="{Y(0.2):.1f}" width="{X(0.04) - X(-0.82):.1f}" '
          f'height="{Y(0.06) - Y(0.2):.1f}" rx="3" fill="var(--ink-2)" stroke="var(--line)" stroke-width="1.5"/>')
    f.raw(polygon_px([(X(-0.7), Y(0.2)), (X(-0.56), Y(m.exit_height - 0.03)),
                      (X(-0.06), Y(m.exit_height - 0.03)), (X(-0.02), Y(0.2))],
                     fill="var(--ink-2)", stroke="var(--line)"))
    styles = {"flat": (C_MISS, "5 5"), "pick": (C_FUEL, None), "steep": (C_ACC, "5 5")}
    for shot in D["geo"]:
        colour, dash = styles[shot["name"]]
        xy = shot["xy"]
        f.line(xy[:, 0], xy[:, 1], colour, width=2.6 if shot["name"] == "pick" else 1.8, dash=dash)
        s = shot["s"]
        i = int(np.argmax(xy[:, 1]))
        words = {"flat": "flat", "pick": "the pick", "steep": "steep"}[shot["name"]]
        below = shot["name"] == "flat"            # keep the lowest label clear of the other arcs
        f.text(xy[i, 0] - (0.55 if below else 0.0), xy[i, 1], f"{s.hood_angle_deg:.0f}° {words}: "
               f"{s.v_exit_ms:.1f} m/s, enters at {s.entry_angle_deg:.0f}°",
               colour, anchor="middle", dy=22 if below else -9, size=13)
    # opening and usable windows
    f.raw(f'<line x1="{X(d - a):.1f}" x2="{X(d + a):.1f}" y1="{Y(H):.1f}" y2="{Y(H):.1f}" '
          f'stroke="{C_T3}" stroke-width="1.5"/>')
    pick = next(s for s in D["geo"] if s["name"] == "pick")["s"]
    f.raw(f'<line x1="{X(d - pick.half_width_m):.1f}" x2="{X(d + pick.half_width_m):.1f}" '
          f'y1="{Y(H):.1f}" y2="{Y(H):.1f}" stroke="{C_SCORE}" stroke-width="4.5" stroke-linecap="round"/>')
    for sgn in (-1, 1):
        f.raw(f'<circle cx="{X(d + sgn * a):.1f}" cy="{Y(H):.1f}" r="3.2" fill="{C_TEXT}"/>')
    f.text_px(X(d + a) + 9, Y(H) + 4, f"opening, {H:.2f} m up", C_T2)
    # distance dimension
    yd = Y(0) + 17
    f.raw(f'<path d="M{X(0):.1f} {Y(0) + 5:.1f}V{yd + 5:.1f}M{X(d):.1f} {Y(0) + 5:.1f}V{yd + 5:.1f}'
          f'M{X(0):.1f} {yd:.1f}H{X(d):.1f}" stroke="{C_T3}" fill="none"/>')
    f.raw(f'<rect x="{X(d / 2) - 30:.1f}" y="{yd - 9:.1f}" width="60" height="18" fill="var(--ink-1)"/>')
    f.text_px(X(d / 2), yd + 4.5, f"{d:.2f} m", C_TEXT, anchor="middle", weight=600, size=13)
    return f.svg(f"Side view: three ways to land a ball on the hub centre from {d} m",
                 "A flat, a middle and a steep trajectory, all landing on the centre of the opening.")


def hex_points(cx, cy, scale, apothem, bearing_deg=0.0):
    circ = apothem / math.cos(math.pi / 6)
    return [(cx + scale * circ * math.cos(math.radians(bearing_deg + 30 + 60 * k)),
             cy - scale * circ * math.sin(math.radians(bearing_deg + 30 + 60 * k))) for k in range(6)]


def region_points(opening: HexOpening, entry: float, cx, cy, scale):
    lo_pts, hi_pts = [], []
    for z in np.arange(-0.7, 0.7001, 0.01):
        lo, hi = opening.along_track_limits(z, entry)
        if hi > lo:
            lo_pts.append((cx + scale * float(lo), cy - scale * z))
            hi_pts.append((cx + scale * float(hi), cy - scale * z))
    return lo_pts + hi_pts[::-1]


def fig_window(D) -> str:
    opening = D["model"].opening
    w, h, scale = 780, 268, 168.0
    parts = []
    for i, entry in enumerate((25.0, 48.0, 90.0)):
        cx, cy = 130 + i * 260, 118
        parts.append(polygon_px(hex_points(cx, cy, scale, opening.apothem), fill="var(--ink-0)",
                                stroke=C_T2, width=2))
        parts.append(polygon_px(region_points(opening, entry, cx, cy, scale), fill=C_SCORE,
                                stroke=C_SCORE, width=1, opacity=0.3))
        rx = scale * opening.ball_radius / math.sin(math.radians(entry))
        ry = scale * opening.ball_radius
        parts.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx:.1f}" ry="{ry:.1f}" fill="{C_FUEL}" opacity="0.9"/>')
        half = float(opening.half_width(entry)) * 100
        parts.append(f'<text class="lab" x="{cx}" y="{h - 26}" text-anchor="middle" fill="{C_TEXT}" '
                     f'font-weight="600" font-size="14">enters at {entry:.0f}°</text>')
        parts.append(f'<text class="lab" x="{cx}" y="{h - 8}" text-anchor="middle" fill="{C_T2}" '
                     f'font-size="13">room ±{half:.0f} cm along the shot</text>')
    parts.append(f'<text class="lab" x="12" y="18" fill="{C_T2}" font-size="12.5">shot direction</text>')
    parts.append(f'<path d="M92 14h30m-7 -5l7 5l-7 5" stroke="{C_T2}" fill="none" stroke-width="1.3"/>')
    return (f'<svg class="fig" viewBox="0 0 {w} {h}" role="img" aria-label="Looking down into the opening '
            f'at three entry angles"><title>The usable part of the opening shrinks as the entry gets '
            f'shallower</title>' + "".join(parts) + '</svg>')


def fig_budget(D) -> str:
    fam, th, s = D["fam"], D["th"], D["pick0"]
    ok = np.isfinite(fam.margin[0]) & (fam.margin[0] > 0.02)
    lo = math.floor(th[ok][0] / 2) * 2
    cm = lambda a: np.where(ok, a[0], np.nan) * 100     # noqa: E731
    f = Fig(width=780, height=300, top=26, bottom=44, xlim=(lo, 85), ylim=(0, 32))
    f.axes(range(int(math.ceil(lo / 5) * 5), 86, 5), range(0, 33, 8), xfmt=lambda v: f"{v:.0f}°",
           ylabel="landing spread along the shot, cm (one standard deviation)")
    f.band(s.hood_angle_deg - 0.01, s.best_angle_deg, C_ACC, 0.10)
    for arr, colour in ((fam.spread.vortex, C_WAKE), (fam.spread.range, C_RANGE),
                        (fam.spread.angle, C_ANGLE), (fam.spread.speed, C_SPEED)):
        f.line(th, np.minimum(cm(arr), 60), colour, width=2.2)
    f.line(th, np.minimum(cm(fam.spread.along), 60), C_TEXT, width=2.6)
    f.vline(s.min_speed_angle_deg).vline(s.best_angle_deg)
    f.vline(s.hood_angle_deg, color=C_FUEL, dash=None, width=1.8)
    f.text(s.min_speed_angle_deg, 32, "slowest", C_T2, anchor="end", dx=-5, dy=12, weight=500, size=12.5)
    f.text(s.best_angle_deg, 32, "best odds", C_T2, anchor="start", dx=5, dy=12, weight=500, size=12.5)
    f.text(s.hood_angle_deg, 32, "pick", C_FUEL, anchor="start", dx=5, dy=28, size=12.5)
    top = f.svg(f"Landing spread by cause against launch angle at {D['d0']} m")

    g = Fig(width=780, height=230, top=22, bottom=46, xlim=(lo, 85), ylim=(0, 3.6))
    g.axes(range(int(math.ceil(lo / 5) * 5), 86, 5), [0, 1, 2, 3], xfmt=lambda v: f"{v:.0f}°",
           yfmt=lambda v: f"{v:.0f} σ", xlabel="launch angle",
           ylabel="sigma margin = room ÷ spread")
    g.band(s.hood_angle_deg - 0.01, s.best_angle_deg, C_ACC, 0.10)
    g.line(th, np.where(ok, fam.margin[0], np.nan), C_ACC, width=2.6)
    g.vline(s.min_speed_angle_deg).vline(s.best_angle_deg)
    g.vline(s.hood_angle_deg, color=C_FUEL, dash=None, width=1.8)
    g.dots([s.hood_angle_deg], [s.sigma_margin], C_FUEL, r=4.5)
    bottom = g.svg(f"Sigma margin against launch angle at {D['d0']} m")
    return (top + legend([("ball speed", C_SPEED, "line"), ("launch angle", C_ANGLE, "line"),
                          ("air wobble", C_WAKE, "line"), ("distance estimate", C_RANGE, "line"),
                          ("total", C_TEXT, "line")]) + bottom)


def heat_png(z: np.ndarray, zmax: float) -> str:
    """RGBA PNG whose opacity encodes z, so the same image tints either theme."""
    from matplotlib.image import imsave
    alpha = np.clip(np.nan_to_num(z, nan=0.0) / zmax, 0, 1) ** 0.85
    rgba = np.zeros(z.shape + (4,))
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = 0x8a / 255, 0x78 / 255, 0xf0 / 255
    rgba[..., 3] = alpha * 0.92
    buf = io.BytesIO()
    imsave(buf, rgba[::-1], format="png")          # row 0 at the top of the image
    return base64.b64encode(buf.getvalue()).decode()


def fig_map(D) -> str:
    from contourpy import contour_generator
    md, mt, fam = D["map_d"], D["map_t"], D["map"]
    z = fam.margin.T                                # (angle, distance)
    f = Fig(width=780, height=420, top=22, bottom=46, right=110, xlim=(md[0], md[-1]),
            ylim=(mt[0], mt[-1]))
    x0, x1 = float(f.X(md[0])), float(f.X(md[-1]))
    y0, y1 = float(f.Y(mt[-1])), float(f.Y(mt[0]))
    f.raw(f'<image href="data:image/png;base64,{heat_png(z, 5.0)}" x="{x0:.1f}" y="{y0:.1f}" '
          f'width="{x1 - x0:.1f}" height="{y1 - y0:.1f}" preserveAspectRatio="none"/>')
    f.axes(np.arange(1.5, 5.01, 0.5), range(30, 86, 10), xfmt=lambda v: f"{v:.1f}",
           yfmt=lambda v: f"{v:.0f}°", xlabel="distance to hub, m", ylabel="launch angle")
    gen = contour_generator(x=md, y=mt, z=np.nan_to_num(z, nan=-1.0))
    for level, at_d in ((2.0, 4.45), (3.0, 2.75), (4.0, 1.78)):
        lines = gen.lines(level)
        for line in lines:
            f.line(line[:, 0], line[:, 1], C_TEXT, width=1, opacity=0.55)
        # label where the contour's upper branch passes a chosen distance
        pts = np.vstack(lines) if lines else np.empty((0, 2))
        near = pts[np.abs(pts[:, 0] - at_d) < 0.06]
        if len(near):
            j = int(np.argmax(near[:, 1]))
            f.text(near[j, 0], near[j, 1], f"{level:.0f} σ", C_TEXT, anchor="middle", dy=-6,
                   weight=500, size=12)
    ds = D["ds"]
    f.line(ds, [s.min_speed_angle_deg for s in D["pick"]], C_T2, width=1.8, dash="2 5")
    f.line(ds, [s.best_angle_deg for s in D["pick"]], C_TEXT, width=1.8, dash="6 4")
    f.line(ds, [v31_hood(d) for d in ds], C_SCORE, width=1.8, dash="1 4")
    f.line(ds, [s.hood_angle_deg for s in D["pick"]], C_FUEL, width=3)
    end = D["pick"][-1]
    lab_x = f.w - f.r + 8
    labels = sorted([(float(f.Y(end.best_angle_deg)), "best odds", C_TEXT),
                     (float(f.Y(v31_hood(ds[-1]))), "old V3.1", C_SCORE),
                     (float(f.Y(end.hood_angle_deg)), "V5 pick", C_FUEL),
                     (float(f.Y(end.min_speed_angle_deg)), "slowest", C_T2)])
    y_prev = -1e9
    for y_px, text, colour in labels:              # keep at least a line apart
        y_px = max(y_px, y_prev + 16)
        f.text_px(lab_x, y_px + 4, text, colour, weight=600)
        y_prev = y_px
    return f.svg("Sigma margin for every distance and launch angle",
                 "Brighter is safer. The pick runs just below the ridge of best odds.")


def fig_trade(D) -> str:
    f = Fig(width=780, height=330, top=24, bottom=46, right=96, xlim=(0, 7), ylim=(94.5, 100.1))
    f.axes(range(0, 8), [95, 96, 97, 98, 99, 100], xfmt=lambda v: f"+{v:.0f} %" if v else "slowest",
           yfmt=lambda v: f"{v:.0f} %", xlabel="ball speed above the slowest shot that reaches the hub",
           ylabel="balls that score")
    for d, colour in ((3.0, C_SCORE), (4.0, C_ACC), (5.0, C_SPEED)):
        t = D["trade"][d]
        f.line(t["extra"], t["p"], colour, width=2.4)
        f.dots([t["best"][0]], [t["best"][1]], "var(--ink-1)", r=4.5, stroke=colour)
        f.dots([t["pick"][0]], [t["pick"][1]], C_FUEL, r=5)
        y_end = float(np.interp(min(7.0, t["extra"][-1]), t["extra"], t["p"]))
        f.text_px(f.w - f.r + 8, float(f.Y(y_end)) + 4, f"{d:.0f} m", colour, weight=600, size=13)
    return f.svg("What extra flywheel speed buys in scoring probability") + legend(
        [("V5 pick", C_FUEL, "dot"), ("best odds (open circle)", C_T2, "dot")])


def fig_prob(D) -> str:
    ds = D["ds"]
    f = Fig(width=780, height=330, top=24, bottom=46, xlim=(ds[0] - 0.08, ds[-1] + 0.08), ylim=(93.5, 100.2))
    f.axes(np.arange(1.5, 5.01, 0.5), [94, 95, 96, 97, 98, 99, 100], xfmt=lambda v: f"{v:.1f}",
           yfmt=lambda v: f"{v:.0f} %", xlabel="distance to hub, m", ylabel="balls that score")
    P = lambda sols: [s.p_score * 100 for s in sols]     # noqa: E731
    f.line(ds, P(D["slow"]), C_T2, width=1.8, dash="2 5")
    f.line(ds, P(D["old"]), C_SCORE, width=1.8, dash="1 4")
    f.line(ds, P(D["best"]), C_TEXT, width=1.8, dash="6 4")
    f.line(ds, P(D["pick"]), C_FUEL, width=3)
    f.dots([m["d"] for m in D["mc"]], [m["r"].p_score * 100 for m in D["mc"]], C_ACC, r=4.5)
    return f.svg("Scoring probability against distance for four ways of choosing the angle") + legend(
        [("V5 pick", C_FUEL, "line"), ("best odds", C_TEXT, "dash"), ("old V3.1 curve", C_SCORE, "dash"),
         ("slowest shot", C_T2, "dash"), ("Monte Carlo, 400,000 balls", C_ACC, "dot")])


def fig_speedreq(D) -> str:
    rd = D["req_d"]
    f = Fig(width=780, height=330, top=24, bottom=46, right=92, xlim=(rd[0] - 0.08, rd[-1] + 0.08),
            ylim=(72, 100.4))
    f.axes(np.arange(1.5, 5.01, 0.5), [75, 80, 85, 90, 95, 100], xfmt=lambda v: f"{v:.1f}",
           yfmt=lambda v: f"{v:.0f} %", xlabel="distance to hub, m", ylabel="balls that score")
    f.hline(99, color=C_T3)
    f.text_px(f.w - f.r - 6, float(f.Y(99)) + 15, "99 %", C_T2, anchor="end")
    show = {0.010: C_SCORE, 0.017: C_FUEL, 0.025: C_ACC, 0.035: C_MISS}
    for sv, row in zip(D["req_scatter"], D["req_grid"]):
        colour = show.get(round(float(sv), 3))
        if colour:
            f.line(rd, row * 100, colour, width=3 if colour == C_FUEL else 2.2)
            f.text_px(f.w - f.r + 8, float(f.Y(max(row[-1] * 100, 72.5))) + 4,
                      f"{sv * 100:.1f} % scatter", colour, weight=600)
    return f.svg("Scoring probability against distance for four levels of ball-speed scatter")


def fig_scatter(D) -> str:
    opening = D["model"].opening
    w, h, scale = 780, 300, 200.0
    parts = []
    for i, c in enumerate(D["clouds"]):
        cx, cy = 200 + i * 380, 132
        parts.append(polygon_px(hex_points(cx, cy, scale, opening.apothem), fill="var(--ink-0)",
                                stroke=C_T2, width=2))
        parts.append(polygon_px(region_points(opening, c["s"].entry_angle_deg, cx, cy, scale),
                                fill=C_SCORE, stroke="none", opacity=0.16))
        r = c["r"]
        for a, l, ok in zip(r.along, r.lateral, r.scored):
            if not (np.isfinite(a) and abs(a) < 0.92 and abs(l) < 0.64):
                continue
            parts.append(f'<circle cx="{cx + scale * a:.1f}" cy="{cy - scale * l:.1f}" r="1.9" '
                         f'fill="{C_SCORE if ok else C_MISS}" opacity="{0.75 if ok else 1}"/>')
        parts.append(f'<text class="lab" x="{cx}" y="{h - 24}" text-anchor="middle" fill="{C_TEXT}" '
                     f'font-weight="600" font-size="14">{c["d"]:.0f} m: {r.scored.sum()} of {r.n} score</text>')
        parts.append(f'<text class="lab" x="{cx}" y="{h - 6}" text-anchor="middle" fill="{C_T2}" '
                     f'font-size="13">spread ±{r.along_std * 100:.0f} cm along, ±{r.lateral_std * 100:.0f} cm across</text>')
    return (f'<svg class="fig" viewBox="0 0 {w} {h}" role="img" aria-label="Simulated landing points at 3 '
            f'and 5 metres"><title>Where 900 simulated balls cross the opening</title>'
            + "".join(parts) + '</svg>')


def fig_fit(D) -> str:
    gd, off = D["grade_d"], D["grade_off"]
    lim = max(1.5, math.ceil(np.max(np.abs(off)) * 2) / 2 + 0.5)
    f = Fig(width=780, height=250, top=24, bottom=46, xlim=(gd[0] - 0.08, gd[-1] + 0.08), ylim=(-lim, lim))
    f.axes(np.arange(1.5, 5.01, 0.5), np.arange(-lim, lim + 0.01, 0.5 if lim <= 2 else 1.0),
           xfmt=lambda v: f"{v:.1f}", yfmt=lambda v: f"{v:+.1f}" if v else "0",
           xlabel="distance to hub, m", ylabel="where the fitted function lands, cm from centre")
    f.hline(0, color=C_T3, dash=None, width=1)
    f.line(gd, off, C_FUEL, width=2.6)
    return f.svg("Landing offset of the shipped quadratics against distance")


def fig_calib(D) -> str:
    c = D["calib"]
    ds = c["ds"]
    f = Fig(width=780, height=340, top=24, bottom=46, right=20, xlim=(ds[0] - 0.08, ds[-1] + 0.08),
            ylim=(-16, 16))
    f.axes(np.arange(1.5, 5.01, 0.5), range(-16, 17, 4), xfmt=lambda v: f"{v:.1f}",
           yfmt=lambda v: f"{v:+.0f}" if v else "0", xlabel="distance to hub, m",
           ylabel="landing error after calibration, cm")
    f.hline(0, color=C_T3, dash=None, width=1)
    f.vline(D["dep"].d_ref, color=C_T3)
    f.text(D["dep"].d_ref, 16, "kSpeed tuned here", C_T2, anchor="start", dx=6, dy=13, weight=500, size=12.5)
    colours = [C_ANGLE, C_SPEED, C_RANGE, C_WAKE, C_MISS]
    for case, colour in zip(c["cases"], colours):
        f.line(ds, case["one"], colour, width=2.2)
        f.line(ds, case["two"], colour, width=1.6, dash="2 4")
    return f.svg("Landing error left after tuning kSpeed, for five ways the physics could be wrong") + \
        legend([(case["label"], colour, "line") for case, colour in zip(c["cases"], colours)]
               + [("dotted: with kSpeedSlope as well", C_T2, "dash")])


def fig_hero(D) -> str:
    """Small trajectory mark for the landing page."""
    m, d = D["model"], D["geo_d"]
    a, H = m.opening.apothem, m.target_height
    f = Fig(width=720, height=384, left=6, right=6, top=10, bottom=14, xlim=(-0.95, 4.75), ylim=(0, 3))
    sx = (f.w - f.l - f.r) / 5.7
    f.ylim = (0.0, (f.h - f.t - f.b) / sx)
    X, Y = (lambda x: float(f.X(x))), (lambda y: float(f.Y(y)))
    f.raw(f'<line x1="{f.l}" x2="{f.w - f.r}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" stroke="var(--line)" stroke-width="1.5"/>')
    f.raw(polygon_px(hub_outline(X, Y, d, a, H), fill="var(--ink-2)", stroke="var(--line)"))
    f.raw(f'<rect x="{X(-0.82):.1f}" y="{Y(0.2):.1f}" width="{X(0.04) - X(-0.82):.1f}" '
          f'height="{Y(0.06) - Y(0.2):.1f}" rx="3" fill="var(--ink-2)" stroke="var(--line)" stroke-width="1.5"/>')
    f.raw(polygon_px([(X(-0.7), Y(0.2)), (X(-0.56), Y(m.exit_height - 0.03)),
                      (X(-0.06), Y(m.exit_height - 0.03)), (X(-0.02), Y(0.2))],
                     fill="var(--ink-2)", stroke="var(--line)"))
    pick = next(s for s in D["geo"] if s["name"] == "pick")
    rng = np.random.default_rng(4)
    for _ in range(40):
        v = pick["s"].v_exit_ms * (1 + D["budget"].speed_sigma * rng.standard_normal())
        th = pick["s"].hood_angle_deg + D["budget"].angle_sigma_deg * rng.standard_normal()
        tr = simulate_trajectory(v, th, m.exit_height, H, 0.215, 0.075, 0.5, max_step=0.02)
        f.line(tr.positions[:, 0], tr.positions[:, 1], C_FUEL, width=1.1, opacity=0.28)
    f.line(pick["xy"][:, 0], pick["xy"][:, 1], C_FUEL, width=2.4)
    f.raw(f'<line x1="{X(d - pick["s"].half_width_m):.1f}" x2="{X(d + pick["s"].half_width_m):.1f}" '
          f'y1="{Y(H):.1f}" y2="{Y(H):.1f}" stroke="{C_SCORE}" stroke-width="4.5" stroke-linecap="round"/>')
    return f.svg("Forty simulated balls arcing from the robot into the hub")


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #

def table(head: list[str], rows: list[list[str]], cls: str = "") -> str:
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tablewrap"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def tbl_lookup(D) -> str:
    dep = D["dep"]
    rows = []
    for s in D["pick"]:
        fly = s.v_exit_ms / dep.speed_transfer
        rows.append([f"{s.distance_m:.2f}", f"{s.hood_angle_deg:.1f}°", f"{s.v_exit_ms:.2f}",
                     f"{fly:.2f}", f"{s.time_of_flight_s:.2f}", f"{s.entry_angle_deg:.0f}°",
                     f"{s.apex_height_m:.2f}", f"{s.p_score * 100:.2f} %", f"{s.sigma_margin:.2f}"])
    return table(["Distance, m", "Launch angle", "Ball speed, m/s", "Flywheel surface, m/s",
                  "Flight, s", "Entry angle", "Peak, m", "Balls that score", "Margin, σ"], rows, "num")


def tbl_budget(D) -> str:
    b, s = D["budget"], D["pick0"]
    rows = [
        ["Ball speed, ball to ball", f"{b.speed_sigma * 100:.1f} % of speed", f"{s.spread_speed_m * 100:.1f} cm",
         "Film 30 shots at one setting, or read it off the landing scatter"],
        ["Launch angle, ball to ball", f"{b.angle_sigma_deg:.1f}°", f"{s.spread_angle_m * 100:.1f} cm",
         "Same film; hood backlash and how the ball seats"],
        ["Air wobble in flight", f"{b.vortex_coeff * 100:.0f} cm × (flight time in s)²",
         f"{s.spread_vortex_m * 100:.1f} cm", "Hard to isolate; whatever scatter is left over"],
        ["Distance estimate", f"{b.range_sigma_m * 100:.0f} cm", f"{s.spread_range_m * 100:.1f} cm",
         "Compare vision range against a tape measure"],
        ["<b>Total along the shot</b>", "", f"<b>{s.spread_along_m * 100:.1f} cm</b>", ""],
        ["Robot heading", f"{b.yaw_sigma_deg:.1f}°", f"{s.distance_m * math.radians(b.yaw_sigma_deg) * 100:.1f} cm sideways",
         "Gyro heading error while aiming"],
        ["Dumper width", f"±{b.lateral_half_width_m * 100:.0f} cm, evenly spread", "up to that, sideways",
         "Measure the exit"],
    ]
    return table(["Error source", "Assumed size (one standard deviation)",
                  f"Landing spread it causes at {D['d0']:.0f} m", "How to measure it"], rows)


def tbl_window(D) -> str:
    opening = D["model"].opening
    rows = [[f"{e}°", f"{opening.ball_radius / math.sin(math.radians(e)) * 100:.1f} cm",
             f"±{float(opening.half_width(e)) * 100:.1f} cm"] for e in (20, 30, 40, 50, 60, 75, 90)]
    return table(["Entry angle", "Ball shadow half-length, r / sin φ", "Room along the shot"], rows, "num")


def tbl_mc(D) -> str:
    rows = []
    for m in D["mc"]:
        s, r = m["s"], m["r"]
        rows.append([f"{m['d']:.0f} m", f"{(1 - s.p_score) * 100:.3f} %",
                     f"{(1 - r.p_score) * 100:.3f} % ± {r.stderr * 100:.3f}",
                     f"{s.spread_along_m * 100:.1f} cm", f"{r.along_std * 100:.1f} cm"])
    return table(["Distance", "Missed, model", "Missed, 400,000 simulated balls",
                  "Spread along the shot, model", "Spread, simulated"], rows, "num")


def tbl_coeffs(D) -> str:
    dep = D["dep"]
    rows = [[name, f"{q.a:+.6f}", f"{q.b:+.6f}", f"{q.c:+.6f}", err] for name, q, err in (
        ("Launch angle, degrees", dep.hood, f"{dep.hood.rmse:.2f}°"),
        ("Flywheel surface speed, m/s", dep.flywheel, f"{dep.flywheel.rmse:.3f} m/s"),
        ("Ball speed, m/s (for reference)", dep.v_ball, f"{dep.v_ball.rmse:.3f} m/s"),
        ("Time of flight, s", dep.tof, f"{dep.tof.rmse:.3f} s"))]
    return table(["Output", "a", "b", "c", "Fit error (RMS)"], rows, "num")


def tbl_calib(D) -> str:
    rows = []
    for c in D["calib"]["cases"]:
        rows.append([c["label"], f"{np.max(np.abs(c['raw'])):.0f} cm", f"{c['k1']:.3f}",
                     f"{np.max(np.abs(c['one'])):.1f} cm", f"{c['slope']:+.4f}",
                     f"{np.max(np.abs(c['two'])):.1f} cm"])
    return table(["If the real air is…", "Worst miss, untuned", "kSpeed that fixes 3 m",
                  "Worst miss, kSpeed only", "kSpeedSlope", "Worst miss, both knobs"], rows, "num")


# --------------------------------------------------------------------------- #
# Slots
# --------------------------------------------------------------------------- #

def slots(D) -> dict:
    dep, m, b = D["dep"], D["model"], D["budget"]
    by_d = {round(float(s.distance_m), 2): s for s in D["pick"]}
    best_d = {round(float(s.distance_m), 2): s for s in D["best"]}
    old_d = {round(float(s.distance_m), 2): s for s in D["old"]}
    slow_d = {round(float(s.distance_m), 2): s for s in D["slow"]}
    s3, s5, p0 = by_d[3.0], by_d[5.0], D["pick0"]
    hood_gap = max(abs(a.hood_angle_deg - o.hood_angle_deg) for a, o in zip(D["pick"], D["old"]))
    p_gap = max(abs(a.p_score - o.p_score) for a, o in zip(D["pick"], D["old"])) * 100
    best_gain = max(bb.p_score - a.p_score for a, bb in zip(D["pick"], D["best"])) * 100
    speed_cost = max(bb.v_exit_ms / a.v_exit_ms - 1 for a, bb in zip(D["pick"], D["best"])) * 100
    share = lambda x, tot: (x / tot) ** 2 * 100        # noqa: E731
    need = {round(float(d), 1): n for d, n in zip(D["req_d"], D["req_need99"])}
    fam = D["fam"]
    idx = lambda ang: int(np.argmin(np.abs(D["th"] - ang)))     # noqa: E731
    game = D["game"]
    worst = max(D["calib"]["cases"], key=lambda c: np.max(np.abs(c["one"])))
    S = {
        # figures and tables
        "fig_geometry": fig_geometry(D), "fig_window": fig_window(D), "fig_budget": fig_budget(D),
        "fig_map": fig_map(D), "fig_trade": fig_trade(D), "fig_prob": fig_prob(D),
        "fig_speedreq": fig_speedreq(D), "fig_scatter": fig_scatter(D), "fig_fit": fig_fit(D),
        "fig_calib": fig_calib(D),
        "tbl_lookup": tbl_lookup(D), "tbl_budget": tbl_budget(D), "tbl_window": tbl_window(D),
        "tbl_mc": tbl_mc(D), "tbl_coeffs": tbl_coeffs(D), "tbl_calib": tbl_calib(D),
        # headline numbers
        "p2": pct(by_d[2.0].p_score, 2), "p3": pct(s3.p_score), "p4": pct(by_d[4.0].p_score),
        "p5": pct(s5.p_score),
        "hood2": f"{by_d[2.0].hood_angle_deg:.1f}", "hood3": f"{s3.hood_angle_deg:.1f}",
        "hood4": f"{by_d[4.0].hood_angle_deg:.1f}", "hood5": f"{s5.hood_angle_deg:.1f}",
        "fly2": f"{by_d[2.0].v_exit_ms / dep.speed_transfer:.2f}",
        "fly3": f"{s3.v_exit_ms / dep.speed_transfer:.2f}",
        "fly4": f"{by_d[4.0].v_exit_ms / dep.speed_transfer:.2f}",
        "fly5": f"{s5.v_exit_ms / dep.speed_transfer:.2f}",
        "d_min": f"{dep.d_min:g}", "d_max": f"{dep.d_max:g}", "d_ref": f"{dep.d_ref:g}",
        # physics constants
        "exit_h": f"{dep.exit_height_m:.2f}", "target_h": f"{m.target_height:.2f}",
        "across_flats": f"{2 * m.opening.apothem:.2f}", "apothem_cm": f"{m.opening.apothem * 100:.0f}",
        "ball_r_cm": f"{m.opening.ball_radius * 100:.1f}", "ball_d_cm": f"{m.opening.ball_radius * 200:.0f}",
        "ball_mass_g": f"{game.ball.mass_kg * 1000:.0f}", "cd": f"{game.ball.drag_coefficient:.2f}",
        "k_drag": f"{m.aero.k_drag:.4f}", "transfer": f"{dep.speed_transfer:.2f}",
        "slip_pct": f"{(1 - dep.speed_transfer) * 100:.0f}",
        "wheel_mm": f"{dep.wheel_diameter_m * 1000:.1f}",
        "theta_min": f"{m.theta_min:.0f}", "theta_max": f"{m.theta_max:.0f}",
        "half_vertical_cm": f"{float(m.opening.half_width(90)) * 100:.1f}",
        # budget
        "sv_pct": f"{b.speed_sigma * 100:.1f}", "sa_deg": f"{b.angle_sigma_deg:.1f}",
        "eps_pct": f"{D['eps'] * 100:.0f}", "keep_pct": f"{(1 - D['eps']) * 100:.0f}",
        "vortex_cm": f"{b.vortex_coeff * 100:.0f}",
        # depth at 3 m
        "d0": f"{D['d0']:.0f}", "d0_hood": f"{p0.hood_angle_deg:.1f}", "d0_best": f"{p0.best_angle_deg:.1f}",
        "d0_slow": f"{p0.min_speed_angle_deg:.1f}", "d0_speed_cm": f"{p0.spread_speed_m * 100:.1f}",
        "d0_angle_cm": f"{p0.spread_angle_m * 100:.1f}", "d0_wake_cm": f"{p0.spread_vortex_m * 100:.1f}",
        "d0_range_cm": f"{p0.spread_range_m * 100:.1f}", "d0_total_cm": f"{p0.spread_along_m * 100:.1f}",
        "d0_room_cm": f"{p0.half_width_m * 100:.0f}", "d0_margin": f"{p0.sigma_margin:.2f}",
        "d0_speed_share": f"{share(p0.spread_speed_m, p0.spread_along_m):.0f}",
        "d0_cm_per_pct": f"{p0.dx_dlnv_m:.1f}",
        "d0_cm_per_deg": f"{abs(p0.dx_dtheta_m_per_deg) * 100:.1f}",
        "sens50": f"{fam.dx_dlnv[0, idx(50.0)]:.1f}", "sens60": f"{fam.dx_dlnv[0, idx(60.0)]:.1f}",
        "sens70": f"{fam.dx_dlnv[0, idx(70.0)]:.1f}",
        "s5_speed_cm": f"{s5.spread_speed_m * 100:.0f}", "s5_angle_cm": f"{s5.spread_angle_m * 100:.0f}",
        "s5_total_cm": f"{s5.spread_along_m * 100:.0f}", "s5_room_cm": f"{s5.half_width_m * 100:.0f}",
        "s5_speed_share": f"{share(s5.spread_speed_m, s5.spread_along_m):.0f}",
        # comparisons
        "hood_gap": f"{hood_gap:.1f}", "p_gap": f"{p_gap:.2f}",
        "best_gain": f"{best_gain:.1f}", "speed_cost": f"{speed_cost:.1f}",
        "p5_best": pct(best_d[5.0].p_score), "p5_old": pct(old_d[5.0].p_score),
        "p5_slow": pct(slow_d[5.0].p_score), "p3_slow": pct(slow_d[3.0].p_score),
        "hood5_best": f"{best_d[5.0].hood_angle_deg:.1f}", "hood5_old": f"{old_d[5.0].hood_angle_deg:.1f}",
        "box5": pct(D["box5"].p_score, 2), "bell5": pct(D["bell5"].p_score),
        "need99_4": f"{need.get(4.0, float('nan')) * 100:.1f}", "need99_5": f"{need.get(5.0, float('nan')) * 100:.1f}",
        "need99_3": f"{need.get(3.0, float('nan')) * 100:.1f}",
        # deployment
        "fit_offset_cm": f"{dep.validation['max_abs_landing_offset_cm']:.1f}",
        "fit_ploss": f"{dep.validation['max_p_loss_vs_solver_pp']:.2f}",
        "fit_hood_max": f"{dep.validation['hood_fit_max_abs_deg']:.2f}",
        "hood_a": f"{dep.hood.a:.6f}", "hood_b": f"{abs(dep.hood.b):.6f}", "hood_c": f"{dep.hood.c:.6f}",
        "hood_bs": "+" if dep.hood.b >= 0 else "-",
        "fly_a": f"{abs(dep.flywheel.a):.6f}", "fly_as": "" if dep.flywheel.a >= 0 else "-",
        "fly_b": f"{abs(dep.flywheel.b):.6f}", "fly_bs": "+" if dep.flywheel.b >= 0 else "-",
        "fly_c": f"{dep.flywheel.c:.6f}",
        "tof_a": f"{abs(dep.tof.a):.6f}", "tof_as": "" if dep.tof.a >= 0 else "-",
        "tof_b": f"{abs(dep.tof.b):.6f}", "tof_bs": "+" if dep.tof.b >= 0 else "-",
        "tof_c": f"{dep.tof.c:.6f}",
        "calib_worst_label": worst["label"], "calib_worst_one": f"{np.max(np.abs(worst['one'])):.0f}",
        "calib_worst_two": f"{max(np.max(np.abs(c['two'])) for c in D['calib']['cases']):.1f}",
        "calib_typ_one": f"{np.median([np.max(np.abs(c['one'])) for c in D['calib']['cases']]):.0f}",
        "geo_d": f"{D['geo_d']:.1f}",
        "mc_n": "400,000",
    }
    return S


def build_report(game, robot, dep: Deployment, assemble) -> str:
    out = assemble("report.html", slots(compute(game, robot, dep)))
    left = sorted(set(__import__("re").findall(r"\{\{(\w+)\}\}", out)))
    if left:
        raise RuntimeError(f"report template has unfilled slots: {left}")
    return out


def landing_slots(game, robot, dep: Deployment) -> dict:
    model = ShotModel.from_configs(game, robot, hex_bearing_deg=robot.aim.hex_bearing_deg)
    budget = robot.error_budget
    D = dict(model=model, budget=budget)
    dg = 3.5
    pg = sweep_aim(model, budget, [dg], dep.margin_concession)[0]
    tr = simulate_trajectory(pg.v_exit_ms, pg.hood_angle_deg, model.exit_height, model.target_height,
                             game.ball.mass_kg, game.ball.radius_m, game.ball.drag_coefficient,
                             max_step=0.01)
    D.update(geo_d=dg, geo=[dict(name="pick", s=pg, xy=tr.positions)])
    s3, s5 = sweep_aim(model, budget, [3.0, 5.0], dep.margin_concession)
    return {"fig_hero": fig_hero(D), "p3": pct(s3.p_score), "p5": pct(s5.p_score),
            "d_min": f"{dep.d_min:g}", "d_max": f"{dep.d_max:g}", "exit_h": f"{dep.exit_height_m:.2f}"}


if __name__ == "__main__":
    from build_site import main
    raise SystemExit(main())
