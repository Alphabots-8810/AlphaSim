"""The simulator's JavaScript core must compute what the Python package computes.

web/alphasim_core.js is a port of the batch integrator, the scoring model and
the V5 solver. This runs both on the same inputs and compares. Skipped when
Node.js is not installed.
"""
import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtr, ndtri

from src.optimizer.maxprob import evaluate_shot, sweep_aim
from src.physics.batch import simulate_batch, speed_for_distance
from src.scoring import ErrorBudget, miss_probability, landing_spread, sigma_margin

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "web" / "alphasim_core.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")


def run_js(expr_body: str):
    """Evaluate a JS function body with `C` bound to the core; return its JSON."""
    script = (f"const C = require({json.dumps(str(CORE))});\n"
              f"const out = (function() {{ {expr_body} }})();\n"
              "process.stdout.write(JSON.stringify(out, (k, v) => "
              "(typeof v === 'number' && !Number.isFinite(v)) ? String(v) : v));")
    res = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    return json.loads(res.stdout)


def js_model(model):
    return json.dumps(dict(
        exitHeight=model.exit_height, targetHeight=model.target_height,
        aero=dict(kd=model.aero.k_drag, kl=model.aero.k_lift, g=model.aero.gravity),
        opening=dict(apothem=model.opening.apothem, ballRadius=model.opening.ball_radius,
                     bearingDeg=model.opening.bearing_deg),
        thetaMin=model.theta_min, thetaMax=model.theta_max, vMax=model.v_max))


def test_special_functions():
    xs = [-9.0, -5.5, -3.0, -1.2, -0.3, 0.0, 0.4, 1.7, 2.49, 2.51, 4.0, 7.5]
    ps = [1e-250, 1e-40, 1e-9, 1e-4, 0.01, 0.024, 0.3, 0.5, 0.8, 0.99, 1 - 1e-9]
    got = run_js(f"return {{cdf: {xs}.map(C.ndtr), inv: {ps}.map(C.ndtri)}};")
    assert got["cdf"] == pytest.approx(ndtr(xs), rel=1e-11, abs=1e-300)
    assert got["inv"] == pytest.approx(ndtri(ps), rel=1e-9, abs=1e-9)


def test_trajectories(model):
    rng = np.random.default_rng(21)
    v, th = rng.uniform(5.5, 11, 12).tolist(), rng.uniform(35, 80, 12).tolist()
    want = simulate_batch(v, th, model.exit_height, model.target_height, model.aero)
    got = run_js(f"const m = {js_model(model)}; const v = {v}, th = {th};"
                 "return v.map((vi, i) => C.fly(vi, th[i], m.exitHeight, m.targetHeight, m.aero));")
    for i, g in enumerate(got):
        assert g["hit"] == bool(want.hit[i])
        if g["hit"]:
            for key, arr in (("x", want.x), ("t", want.t), ("vx", want.vx), ("vy", want.vy),
                             ("apexY", want.apex_y), ("apexX", want.apex_x)):
                assert g[key] == pytest.approx(float(arr[i]), abs=1e-10)


def test_centring_speed(model):
    cases = [(1.5, 72.0), (3.0, 45.0), (3.0, 62.5), (5.0, 55.0), (5.0, 80.0), (1.5, 30.0)]
    want = [float(speed_for_distance(d, th, model.exit_height, model.target_height, model.aero,
                                     v_max=model.v_max, dt=0.01)) for d, th in cases]
    got = run_js(f"const m = {js_model(model)};"
                 f"return {json.dumps(cases)}.map(([d, th]) => C.speedForDistance("
                 "d, th, m.exitHeight, m.targetHeight, m.aero, m.vMax, 0.01));")
    for g, w in zip(got, want):
        if np.isnan(w):
            assert g == "NaN"
        else:
            assert g == pytest.approx(w, abs=1e-8)


@pytest.mark.parametrize("bearing", [0.0, 30.0, 12.5])
def test_miss_probability(model, bearing):
    from dataclasses import replace
    opening = replace(model.opening, bearing_deg=bearing)
    budget = ErrorBudget(lateral_half_width_m=0.2)
    rows = []
    for dxv, dxt, tof, d, entry, bias in [(7.6, -0.05, 0.97, 3.0, 48.7, 0.0),
                                          (11.2, -0.04, 1.12, 5.0, 45.7, 0.12),
                                          (4.6, -0.06, 0.81, 1.5, 52.9, -0.3),
                                          (20.0, 0.3, 0.6, 3.0, 12.0, 0.0)]:
        sp = landing_spread(budget, np.array(dxv), np.array(dxt), np.array(tof), np.array(d))
        rows.append((dxv, dxt, tof, d, entry, bias,
                     float(miss_probability(opening, budget, sp, entry, bias))))
    js_open = json.dumps(dict(apothem=opening.apothem, ballRadius=opening.ball_radius,
                              bearingDeg=bearing))
    got = run_js(f"const o = {js_open}, b = {json.dumps(asdict(budget))};"
                 f"return {json.dumps([r[:6] for r in rows])}.map(([dxv, dxt, tof, d, e, bias]) => {{"
                 "const s = C.landingSpread(b, dxv, dxt, tof, d);"
                 "const miss = C.missProbability(o, b, s, e, bias);"
                 "return [miss, C.sigmaMargin(miss)]; });")
    for (miss_js, margin_js), row in zip(got, rows):
        assert miss_js == pytest.approx(row[6], rel=1e-9, abs=1e-15)
        assert margin_js == pytest.approx(float(sigma_margin(row[6])), rel=1e-8)


@pytest.mark.parametrize("concession", [0.0, 0.03])
def test_solver_picks_the_same_shot(model, budget, concession):
    distances = [1.5, 2.75, 4.0, 5.0]
    want = sweep_aim(model, budget, distances, concession)
    got = run_js(f"const m = {js_model(model)}, b = {json.dumps(asdict(budget))};"
                 f"return {distances}.map((d) => {{ const s = C.solveAim(m, b, d, {concession});"
                 "return {hood: s.theta, v: s.v, tof: s.tof, entry: s.entry, apex: s.apex,"
                 " p: s.pScore, margin: s.margin, best: s.choice.best, slow: s.choice.slow,"
                 " along: s.spread.along, lateral: s.spread.lateral}; });")
    for g, w in zip(got, want):
        assert g["hood"] == pytest.approx(w.hood_angle_deg, abs=1e-6)
        assert g["v"] == pytest.approx(w.v_exit_ms, abs=1e-7)
        assert g["tof"] == pytest.approx(w.time_of_flight_s, abs=1e-7)
        assert g["entry"] == pytest.approx(w.entry_angle_deg, abs=1e-6)
        assert g["apex"] == pytest.approx(w.apex_height_m, abs=1e-7)
        assert g["p"] == pytest.approx(w.p_score, abs=1e-9)
        assert g["margin"] == pytest.approx(w.sigma_margin, abs=1e-6)
        assert g["best"] == pytest.approx(w.best_angle_deg, abs=1e-6)
        assert g["slow"] == pytest.approx(w.min_speed_angle_deg, abs=1e-6)
        assert g["along"] == pytest.approx(w.spread_along_m, abs=1e-8)
        assert g["lateral"] == pytest.approx(w.spread_lateral_m, abs=1e-8)


def test_grading_an_off_centre_shot(model, budget):
    want = evaluate_shot(model, budget, 3.0, 62.0, 7.3)
    got = run_js(f"const m = {js_model(model)}, b = {json.dumps(asdict(budget))};"
                 "const e = C.evaluateShot(m, b, 3.0, 62.0, 7.3);"
                 "return {offset: e.offset, p: e.pScore, margin: e.margin, tof: e.tof};")
    assert got["offset"] == pytest.approx(want["offset_m"], abs=1e-9)
    assert got["p"] == pytest.approx(want["p_score"], abs=1e-9)
    assert got["margin"] == pytest.approx(want["sigma_margin"], abs=1e-6)


def test_js_burst_agrees_with_the_model_statistically(model, budget):
    s = sweep_aim(model, budget, [4.5])[0]
    got = run_js(f"const m = {js_model(model)}, b = {json.dumps(asdict(budget))};"
                 f"const r = C.burst(m, b, 4.5, {s.hood_angle_deg}, {s.v_exit_ms}, 120000, 7, false);"
                 "let sx = 0, sxx = 0, n = 0; for (const q of r.balls) if (q.hit)"
                 " { sx += q.along; sxx += q.along * q.along; n++; }"
                 "return {p: r.pScore, std: Math.sqrt(sxx / n - (sx / n) ** 2)};")
    assert got["p"] == pytest.approx(s.p_score, abs=0.004)
    assert got["std"] == pytest.approx(s.spread_along_m, rel=0.02)
