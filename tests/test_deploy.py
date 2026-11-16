"""The shipped function: fitted, graded, rendered - and in sync with the repo."""
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.deploy import Quadratic, render_java, render_python, to_json
from src.optimizer.maxprob import evaluate_shot, sweep_aim

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from generate_coefficients import (SIM_BLOCK, deployment_from_configs,  # noqa: E402
                                   sim_block)


@pytest.fixture(scope="module")
def dep(configs):
    return deployment_from_configs(*configs)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_quadratic_fit_rounds_to_shipped_precision():
    d = np.linspace(1.5, 5, 20)
    q = Quadratic.fit(d, 0.5 * d * d - 2 * d + 3.123456789)
    assert (q.a, q.b, q.c) == (0.5, -2.0, 3.123457)
    assert q.rmse < 1e-6
    assert q.scaled(2).c == pytest.approx(6.246914)


def test_fitted_function_lands_on_centre(dep, model, budget):
    """End to end: evaluate the shipped quadratics, fly the ball, see where it lands."""
    for d in np.arange(dep.d_min, dep.d_max + 1e-9, 0.125):
        hood, fly = dep.aim(d)
        shot = evaluate_shot(model, budget, float(d), float(hood), float(fly) * dep.speed_transfer)
        assert shot["hit"]
        assert abs(shot["offset_m"]) < 0.02          # 2 cm, against a +/-43 cm opening
    assert dep.validation["max_abs_landing_offset_cm"] < 2.0


def test_fitting_costs_almost_no_hit_probability(dep, model, budget):
    assert dep.validation["max_p_loss_vs_solver_pp"] < 0.1
    exact = sweep_aim(model, budget, [3.0, 5.0], dep.margin_concession)
    for s in exact:
        hood, fly = dep.aim(s.distance_m)
        fitted = evaluate_shot(model, budget, s.distance_m, float(hood),
                               float(fly) * dep.speed_transfer)
        assert fitted["p_score"] == pytest.approx(s.p_score, abs=1e-3)
        assert float(dep.tof(s.distance_m)) == pytest.approx(s.time_of_flight_s, abs=0.02)


def test_inputs_outside_the_range_are_clamped(dep):
    assert dep.aim(0.2) == dep.aim(dep.d_min)
    assert dep.aim(9.0) == dep.aim(dep.d_max)


def test_gain_knobs(dep):
    hood, fly = dep.aim(4.0)
    hood2, fly2 = dep.aim(4.0, k_speed=1.05, k_speed_slope=0.01)
    assert hood2 == hood
    assert fly2 == pytest.approx(fly * (1.05 + 0.01 * (4.0 - dep.d_ref)))


def test_generated_python_matches_the_deployment(dep, tmp_path):
    path = tmp_path / "shooter_aim.py"
    path.write_text(render_python(dep))
    mod = _load(path, "generated_shooter_aim")
    for d in (0.5, 1.5, 2.75, 4.2, 5.0, 7.0):
        hood, fly = dep.aim(d)
        assert mod.aim(d)[0] == pytest.approx(float(hood), abs=1e-9)
        assert mod.aim(d)[1] == pytest.approx(float(fly), abs=1e-9)
        assert mod.time_of_flight_s(d) == pytest.approx(float(dep.tof(dep.clamp(d))), abs=1e-9)
    mod.K_SPEED, mod.K_SPEED_SLOPE = 1.9, 0.02
    assert mod.flywheel_speed_mps(4.0) == pytest.approx(
        float(dep.aim(4.0, k_speed=1.9, k_speed_slope=0.02)[1]), abs=1e-9)


@pytest.mark.skipif(shutil.which("javac") is None, reason="no JDK on PATH")
def test_generated_java_compiles_and_agrees(dep, tmp_path):
    (tmp_path / "ShooterAim.java").write_text(render_java(dep))
    subprocess.run(["javac", "ShooterAim.java"], cwd=tmp_path, check=True,
                   capture_output=True)
    out = subprocess.run(["java", "-cp", ".", "ShooterAim"], cwd=tmp_path, check=True,
                         capture_output=True, text=True).stdout
    rows = re.findall(r"d=([\d.]+) m\s+->\s+hood\s+([\d.]+) deg\s+flywheel\s+([\d.]+) m/s"
                      r"\s+flight ([\d.]+) s", out)
    assert len(rows) == 5
    for d, hood, fly, tof in rows:
        want_hood, want_fly = dep.aim(float(d))
        assert float(hood) == pytest.approx(float(want_hood), abs=0.006)
        assert float(fly) == pytest.approx(float(want_fly), abs=0.006)
        assert float(tof) == pytest.approx(float(dep.tof(float(d))), abs=0.006)


def test_json_carries_coefficients_budget_and_validation(dep):
    doc = json.loads(to_json(dep))
    assert doc["hood_angle_deg"]["a"] == dep.hood.a
    assert doc["flywheel_speed_ms"]["c"] == dep.flywheel.c
    assert doc["error_budget"]["speed_sigma"] == 0.017
    assert doc["validation"]["max_abs_landing_offset_cm"] < 2.0
    assert len(doc["table"]) == 15 and doc["table"][0]["d"] == 1.5
    # Surface speed and ball speed differ by exactly the transfer ratio.
    assert doc["flywheel_speed_ms"]["b"] == pytest.approx(
        doc["v_ball_ms"]["b"] / doc["speed_transfer"], abs=2e-6)


# --- the artifacts checked into the repo must be what the generator produces ---

def test_repo_json_is_current(dep):
    want = to_json(dep)
    assert (ROOT / "lookup_polynomial.json").read_text() == want
    assert (ROOT / "handoff" / "lookup_polynomial.json").read_text() == want


def test_repo_handoff_sources_are_current(dep):
    assert (ROOT / "handoff" / "ShooterAim.java").read_text() == render_java(dep)
    assert (ROOT / "handoff" / "shooter_aim.py").read_text() == render_python(dep)


def test_simulator_embeds_the_current_function_and_handoff_copies_match(dep):
    sim = (ROOT / "shooter_sim.html").read_text()
    block = SIM_BLOCK.search(sim)
    assert block, "shooter_sim.html lost its /*DEPLOYMENT_BEGIN*/ ... /*DEPLOYMENT_END*/ block"
    assert block.group(2) == sim_block(dep)
    assert (ROOT / "handoff" / "shooter_sim.html").read_text() == sim
    assert ((ROOT / "handoff" / "research_report.html").read_bytes()
            == (ROOT / "research_report.html").read_bytes())
