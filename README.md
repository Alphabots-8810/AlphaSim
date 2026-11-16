# AlphaSim — FRC 2026 REBUILT shooter aim

Co-authored by **Claude Mythos 5.2** (Anthropic) and Yihan Zhu, for FRC Team 8810 Alphabots.

Distance in, aim out. AlphaSim works out the **launch angle and flywheel speed** that give a
dumper-style shooter the best chance of scoring from 1.5 to 5 m, and ships the answer as three
short formulas for robot code. One number, `kSpeed`, is tuned at the field.

**Live** (GitHub Pages): **https://alphabots-8810.github.io/AlphaSim/**

- [Simulator](https://alphabots-8810.github.io/AlphaSim/shooter_sim.html) — pick a distance, watch a
  100-ball burst, see where each ball lands and why that angle was chosen.
- [Report](https://alphabots-8810.github.io/AlphaSim/research_report.html) — the flight model, the
  error budget, the results, and what still needs measuring.

## Use it on a robot

Copy [`handoff/ShooterAim.java`](handoff/ShooterAim.java) (or
[`shooter_aim.py`](handoff/shooter_aim.py)) into your project:

```java
ShooterAim.Aim a = ShooterAim.aimFor(distanceMeters);
setHoodAngle(a.hoodAngleDeg());                    // degrees above horizontal
setFlywheelSurfaceSpeed(a.flywheelSpeedMps());     // m/s; rpm = v / (π · wheelDiameter) · 60
```

Inputs outside 1.5–5.0 m are clamped. At the field, tune `kSpeed` only: balls short → raise it,
balls long → lower it. Read [`handoff/README.md`](handoff/README.md) before the first shot; it
covers where `kSpeed` should start for your kind of shooter and what the angle means.

## How the aim is chosen

For each distance there is a whole family of shots that land on the hub centre, one speed per
launch angle. The solver (version 5) walks that family and, for each shot, works out how far the
landing point moves for each thing that varies from ball to ball — flywheel speed, launch angle,
heading, the distance estimate, turbulence in flight, where across the dumper the ball started —
and how much room the hexagonal opening leaves at that entry angle. That gives a probability of
scoring. It takes the angle with the best odds, then slides toward a slower flywheel until 3 % of
the safety margin is spent.

Every input is a physical quantity you can measure, listed in
[`config/robot.yaml`](config/robot.yaml) under `error_budget`. There are no tuning weights.

With the scatter currently assumed, about 99.7 % of balls score from 3 m and 96 % from 5 m, and
the far shots are limited by how repeatable the flywheel speed is. Those scatter values are
estimates; the report says how to measure them.

## Running from source

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/aim.py -d 3.5                    # one distance, with the error budget
python scripts/aim.py --sweep                   # the whole operating range
python scripts/aim.py -d 3.5 --monte-carlo 200000   # cross-check by simulation
python scripts/hood_sweep_hitrate.py -d 3       # every angle at one distance
python scripts/build_site.py                    # regenerate formulas, simulator, report
pytest -q
```

## Changing the robot

Everything shipped is generated. Edit [`config/robot.yaml`](config/robot.yaml) — exit height, hood
range, the error budget, how much margin to trade for speed — then run

```bash
python scripts/build_site.py
```

That re-solves, refits the formulas, grades them end to end, and rewrites `handoff/`, the
simulator and the report together. A test fails if any of them is out of date. The exit height is
baked into the formulas, so this is the only way to change it.

## Layout

```
src/        physics (batch integrator, reference integrator, launcher) · scoring (opening geometry,
            error budget) · optimizer (maxprob = V5; solve, robust = earlier versions) · deploy · montecarlo
scripts/    aim.py · hood_sweep_hitrate.py · generate_coefficients.py · build_site.py · build_report.py
web/        sources of the three pages; alphasim_core.js is the solver ported to JavaScript
config/     game_2026.yaml (hub and ball) · robot.yaml (robot, error budget, solver settings)
handoff/    what a robot programmer needs: the function in Java and Python, the JSON, both pages
tests/      pytest; includes Python-vs-JavaScript and repo-is-up-to-date checks
```

`shooter_sim.html`, `research_report.html` and `index.html` at the top level are built files, each
a single self-contained page with no network requests. Edit the sources in `web/`, not these.

## Earlier versions

The curve shipped before November 2026 (version 3.1) came from a weighted cost function. The new
solver lands within about 1° of it at every distance and scores it within 0.2 percentage points,
so robots already tuned on the old curve do not need retuning. The old solvers are kept in
`src/optimizer/` (`python scripts/aim.py --solver v31`, `scripts/robust_aim.py`).

## Credits

Co-authored by Claude Mythos 5.2 (Anthropic), which wrote the version 5 solver, the simulator and
the report, and by Yihan Zhu, who built AlphaSim for FRC Team 8810 Alphabots.

## License

MIT — see [LICENSE](LICENSE). © 2026 Alphabots — FRC Team 8810.
The pages embed the Barlow typeface, © The Barlow Project Authors, under the SIL Open Font
License 1.1 ([`web/fonts/OFL.txt`](web/fonts/OFL.txt)).
