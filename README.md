# AlphaSim — FRC 2026 REBUILT Shooter Aim

Ballistics physics + optimization pipeline that solves
**distance → hood angle + flywheel surface speed** for an FRC 2026 big-dumper-style shooter.
The final deliverable is a closed-form function: robot code feeds in the distance to the HUB
and gets the hood angle and flywheel speed back; at competition you calibrate the landing
point with **a single coefficient**, `kSpeed`.

> Exit height calibrated at **0.50 m** (V3.1: centered-landing + physics-calibrated cost,
> `w_tof=1.1` derived from vortex-shedding dispersion).

**🔗 Live** (GitHub Pages): report and the Alpha Sim simulator → **https://alphabots-8810.github.io/AlphaSim/**

## Use it directly

Easiest path: copy [`handoff/ShooterAim.java`](handoff/ShooterAim.java) (or [`shooter_aim.py`](handoff/shooter_aim.py)) into your robot project.

```java
ShooterAim.Aim a = ShooterAim.aimFor(distanceMeters);
setHoodAngle(a.hoodAngleDeg());            // degrees
setFlywheelSurfaceSpeed(a.flywheelSpeedMps());   // m/s surface speed (to RPM: v/(π·D)·60)
```

Valid range 1.5–5.0 m (clamped automatically outside it). Field calibration touches only `kSpeed`:
falling short → increase it; overshooting → decrease it.
Full handoff notes in [`handoff/README.md`](handoff/README.md).

## Docs and tools

- **Research report** [`research_report.html`](research_report.html) — physics model → algorithm evolution (V1→V4r2) → how the formulas/coefficients were derived (§5).
- **Alpha Sim** [`shooter_sim.html`](shooter_sim.html) — interactive in-browser simulator; drag the sliders and watch the 100-ball burst hit rate.
- **Coefficient table** [`lookup_polynomial.json`](lookup_polynomial.json) — machine-readable fitted coefficients (hood / v_ball / flywheel surface speed / reference RPM).

## Running from source

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/aim.py -d 3.5                      # single-distance solve
python scripts/aim.py -d 0 --sweep                # full-distance sweep
python scripts/robust_aim.py -d 3.5 --compare     # robust vs heuristic
python scripts/generate_coefficients.py --shooter-height 0.50 --json lookup_polynomial.json
pytest -q                                         # 28 tests
```

## Layout

```
src/        physics (trajectory/shooter) · geometry (target) · optimizer (solve/robust) · viz (plot)
scripts/    aim.py · robust_aim.py · hood_sweep_hitrate.py · generate_coefficients.py
config/     game_2026.yaml (HUB/FUEL parameters) · robot.yaml (robot geometry / cost weights)
handoff/    minimal package delivered to the shooter programmer (function + report + sim + json + readme)
tests/      28 pytest
```

## Changing the exit height

The coefficients are baked into the polynomials at 0.50 m. If your real robot's height differs
meaningfully, regenerate and paste back into `ShooterAim.java`:

```bash
python scripts/generate_coefficients.py --shooter-height <new height>
```

## License

MIT — see [LICENSE](LICENSE). © 2026 Alphabots — FRC Team 8810.

---
FRC Team 8810 · Alphabots
