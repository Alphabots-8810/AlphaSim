# FRC 2026 REBUILT — Shooter Aim Algorithm Handoff Package

Input: horizontal distance to the HUB → output: **hood angle** and **flywheel surface speed**.
At the field you calibrate the landing point with **one coefficient**, `kSpeed` — no
parameter-by-parameter retuning.

The algorithm comes from the *Alpha Sim* ballistics optimization pipeline (V3.1:
centered-landing + physics-calibrated cost, `w_tof=1.1` derived from vortex-shedding
dispersion). Full derivation in `research_report.html`.

---

## Files

| File | What it is |
|---|---|
| `ShooterAim.java` | The delivered function (WPILib / Java 17). Drops straight into robot code. |
| `shooter_aim.py` | Python version with the identical coefficients (RobotPy or offline validation). |
| `research_report.html` | Full research report: physics model → algorithm evolution → **how the formulas and coefficients were derived** (§5). |
| `shooter_sim.html` | Alpha Sim interactive simulator — open directly in a browser, drag the sliders, watch the hit rate. |
| `lookup_polynomial.json` | Machine-readable copy of the same coefficients (hood / v_ball / flywheel surface speed / reference RPM). |

---

## Quick start

Drop `ShooterAim.java` into your robot project, then:

```java
double d = distanceToHubMeters();          // your range source (vision/odometry)
ShooterAim.Aim a = ShooterAim.aimFor(d);
setHoodAngle(a.hoodAngleDeg());            // degrees, 90° = straight-down reference
setFlywheelSurfaceSpeed(a.flywheelSpeedMps());   // m/s, surface speed
```

Python:

```python
from shooter_aim import aim
hood_deg, flywheel_mps = aim(d)
```

**Valid range 1.5–5.0 m**; outside it the input is clamped to the boundary (no extrapolation —
the hood curve is U-shaped and would bend back upward past 5 m).

---

## The one field knob: `kSpeed`

`flywheel surface speed = kSpeed × fitted_curve(distance)`, default `kSpeed = 1.0`.

- Balls **falling short** → increase `kSpeed` (e.g. 1.03)
- Balls **overshooting** → decrease `kSpeed` (e.g. 0.97)

This single number absorbs **wheel–ball slip, ball compression, battery voltage, and air-model
error** all at once. **The hood angle gets no knob** — it follows the fitted curve, accurate to
±0.1°, well below servo/assembly error. Field calibration = shoot a few balls → nudge `kSpeed`
→ landing centered, **one parameter only**.

---

## Surface speed ↔ RPM

The primary output is **surface speed in m/s** because it is **wheel-diameter-independent**
(your new robot's wheel diameter may differ). Convert to RPM with your own wheel diameter `D` (m):

```
rpm = v / (Math.PI * D) * 60
```

The RPM numbers that appear in the report and the sim are the equivalent reference values for
this project's 0.1016 m (4") wheel (e.g. `d=3 m`: 8.70 m/s ↔ 1636 RPM) — the two sets of
numbers don't contradict each other.

The `v_exit` shown in Alpha Sim (ball exit speed, e.g. 7.14 m/s at `d=3 m`) is the function's
flywheel surface speed **× (1−0.18)**: `v_exit = flywheel_surface_speed × 0.82`, i.e.
7.14 = 8.70 × 0.82. The function outputs rim speed (slip included); the sim displays ball
speed — the same physical quantity seen from its two ends.

---

## Exit height (shooter height)

The coefficients were generated at an **exit height of 0.50 m** (8810 measured value), and that
height is **already baked into the polynomial coefficients** — the `SHOOTER_HEIGHT_M = 0.50`
in the code is provenance only; changing it at runtime **does nothing**.

If your measured robot height differs meaningfully, the coefficients must be **regenerated** —
that step runs in the full `frc-shooter-sim` simulation repo (this package ships without
`src/`, so the generator isn't included, to avoid the impression it can run standalone):

```bash
# in the frc-shooter-sim repo root:
python scripts/generate_coefficients.py --shooter-height 0.55
```

It prints the new a/b/c; paste them back into the two methods in `ShooterAim.java` /
`shooter_aim.py`. Ask 8810 for the repo, or just send them the height and have them generate.

---

## Report rendering note

`research_report.html` renders formulas with MathJax (CDN), so **the first open needs
internet**; otherwise formulas display as LaTeX source (content intact, just ugly). The
original `polyfill.io` shim was removed (that CDN was compromised in 2024, and MathJax 3
doesn't need it).

---

*Algorithm/report/simulator attribution is in the report footer. Questions go straight to 8810.*
