"""Turn the solver's sweep into what actually ships to the robot.

The robot does not run the solver. It evaluates three quadratics in distance:
hood angle, flywheel surface speed, and time of flight. This module fits them,
grades the fitted function end to end (where does the ball really land, what
does the fit cost in hit probability), and renders every shipped artifact from
one `Deployment` object so the Java, the Python, the JSON and the simulator can
never drift apart.

Fit order matters. The hood polynomial is fitted first; the speed polynomial is
then fitted to the speed that centres the ball *at the fitted hood angle*, not
at the solver's exact angle. The speed curve therefore absorbs the hood fit's
residual, and the shipped function lands on centre to within the (much smaller)
speed-fit residual alone.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from src.optimizer.maxprob import ShotModel, aim_at_angle, evaluate_shot, sweep_aim
from src.physics.batch import simulate_batch, speed_for_distance
from src.scoring import ErrorBudget

COEFF_DECIMALS = 6      # what the generated Java / Python literally contains


@dataclass
class Quadratic:
    """y(d) = a d^2 + b d + c, coefficients already rounded to what ships."""
    a: float
    b: float
    c: float
    rmse: float = 0.0

    def __call__(self, d):
        d = np.asarray(d, dtype=float)
        return self.a * d * d + self.b * d + self.c

    @classmethod
    def fit(cls, d, y) -> "Quadratic":
        a, b, c = (round(float(k), COEFF_DECIMALS) for k in np.polyfit(d, y, 2))
        q = cls(a, b, c)
        q.rmse = float(np.sqrt(np.mean((q(d) - np.asarray(y)) ** 2)))
        return q

    def scaled(self, factor: float) -> "Quadratic":
        return Quadratic(round(self.a * factor, COEFF_DECIMALS),
                         round(self.b * factor, COEFF_DECIMALS),
                         round(self.c * factor, COEFF_DECIMALS), self.rmse * factor)


@dataclass
class Deployment:
    solver: str
    exit_height_m: float
    speed_transfer: float           # ball speed / flywheel surface speed
    wheel_diameter_m: float
    d_min: float
    d_max: float
    d_ref: float                    # pivot distance of the optional kSpeedSlope knob
    margin_concession: float
    error_budget: dict
    hood: Quadratic
    v_ball: Quadratic
    flywheel: Quadratic
    rpm: Quadratic
    tof: Quadratic
    table: list = field(default_factory=list)       # raw solver rows
    validation: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)       # the physical constants it was solved with

    def clamp(self, d):
        return np.clip(np.asarray(d, dtype=float), self.d_min, self.d_max)

    def aim(self, d, k_speed: float = 1.0, k_speed_slope: float = 0.0):
        """The shipped function: distance -> (hood deg, flywheel surface m/s)."""
        d = self.clamp(d)
        gain = k_speed + k_speed_slope * (d - self.d_ref)
        return self.hood(d), gain * self.flywheel(d)


def build_deployment(model: ShotModel, budget: ErrorBudget, d_min: float, d_max: float,
                     speed_transfer: float, wheel_diameter_m: float,
                     margin_concession: float = 0.03, fit_step: float = 0.1,
                     table_step: float = 0.25, grade_step: float = 0.05,
                     legacy_solver=None, model_info: Optional[dict] = None) -> Deployment:
    """Sweep, fit, grade.

    legacy_solver: optional callable distance -> hood angle (deg), used instead
    of the V5 solver for the hood samples (e.g. to reproduce the V3.1 curve).
    """
    def solve(ds):
        if legacy_solver is None:
            return sweep_aim(model, budget, ds, margin_concession)
        return [aim_at_angle(model, budget, float(d), legacy_solver(float(d))) for d in ds]

    def grid(step):
        return np.round(np.arange(d_min, d_max + 1e-9, step), 6)

    d_fit = grid(fit_step)
    sols = solve(d_fit)
    if not all(s.success for s in sols):
        bad = [s.distance_m for s in sols if not s.success]
        raise RuntimeError(f"no feasible shot at distances {bad}; narrow the range")
    hood = Quadratic.fit(d_fit, [s.hood_angle_deg for s in sols])

    # Speed that centres the ball at the *fitted* hood angle.
    h0, ht, aero = model.exit_height, model.target_height, model.aero
    v_centered = speed_for_distance(d_fit, hood(d_fit), h0, ht, aero, v_max=model.v_max)
    if not np.all(np.isfinite(v_centered)):
        raise RuntimeError("fitted hood curve leaves the feasible region; check the range")
    v_ball = Quadratic.fit(d_fit, v_centered)
    flywheel = v_ball.scaled(1.0 / speed_transfer)
    rpm = flywheel.scaled(60.0 / (math.pi * wheel_diameter_m))
    flight = simulate_batch(v_ball(d_fit), hood(d_fit), h0, ht, aero)
    tof = Quadratic.fit(d_fit, flight.t)

    # Grade the shipped (rounded) function on a finer grid than it was fitted on.
    d_grade = grid(grade_step)
    exact = solve(d_grade)
    rows = [evaluate_shot(model, budget, float(d), float(hood(d)), float(v_ball(d)))
            for d in d_grade]
    offsets = np.array([r["offset_m"] for r in rows])
    p_fit = np.array([r["p_score"] for r in rows])
    p_exact = np.array([getattr(s, "p_score", float("nan")) for s in exact])
    validation = dict(
        grade_step_m=grade_step,
        max_abs_landing_offset_cm=round(float(np.max(np.abs(offsets))) * 100, 3),
        min_p_score=round(float(p_fit.min()), 6),
        max_p_score=round(float(p_fit.max()), 6),
        max_p_loss_vs_solver_pp=round(float(np.nanmax(p_exact - p_fit)) * 100, 4),
        hood_fit_rmse_deg=round(hood.rmse, 4),
        hood_fit_max_abs_deg=round(float(np.max(np.abs(
            hood(d_fit) - np.array([s.hood_angle_deg for s in sols])))), 4),
        speed_fit_rmse_ms=round(v_ball.rmse, 5),
        tof_fit_rmse_s=round(tof.rmse, 5),
    )

    table = []
    for s in solve(grid(table_step)):
        table.append(dict(
            d=round(s.distance_m, 3), hood=round(s.hood_angle_deg, 3),
            v=round(s.v_exit_ms, 4), tof=round(s.time_of_flight_s, 4),
            entry=round(s.entry_angle_deg, 2), apex=round(s.apex_height_m, 3),
            p_score=round(getattr(s, "p_score", float("nan")), 6),
            sigma_margin=round(getattr(s, "sigma_margin", float("nan")), 3),
        ))

    return Deployment(
        solver="v5" if legacy_solver is None else "v3.1",
        exit_height_m=round(model.exit_height, 4), speed_transfer=round(speed_transfer, 4),
        wheel_diameter_m=wheel_diameter_m, d_min=d_min, d_max=d_max,
        d_ref=3.0 if d_min <= 3.0 <= d_max else round(0.5 * (d_min + d_max), 2),
        margin_concession=margin_concession, error_budget=asdict(budget),
        hood=hood, v_ball=v_ball, flywheel=flywheel, rpm=rpm, tof=tof,
        table=table, validation=validation, model=dict(model_info or {}),
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _poly(q: Quadratic, var: str = "d", sq: Optional[str] = None) -> str:
    sq = sq or f"{var}*{var}"
    return (f"{q.a:.{COEFF_DECIMALS}f} * {sq} "
            f"{'+' if q.b >= 0 else '-'} {abs(q.b):.{COEFF_DECIMALS}f} * {var} "
            f"{'+' if q.c >= 0 else '-'} {abs(q.c):.{COEFF_DECIMALS}f}")


def to_json(dep: Deployment) -> str:
    def q(x: Quadratic, unit_key: str):
        return {"a": x.a, "b": x.b, "c": x.c, unit_key: round(x.rmse, 6)}

    doc = {
        "fit_form": "y(d) = a * d**2 + b * d + c   (d in meters)",
        "solver": dep.solver,
        "shooter_exit_height_m": dep.exit_height_m,
        "slip_factor": round(1.0 - dep.speed_transfer, 4),
        "speed_transfer": dep.speed_transfer,
        "distance_range_m": [dep.d_min, dep.d_max],
        "k_speed_slope_pivot_m": dep.d_ref,
        "source": (
            "V5 sweep: maximum scoring probability under the error budget below, "
            f"margin concession {dep.margin_concession:g}; generated by "
            "scripts/generate_coefficients.py" if dep.solver == "v5" else
            "V3.1 sweep (legacy weighted cost, w_tof=1.1); generated by "
            "scripts/generate_coefficients.py --solver v31"),
        "note": (
            "flywheel_speed_ms is the SURFACE (rim) linear speed shipped to robot code "
            "= v_ball / speed_transfer. flywheel_rpm is the same speed at this project's "
            f"wheel diameter ({dep.wheel_diameter_m:.4f} m), shown for reference only. "
            "hood_angle_deg is the ball's launch angle above horizontal."),
        "margin_concession": dep.margin_concession,
        "error_budget": dep.error_budget,
        "model": dep.model,
        "hood_angle_deg": q(dep.hood, "rmse_deg"),
        "v_ball_ms": q(dep.v_ball, "rmse_ms"),
        "flywheel_speed_ms": q(dep.flywheel, "rmse_ms"),
        "flywheel_rpm": q(dep.rpm, "rmse_rpm"),
        "time_of_flight_s": q(dep.tof, "rmse_s"),
        "validation": dep.validation,
        "table": dep.table,
        "java_snippet": (
            "// FRC 2026 shooter lookup - see ShooterAim.java (kSpeed = field knob)\n"
            f"double hood_deg     = {_poly(dep.hood)};\n"
            f"double flywheel_mps = kSpeed * ({_poly(dep.flywheel)});"),
    }
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def render_python(dep: Deployment) -> str:
    return f'''"""FRC 2026 REBUILT - closed-form shooter aim lookup  (Alpha Sim, {dep.solver.upper()} fit).

distance d (m, ball exit point -> HUB centre)  ->  (launch angle deg, flywheel surface speed m/s)

GENERATED by scripts/generate_coefficients.py - do not edit the numbers by hand.

Field calibration is ONE number: K_SPEED. A second, optional number
(K_SPEED_SLOPE) is there if far shots drift after near shots are centred.

The exit height below is baked into the coefficients; changing the constant
does nothing at runtime. To retarget, rerun the generator with
--shooter-height and take the new file.
"""

SHOOTER_HEIGHT_M = {dep.exit_height_m:.2f}   # exit height the coeffs were generated at (provenance, NOT a runtime input)
D_MIN, D_MAX = {dep.d_min:g}, {dep.d_max:g}   # valid fit range (m); inputs outside this are clamped
D_REF = {dep.d_ref:g}               # pivot distance of K_SPEED_SLOPE (m)

K_SPEED = 1.00            # <- THE field-calibration knob. short -> raise, long -> lower.
                          #    Start near 1.0 with powered rollers on both sides of the ball,
                          #    near 2.0 if the hood is a fixed plate (the ball rolls on it).
K_SPEED_SLOPE = 0.00      # optional second knob: change in gain per metre from D_REF.
                          #    Leave 0 unless near shots are centred but far shots are not.


def _clamp(d):
    return max(D_MIN, min(D_MAX, d))


def hood_angle_deg(d):
    """Launch angle above horizontal (deg) for distance d (m). 90 = straight up."""
    d = _clamp(d)
    return {_poly(dep.hood, sq="d * d")}


def flywheel_speed_mps(d):
    """Flywheel surface (rim) linear speed (m/s). RPM = v / (pi * wheel_diameter_m) * 60."""
    d = _clamp(d)
    gain = K_SPEED + K_SPEED_SLOPE * (d - D_REF)
    return gain * ({_poly(dep.flywheel, sq="d * d")})


def time_of_flight_s(d):
    """Seconds from launch until the ball drops through the opening.
    For leading a moving shot: aim at where the hub will be relative to the
    robot after this long."""
    d = _clamp(d)
    return {_poly(dep.tof, sq="d * d")}


def aim(d):
    """distance d (m) -> (hood_angle_deg, flywheel_speed_mps)."""
    return hood_angle_deg(d), flywheel_speed_mps(d)


if __name__ == "__main__":
    for _d in (1.5, 2.0, 3.0, 4.0, 5.0):
        _h, _v = aim(_d)
        print(f"d={{_d:.1f}} m  ->  hood {{_h:6.2f}} deg   flywheel {{_v:6.2f}} m/s   "
              f"flight {{time_of_flight_s(_d):.2f}} s")
'''


def render_java(dep: Deployment) -> str:
    return f'''/*
 * FRC 2026 REBUILT - closed-form shooter aim lookup  (Alpha Sim, {dep.solver.upper()} fit)
 * ---------------------------------------------------------------------------
 * Input : horizontal distance d (m) from the ball's EXIT point to the HUB centre.
 * Output: launch angle (deg above horizontal) + flywheel SURFACE (rim) speed (m/s).
 *
 * GENERATED by scripts/generate_coefficients.py - do not edit the numbers by hand.
 *
 * Field calibration is ONE number: kSpeed. A second, optional number
 * (kSpeedSlope) is there if far shots drift after near shots are centred.
 *
 * The exit height below is BAKED INTO the coefficients - editing the constant
 * does nothing at runtime. To retarget, rerun the generator with
 * --shooter-height and take the new file.
 *
 * Java 17 (WPILib 2024+).  One call is a few multiply-adds - run it every loop.
 */
public final class ShooterAim {{

    /** Ball exit height above carpet (m) the coefficients were generated at.
     *  Provenance ONLY - not a runtime input. */
    public static final double SHOOTER_HEIGHT_M = {dep.exit_height_m:.2f};

    /** Valid distance range of the fit (m). Inputs outside it are clamped:
     *  the hood curve is a U and would turn back up if extrapolated. */
    public static final double D_MIN = {dep.d_min:g}, D_MAX = {dep.d_max:g};

    /** Pivot distance of kSpeedSlope (m). */
    public static final double D_REF = {dep.d_ref:g};

    // -- THE field-calibration knob ------------------------------------------
    //  Single gain on flywheel speed.
    //    balls landing SHORT -> raise  (e.g. 1.03)
    //    balls landing LONG  -> lower  (e.g. 0.97)
    //  Absorbs wheel slip, ball compression, battery voltage and air-model
    //  error in one number. Where to start:
    //    ~1.0  powered rollers on both sides of the ball, same surface speed
    //    ~2.0  flywheel on one side, fixed hood plate on the other (the ball
    //          rolls on the plate, so its centre moves at half the rim speed)
    public static double kSpeed = 1.00;

    // -- Optional second knob ------------------------------------------------
    //  Change in gain per metre away from D_REF. Leave at 0 unless shots near
    //  D_REF are centred but far (or close) shots are not; then
    //    far shots SHORT -> raise (e.g. +0.01),  far shots LONG -> lower.
    public static double kSpeedSlope = 0.00;

    /** Launch angle above horizontal (deg) for distance d (m). 90 = straight up.
     *  This is the ball's direction as it leaves, not a hood encoder reading:
     *  add your mechanism's own zero offset. */
    public static double hoodAngleDeg(double d) {{
        d = clamp(d);
        return {_poly(dep.hood)};
    }}

    /** Flywheel SURFACE (rim) linear speed (m/s) for distance d (m).
     *  To RPM with your wheel diameter D (m): rpm = v / (Math.PI * D) * 60. */
    public static double flywheelSpeedMps(double d) {{
        d = clamp(d);
        double gain = kSpeed + kSpeedSlope * (d - D_REF);
        return gain * ({_poly(dep.flywheel)});
    }}

    /** Seconds from launch until the ball drops through the opening. For
     *  leading a moving shot: aim at where the hub will be relative to the
     *  robot after this long. */
    public static double timeOfFlightS(double d) {{
        d = clamp(d);
        return {_poly(dep.tof)};
    }}

    /** Convenience: both outputs at once. */
    public static Aim aimFor(double d) {{
        return new Aim(hoodAngleDeg(d), flywheelSpeedMps(d));
    }}

    /** (hoodAngleDeg, flywheelSpeedMps) */
    public record Aim(double hoodAngleDeg, double flywheelSpeedMps) {{}}

    private static double clamp(double d) {{ return Math.max(D_MIN, Math.min(D_MAX, d)); }}

    private ShooterAim() {{}}

    // Quick sanity demo:  javac ShooterAim.java && java ShooterAim
    public static void main(String[] args) {{
        for (double d : new double[]{{1.5, 2.0, 3.0, 4.0, 5.0}}) {{
            Aim a = aimFor(d);
            System.out.printf("d=%.1f m  ->  hood %6.2f deg   flywheel %6.2f m/s   flight %.2f s%n",
                              d, a.hoodAngleDeg(), a.flywheelSpeedMps(), timeOfFlightS(d));
        }}
    }}
}}
'''
