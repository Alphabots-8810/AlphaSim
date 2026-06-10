#!/usr/bin/env python3
"""Generate the closed-form aim coefficients shipped to robot code.

This is the script that makes SHOOTER_HEIGHT_M a *real* knob: the shipped
ShooterAim.java / shooter_aim.py bake the ball exit height into their
polynomial coefficients, so you cannot change the height at runtime. To
retarget a different exit height, run this script and paste the new a/b/c.

What it does:
  1. Sweep the V3.1 solver (centered-landing + w_tof=1.1 cost) across the
     operating distance range at the given shooter (exit) height.
  2. Fit degree-2 polynomials  y(d) = a·d² + b·d + c  for:
       - hood angle (deg)
       - ball exit velocity v_ball (m/s)         [pure physics]
       - flywheel surface speed v_fly (m/s) = v_ball / (1 - slip)
  3. Print coefficients + fit RMSE, and emit ready-to-paste Java/Python.

Usage:
    .venv/bin/python scripts/generate_coefficients.py                       # robot.yaml exit height (0.50)
    .venv/bin/python scripts/generate_coefficients.py --shooter-height 0.55
    .venv/bin/python scripts/generate_coefficients.py --slip 0.15 --dstep 0.1
"""
import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src.config import load_game, load_robot  # noqa: E402
from src.geometry.target import HexTarget  # noqa: E402
from src.optimizer.solve import solve_shot  # noqa: E402
from src.physics.shooter import DualRoller, Roller  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--shooter-height', type=float, default=None,
                   help='ball exit height above carpet (m). Default: robot.yaml '
                        'shooter.exit_height_m — the single source of truth, so a bare '
                        'rerun reproduces the shipped coefficients instead of silently '
                        'reverting to an old height.')
    p.add_argument('--slip', type=float, default=None,
                   help='wheel→ball slip fraction for v_fly. Default: robot.yaml flywheel slip_factor.')
    p.add_argument('--game-config', default=str(REPO_ROOT / 'config' / 'game_2026.yaml'))
    p.add_argument('--robot-config', default=str(REPO_ROOT / 'config' / 'robot.yaml'))
    p.add_argument('--dmin', type=float, default=None, help='min distance (m); default robot.yaml')
    p.add_argument('--dmax', type=float, default=None, help='max distance (m); default robot.yaml')
    p.add_argument('--dstep', type=float, default=0.25, help='sweep step (m)')
    p.add_argument('--json', dest='json_out', default=None,
                   help='also write coefficients to this JSON path (e.g. lookup_polynomial.json)')
    args = p.parse_args()

    game = load_game(args.game_config)
    robot = load_robot(args.robot_config)

    dmin = args.dmin if args.dmin is not None else robot.operating_distance_m.min
    dmax = args.dmax if args.dmax is not None else robot.operating_distance_m.max
    slip = args.slip if args.slip is not None else robot.flywheel.slip_factor
    exit_h = (args.shooter_height if args.shooter_height is not None
              else robot.shooter.exit_height_m)

    target = HexTarget(across_flats_m=game.hub.target_across_flats_m,
                       height_m=game.hub.target_height_m)
    shooter = DualRoller(
        top=Roller(robot.flywheel.diameter_m, robot.flywheel.gear_ratio,
                   robot.flywheel.motor_free_rpm, robot.flywheel.slip_factor),
        bottom=Roller(robot.hood_roller.diameter_m, robot.hood_roller.gear_ratio,
                      robot.hood_roller.motor_free_rpm, robot.hood_roller.slip_factor),
        ball_radius_m=game.ball.radius_m,
    )

    ds, hoods, vballs = [], [], []
    for d in np.arange(dmin, dmax + 1e-9, args.dstep):
        s = solve_shot(
            distance=float(d), exit_height=exit_h,
            ball_mass=game.ball.mass_kg, ball_radius=game.ball.radius_m,
            drag_coefficient=game.ball.drag_coefficient, target=target, shooter=shooter,
            theta_range_deg=(robot.shooter.hood_angle_min_deg, robot.shooter.hood_angle_max_deg),
            air_density=game.environment.air_density_kg_m3, gravity=game.environment.gravity_m_s2,
            w_entry_angle=robot.cost_weights.w_entry_angle, w_v_exit=robot.cost_weights.w_v_exit,
            w_tof=robot.cost_weights.w_tof, ball_clump_half_width_m=robot.shooter.ball_clump_half_width_m,
        )
        if not s.success:
            print(f"  WARN: no feasible shot at d={d:.2f} m, skipping", file=sys.stderr)
            continue
        ds.append(float(d)); hoods.append(s.hood_angle_deg); vballs.append(s.v_exit_ms)

    ds = np.array(ds); hoods = np.array(hoods); vballs = np.array(vballs)
    vfly = vballs / (1.0 - slip)
    # Reference flywheel RPM at the project's wheel diameter (rim speed -> rpm).
    wheel_d = robot.flywheel.diameter_m
    vrpm = vfly / (math.pi * wheel_d) * 60.0

    def fit(y):
        a, b, c = np.polyfit(ds, y, 2)
        rmse = float(np.sqrt(np.mean((np.polyval([a, b, c], ds) - y) ** 2)))
        return a, b, c, rmse

    ha, hb, hc, h_rmse = fit(hoods)
    va, vb, vc, v_rmse = fit(vballs)
    fa, fb, fc, f_rmse = fit(vfly)
    ra, rb, rc, r_rmse = fit(vrpm)

    print(f"# Generated at shooter (exit) height = {exit_h:.3f} m, slip = {slip:.3f}")
    print(f"# Distance range [{dmin:.2f}, {dmax:.2f}] m, step {args.dstep} m, {len(ds)} points")
    print(f"# Form: y(d) = a*d^2 + b*d + c\n")
    print(f"{'output':<22} {'a':>14} {'b':>14} {'c':>14} {'RMSE':>10}")
    print(f"{'hood angle (deg)':<22} {ha:>14.6f} {hb:>14.6f} {hc:>14.6f} {h_rmse:>9.4f}°")
    print(f"{'v_ball (m/s)':<22} {va:>14.6f} {vb:>14.6f} {vc:>14.6f} {v_rmse:>8.4f}m/s")
    print(f"{'v_flywheel (m/s)':<22} {fa:>14.6f} {fb:>14.6f} {fc:>14.6f} {f_rmse:>8.4f}m/s")

    print("\n# ── paste into ShooterAim.java ──")
    print(f"    static final double SHOOTER_HEIGHT_M = {exit_h:.2f};  // coeffs below baked at this height")
    print(f"    public static double hoodAngleDeg(double d) {{")
    print(f"        return {ha:.6f} * d*d {hb:+.6f} * d {hc:+.6f};")
    print(f"    }}")
    print(f"    public static double flywheelSpeedMps(double d) {{")
    print(f"        return kSpeed * ({fa:.6f} * d*d {fb:+.6f} * d {fc:+.6f});")
    print(f"    }}")

    print("\n# ── paste into shooter_aim.py ──")
    print(f"SHOOTER_HEIGHT_M = {exit_h:.2f}   # coeffs below baked at this height")
    print(f"def aim(d):")
    print(f"    hood_deg     = {ha:.6f}*d*d {hb:+.6f}*d {hc:+.6f}")
    print(f"    flywheel_mps = K_SPEED * ({fa:.6f}*d*d {fb:+.6f}*d {fc:+.6f})")
    print(f"    return hood_deg, flywheel_mps")

    if args.json_out:
        doc = {
            "fit_form": "y(d) = a * d**2 + b * d + c   (d in meters)",
            "shooter_exit_height_m": round(exit_h, 4),
            "slip_factor": round(slip, 4),
            "distance_range_m": [round(dmin, 3), round(dmax, 3)],
            "source": ("V3.1 sweep (w_tof=1.1, physics-derived from Darbois-Texier "
                       "vortex shedding); generated by scripts/generate_coefficients.py"),
            "note": ("flywheel_speed_ms is the SURFACE (rim) linear speed shipped to robot "
                     "code = v_ball / (1 - slip_factor). flywheel_rpm is the same speed at "
                     "this project's wheel diameter (%.4f m), shown for reference only." % wheel_d),
            "hood_angle_deg": {"a": ha, "b": hb, "c": hc, "rmse_deg": h_rmse},
            "v_ball_ms":      {"a": va, "b": vb, "c": vc, "rmse_ms": v_rmse},
            "flywheel_speed_ms": {"a": fa, "b": fb, "c": fc, "rmse_ms": f_rmse},
            "flywheel_rpm":   {"a": ra, "b": rb, "c": rc, "rmse_rpm": r_rmse},
            "java_snippet": (
                "// FRC 2026 shooter lookup — see ShooterAim.java (kSpeed = field knob)\n"
                "double hood_deg     = %.6f * d*d %+.6f * d %+.6f;\n"
                "double flywheel_mps = kSpeed * (%.6f * d*d %+.6f * d %+.6f);"
                % (ha, hb, hc, fa, fb, fc)
            ),
        }
        Path(args.json_out).write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
        print(f"\n# wrote {args.json_out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
