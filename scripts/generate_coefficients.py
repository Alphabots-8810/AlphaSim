#!/usr/bin/env python3
"""Generate everything that ships to the robot from one solver sweep.

The shipped ShooterAim.java / shooter_aim.py are three quadratics in distance
(hood angle, flywheel surface speed, time of flight). The ball exit height is
baked into their coefficients, so it cannot be changed at runtime: to retarget
a different height, rerun this script.

What it does:
  1. Sweep the aim solver over the operating range (V5 by default).
  2. Fit the quadratics (hood first, then the speed that centres the ball at
     the fitted hood angle - see src/deploy.py).
  3. Grade the fitted function end to end and print the result.
  4. With --write, regenerate every shipped artifact in place:
       lookup_polynomial.json, handoff/lookup_polynomial.json,
       handoff/ShooterAim.java, handoff/shooter_aim.py,
       and the coefficient block embedded in shooter_sim.html.

Usage:
    .venv/bin/python scripts/generate_coefficients.py                # print only
    .venv/bin/python scripts/generate_coefficients.py --write        # regenerate artifacts
    .venv/bin/python scripts/generate_coefficients.py --shooter-height 0.55 --write
    .venv/bin/python scripts/generate_coefficients.py --concession 0 # maximum hit probability
    .venv/bin/python scripts/generate_coefficients.py --solver v31   # legacy weighted-cost curve
"""
import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_game, load_robot  # noqa: E402
from src.deploy import (Deployment, build_deployment, render_java,  # noqa: E402
                        render_python, to_json)
from src.geometry.target import HexTarget  # noqa: E402
from src.optimizer.maxprob import ShotModel  # noqa: E402
from src.physics.shooter import DualRoller, Roller  # noqa: E402

SIM_BLOCK = re.compile(r"(/\*DEPLOYMENT_BEGIN\*/)(.*?)(/\*DEPLOYMENT_END\*/)", re.S)


def deployment_from_configs(game, robot, exit_height=None, speed_transfer=None,
                            d_min=None, d_max=None, concession=None,
                            solver: str = "v5") -> Deployment:
    """The single entry point tests and the report builder share with the CLI."""
    model = ShotModel.from_configs(game, robot, exit_height=exit_height,
                                   hex_bearing_deg=robot.aim.hex_bearing_deg)
    legacy = None
    if solver == "v31":
        from src.optimizer.solve import solve_shot
        target = HexTarget(game.hub.target_across_flats_m, game.hub.target_height_m)
        shooter = DualRoller(
            top=Roller(robot.flywheel.diameter_m, robot.flywheel.gear_ratio,
                       robot.flywheel.motor_free_rpm, robot.flywheel.slip_factor),
            bottom=Roller(robot.hood_roller.diameter_m, robot.hood_roller.gear_ratio,
                          robot.hood_roller.motor_free_rpm, robot.hood_roller.slip_factor),
            ball_radius_m=game.ball.radius_m)

        def legacy(d):
            s = solve_shot(
                distance=d, exit_height=model.exit_height, ball_mass=game.ball.mass_kg,
                ball_radius=game.ball.radius_m, drag_coefficient=game.ball.drag_coefficient,
                target=target, shooter=shooter,
                theta_range_deg=(model.theta_min, model.theta_max),
                air_density=game.environment.air_density_kg_m3,
                gravity=game.environment.gravity_m_s2,
                w_entry_angle=robot.cost_weights.w_entry_angle,
                w_v_exit=robot.cost_weights.w_v_exit, w_tof=robot.cost_weights.w_tof)
            if not s.success:
                raise RuntimeError(f"V3.1 found no shot at {d:.2f} m: {s.note}")
            return s.hood_angle_deg

    model_info = dict(
        target_height_m=model.target_height, apothem_m=model.opening.apothem,
        ball_radius_m=game.ball.radius_m, ball_mass_kg=game.ball.mass_kg,
        drag_coefficient=game.ball.drag_coefficient, lift_coefficient=game.ball.magnus_cl,
        air_density=game.environment.air_density_kg_m3, gravity=game.environment.gravity_m_s2,
        theta_min=model.theta_min, theta_max=model.theta_max, v_max=round(model.v_max, 4),
        wheel_diameter_m=robot.flywheel.diameter_m, hex_bearing_deg=model.opening.bearing_deg,
    )
    return build_deployment(
        model, robot.error_budget,
        d_min=robot.operating_distance_m.min if d_min is None else d_min,
        d_max=robot.operating_distance_m.max if d_max is None else d_max,
        speed_transfer=(1.0 - robot.flywheel.slip_factor) if speed_transfer is None else speed_transfer,
        wheel_diameter_m=robot.flywheel.diameter_m,
        margin_concession=robot.aim.margin_concession if concession is None else concession,
        legacy_solver=legacy, model_info=model_info,
    )


def sim_block(dep: Deployment) -> str:
    """Compact JSON the simulator embeds: the shipped function + the model it came from."""
    doc = json.loads(to_json(dep))
    keep = ("solver", "shooter_exit_height_m", "speed_transfer", "distance_range_m",
            "k_speed_slope_pivot_m", "margin_concession", "error_budget", "model",
            "hood_angle_deg", "v_ball_ms", "flywheel_speed_ms", "time_of_flight_s")
    return json.dumps({k: doc[k] for k in keep}, separators=(",", ":"))


def write_artifacts(dep: Deployment, root: Path = REPO_ROOT) -> list[Path]:
    written = []

    def put(path: Path, text: str):
        path.write_text(text)
        written.append(path)

    put(root / "lookup_polynomial.json", to_json(dep))
    put(root / "handoff" / "lookup_polynomial.json", to_json(dep))
    put(root / "handoff" / "ShooterAim.java", render_java(dep))
    put(root / "handoff" / "shooter_aim.py", render_python(dep))
    sim = root / "shooter_sim.html"
    if sim.exists() and SIM_BLOCK.search(sim.read_text()):
        text = SIM_BLOCK.sub(lambda m: m.group(1) + sim_block(dep) + m.group(3), sim.read_text())
        put(sim, text)
        put(root / "handoff" / "shooter_sim.html", text)
    return written


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--solver', choices=['v5', 'v31'], default='v5',
                   help='v5 = maximum scoring probability (default); v31 = legacy weighted cost')
    p.add_argument('--shooter-height', type=float, default=None,
                   help='ball exit height above carpet (m). Default: robot.yaml shooter.exit_height_m')
    p.add_argument('--speed-transfer', type=float, default=None,
                   help='ball speed / flywheel surface speed. Default: 1 - flywheel slip_factor')
    p.add_argument('--slip', type=float, default=None,
                   help='(older spelling) wheel-to-ball slip fraction; same as --speed-transfer 1-slip')
    p.add_argument('--concession', type=float, default=None,
                   help='fraction of the best sigma margin to trade for lower speed. Default: robot.yaml aim')
    p.add_argument('--game-config', default=str(REPO_ROOT / 'config' / 'game_2026.yaml'))
    p.add_argument('--robot-config', default=str(REPO_ROOT / 'config' / 'robot.yaml'))
    p.add_argument('--dmin', type=float, default=None, help='min distance (m); default robot.yaml')
    p.add_argument('--dmax', type=float, default=None, help='max distance (m); default robot.yaml')
    p.add_argument('--json', dest='json_out', default=None,
                   help='also write the coefficient JSON to this path')
    p.add_argument('--write', action='store_true',
                   help='regenerate the shipped artifacts in the repo (JSON, Java, Python, simulator block)')
    args = p.parse_args()

    transfer = args.speed_transfer
    if transfer is None and args.slip is not None:
        transfer = 1.0 - args.slip
    game, robot = load_game(args.game_config), load_robot(args.robot_config)
    dep = deployment_from_configs(game, robot, args.shooter_height, transfer,
                                  args.dmin, args.dmax, args.concession, args.solver)

    v = dep.validation
    print(f"# Solver {dep.solver.upper()}, exit height {dep.exit_height_m:.3f} m, "
          f"speed transfer {dep.speed_transfer:.3f}, margin concession {dep.margin_concession:g}")
    print(f"# Distance range [{dep.d_min:.2f}, {dep.d_max:.2f}] m.  Form: y(d) = a*d^2 + b*d + c\n")
    print(f"{'output':<26} {'a':>12} {'b':>12} {'c':>12} {'fit RMSE':>10}")
    for label, q, unit in (("launch angle (deg)", dep.hood, "deg"),
                           ("ball exit speed (m/s)", dep.v_ball, "m/s"),
                           ("flywheel surface (m/s)", dep.flywheel, "m/s"),
                           ("time of flight (s)", dep.tof, "s")):
        print(f"{label:<26} {q.a:>12.6f} {q.b:>12.6f} {q.c:>12.6f} {q.rmse:>8.4f} {unit}")
    print("\n# End-to-end check of the fitted function "
          f"(every {v['grade_step_m']} m, full error budget):")
    print(f"#   landing offset from hub centre   max {v['max_abs_landing_offset_cm']:.2f} cm")
    print(f"#   scoring probability              {v['min_p_score']*100:.2f} % .. {v['max_p_score']*100:.3f} %")
    print(f"#   cost of fitting vs exact solver  max {v['max_p_loss_vs_solver_pp']:.3f} percentage points")

    print(f"\n{'d (m)':>6} {'hood':>7} {'v_ball':>7} {'flight':>7} {'entry':>6} {'apex':>6} {'P(score)':>9} {'margin':>7}")
    for r in dep.table:
        print(f"{r['d']:6.2f} {r['hood']:7.2f} {r['v']:7.3f} {r['tof']:7.2f} {r['entry']:6.1f} "
              f"{r['apex']:6.2f} {r['p_score']*100:8.3f}% {r['sigma_margin']:6.2f}s")

    if args.json_out:
        Path(args.json_out).write_text(to_json(dep))
        print(f"\n# wrote {args.json_out}")
    if args.write:
        for path in write_artifacts(dep):
            print(f"# wrote {path.relative_to(REPO_ROOT)}")
    else:
        print("\n# (dry run - add --write to regenerate the shipped artifacts)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
