#!/usr/bin/env python3
"""FRC 2026 shooter aim solver.

Usage:
    python scripts/aim.py -d 3.5                  # V5: best shot + why
    python scripts/aim.py -d 12 --unit ft
    python scripts/aim.py --sweep                 # whole operating range
    python scripts/aim.py -d 3.5 --monte-carlo 200000   # cross-check by simulation
    python scripts/aim.py -d 3.5 --concession 0   # maximum hit probability
    python scripts/aim.py -d 3.5 --solver v31     # legacy weighted-cost solver
"""
import argparse
import sys
from pathlib import Path

# Make src importable when running this script directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src.config import load_game, load_robot  # noqa: E402
from src.geometry.target import HexTarget  # noqa: E402
from src.optimizer.maxprob import ShotModel, sweep_aim  # noqa: E402
from src.physics.shooter import DualRoller, Roller  # noqa: E402


def _shooter(robot, game) -> DualRoller:
    return DualRoller(
        top=Roller(robot.flywheel.diameter_m, robot.flywheel.gear_ratio,
                   robot.flywheel.motor_free_rpm, robot.flywheel.slip_factor),
        bottom=Roller(robot.hood_roller.diameter_m, robot.hood_roller.gear_ratio,
                      robot.hood_roller.motor_free_rpm, robot.hood_roller.slip_factor),
        ball_radius_m=game.ball.radius_m,
    )


def _legacy(args, game, robot, distances) -> int:
    """The V3.1 weighted-cost solver, output unchanged from earlier releases."""
    from src.optimizer.solve import solve_shot
    target = HexTarget(game.hub.target_across_flats_m, game.hub.target_height_m)
    shooter = _shooter(robot, game)
    print(f"{'dist (m)':>9} {'fly RPM':>9} {'hood RPM':>9} {'hood deg':>8} "
          f"{'v_exit':>7} {'entry':>7} {'tof (s)':>7}  note")
    for d in distances:
        s = solve_shot(
            distance=float(d), exit_height=robot.shooter.exit_height_m,
            ball_mass=game.ball.mass_kg, ball_radius=game.ball.radius_m,
            drag_coefficient=game.ball.drag_coefficient, target=target, shooter=shooter,
            theta_range_deg=(robot.shooter.hood_angle_min_deg, robot.shooter.hood_angle_max_deg),
            air_density=game.environment.air_density_kg_m3,
            gravity=game.environment.gravity_m_s2,
            w_entry_angle=robot.cost_weights.w_entry_angle,
            w_v_exit=robot.cost_weights.w_v_exit, w_tof=robot.cost_weights.w_tof,
            ball_clump_half_width_m=robot.shooter.ball_clump_half_width_m)
        if not s.success:
            print(f"{d:9.2f}   no feasible shot ({s.note})")
            continue
        print(f"{d:9.2f} {s.flywheel_rpm:9.0f} {s.hood_roller_rpm:9.0f} {s.hood_angle_deg:8.2f} "
              f"{s.v_exit_ms:7.2f} {s.entry_angle_deg:7.2f} {s.time_of_flight_s:7.3f}  {s.note}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="FRC 2026 shooter aim solver")
    p.add_argument('--distance', '-d', type=float, default=None,
                   help='horizontal distance from shooter exit to hub centre')
    p.add_argument('--unit', choices=['m', 'ft'], default='m')
    p.add_argument('--sweep', action='store_true', help='solve the whole operating range')
    p.add_argument('--solver', choices=['v5', 'v31'], default='v5')
    p.add_argument('--concession', type=float, default=None,
                   help='fraction of the best sigma margin to trade for lower speed '
                        '(default: robot.yaml aim.margin_concession)')
    p.add_argument('--monte-carlo', type=int, default=0, metavar='N',
                   help='also fire N simulated balls and report the measured hit rate')
    p.add_argument('--game-config', default=str(REPO_ROOT / 'config' / 'game_2026.yaml'))
    p.add_argument('--robot-config', default=str(REPO_ROOT / 'config' / 'robot.yaml'))
    p.add_argument('--plot', action='store_true', help='show the trajectory')
    args = p.parse_args()
    if args.distance is None and not args.sweep:
        p.error("provide --distance or --sweep")

    game, robot = load_game(args.game_config), load_robot(args.robot_config)
    rng = robot.operating_distance_m
    distances = (np.arange(rng.min, rng.max + 1e-9, rng.step) if args.sweep
                 else [args.distance * (0.3048 if args.unit == 'ft' else 1.0)])
    if args.solver == 'v31':
        return _legacy(args, game, robot, distances)

    model = ShotModel.from_configs(game, robot, hex_bearing_deg=robot.aim.hex_bearing_deg)
    budget = robot.error_budget
    concession = robot.aim.margin_concession if args.concession is None else args.concession
    sols = sweep_aim(model, budget, distances, concession)
    roller = _shooter(robot, game).top

    if args.sweep:
        print(f"{'d (m)':>6} {'hood':>7} {'v_ball':>7} {'fly m/s':>8} {'fly RPM':>8} {'flight':>7} "
              f"{'entry':>6} {'apex':>6} {'P(score)':>9} {'margin':>7}")
        for s in sols:
            if not s.success:
                print(f"{s.distance_m:6.2f}   no feasible shot ({s.note})")
                continue
            print(f"{s.distance_m:6.2f} {s.hood_angle_deg:7.2f} {s.v_exit_ms:7.3f} "
                  f"{s.v_exit_ms / (1 - roller.slip_factor):8.3f} "
                  f"{roller.rpm_for_surface_speed(s.v_exit_ms):8.0f} {s.time_of_flight_s:7.2f} "
                  f"{s.entry_angle_deg:6.1f} {s.apex_height_m:6.2f} {s.p_score * 100:8.3f}% "
                  f"{s.sigma_margin:6.2f}s")
        return 0

    s = sols[0]
    if not s.success:
        print(f"NO FEASIBLE SHOT at {s.distance_m:.2f} m\n  {s.note}")
        return 1
    d = s.distance_m
    print(f"=== Best shot at {d:.3f} m ({d / 0.3048:.2f} ft) - V5, margin concession {concession:g} ===")
    print(f"  Launch angle          {s.hood_angle_deg:8.2f} deg above horizontal")
    print(f"  Ball exit speed       {s.v_exit_ms:8.3f} m/s")
    print(f"  Flywheel surface      {s.v_exit_ms / (1 - roller.slip_factor):8.3f} m/s  "
          f"({roller.rpm_for_surface_speed(s.v_exit_ms):.0f} RPM at a "
          f"{roller.diameter_m * 1000:.1f} mm wheel)")
    print(f"  Time of flight        {s.time_of_flight_s:8.3f} s")
    print(f"  Entry angle           {s.entry_angle_deg:8.2f} deg  (90 = straight down)")
    print(f"  Apex height           {s.apex_height_m:8.2f} m")
    print("  --- how likely it scores ---")
    print(f"  Scoring probability   {s.p_score * 100:8.3f} %   (sigma margin {s.sigma_margin:.2f})")
    print(f"  Room along the shot   +/-{s.half_width_m * 100:5.1f} cm at this entry angle")
    print("  --- landing spread along the shot, 1 sigma, by source ---")
    for label, val in (("flywheel speed", s.spread_speed_m), ("launch angle", s.spread_angle_m),
                       ("wake turbulence", s.spread_vortex_m), ("range estimate", s.spread_range_m)):
        print(f"    {label:<18} {val * 100:6.1f} cm")
    print(f"    {'total':<18} {s.spread_along_m * 100:6.1f} cm   "
          f"(across the shot: {s.spread_lateral_m * 100:.1f} cm + dumper width)")
    print("  --- context ---")
    print(f"  Highest-probability angle {s.best_angle_deg:6.2f} deg (sigma margin {s.best_sigma_margin:.2f})")
    print(f"  Lowest-speed angle        {s.min_speed_angle_deg:6.2f} deg ({s.min_speed_ms:.3f} m/s)")

    if args.monte_carlo:
        from src.montecarlo import simulate_burst
        mc = simulate_burst(model, budget, d, s.hood_angle_deg, s.v_exit_ms, n=args.monte_carlo)
        print(f"  --- Monte Carlo, {mc.n:,} balls ---")
        print(f"  Hit rate              {mc.p_score * 100:8.3f} %  +/- {mc.stderr * 100:.3f}")
        print(f"  Landing spread        {mc.along_std * 100:6.1f} cm along, "
              f"{mc.lateral_std * 100:.1f} cm across")

    if args.plot:
        import matplotlib.pyplot as plt
        from src.physics.trajectory import simulate_trajectory
        from src.viz.plot import plot_trajectory
        traj = simulate_trajectory(
            s.v_exit_ms, s.hood_angle_deg, model.exit_height, model.target_height,
            game.ball.mass_kg, game.ball.radius_m, game.ball.drag_coefficient,
            air_density=game.environment.air_density_kg_m3,
            gravity=game.environment.gravity_m_s2)
        plot_trajectory(traj, target=HexTarget(game.hub.target_across_flats_m,
                                               game.hub.target_height_m),
                        distance=d, exit_height=model.exit_height,
                        ball_radius=game.ball.radius_m)
        plt.title(f"d = {d:.2f} m: {s.hood_angle_deg:.1f} deg, {s.v_exit_ms:.2f} m/s, "
                  f"P(score) {s.p_score * 100:.2f} %")
        plt.show()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
