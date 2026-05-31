#!/usr/bin/env python3
"""FRC 2026 shooter aim solver.

Usage:
    python scripts/aim.py --distance 3.5
    python scripts/aim.py -d 12 --unit ft --plot
"""
import argparse
import sys
from pathlib import Path

# Make src importable when running this script directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_game, load_robot  # noqa: E402
from src.geometry.target import HexTarget  # noqa: E402
from src.optimizer.solve import solve_shot  # noqa: E402
from src.physics.shooter import DualRoller, Roller  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="FRC 2026 shooter aim solver")
    p.add_argument('--distance', '-d', type=float, required=True,
                   help='horizontal distance from shooter exit to hub center')
    p.add_argument('--unit', choices=['m', 'ft'], default='m')
    p.add_argument('--game-config', default=str(REPO_ROOT / 'config' / 'game_2026.yaml'))
    p.add_argument('--robot-config', default=str(REPO_ROOT / 'config' / 'robot.yaml'))
    p.add_argument('--plot', action='store_true', help='show trajectory plot')
    p.add_argument('--sweep', action='store_true',
                   help='sweep operating distance range instead of single shot')
    args = p.parse_args()

    distance_m = args.distance * (0.3048 if args.unit == 'ft' else 1.0)

    game = load_game(args.game_config)
    robot = load_robot(args.robot_config)

    target = HexTarget(
        across_flats_m=game.hub.target_across_flats_m,
        height_m=game.hub.target_height_m,
    )
    shooter = DualRoller(
        top=Roller(
            diameter_m=robot.flywheel.diameter_m,
            gear_ratio=robot.flywheel.gear_ratio,
            motor_free_rpm=robot.flywheel.motor_free_rpm,
            slip_factor=robot.flywheel.slip_factor,
        ),
        bottom=Roller(
            diameter_m=robot.hood_roller.diameter_m,
            gear_ratio=robot.hood_roller.gear_ratio,
            motor_free_rpm=robot.hood_roller.motor_free_rpm,
            slip_factor=robot.hood_roller.slip_factor,
        ),
        ball_radius_m=game.ball.radius_m,
    )

    def solve(d):
        return solve_shot(
            distance=d,
            exit_height=robot.shooter.exit_height_m,
            ball_mass=game.ball.mass_kg,
            ball_radius=game.ball.radius_m,
            drag_coefficient=game.ball.drag_coefficient,
            target=target, shooter=shooter,
            theta_range_deg=(robot.shooter.hood_angle_min_deg, robot.shooter.hood_angle_max_deg),
            air_density=game.environment.air_density_kg_m3,
            gravity=game.environment.gravity_m_s2,
            w_entry_angle=robot.cost_weights.w_entry_angle,
            w_v_exit=robot.cost_weights.w_v_exit,
            w_tof=robot.cost_weights.w_tof,
            ball_clump_half_width_m=robot.shooter.ball_clump_half_width_m,
        )

    if args.sweep:
        import numpy as np
        rng = robot.operating_distance_m
        ds = np.arange(rng.min, rng.max + 1e-9, rng.step)
        hdr = (f"{'dist (m)':>9} {'fly RPM':>9} {'hood RPM':>9} {'hood °':>7} "
               f"{'v_exit':>7} {'entry °':>8} {'tof (s)':>7} "
               f"{'margin':>8} {'note':<30}")
        print(hdr)
        for d in ds:
            sol = solve(d)
            if not sol.success:
                msg = sol.note if sol.note else "no feasible"
                print(f"{d:9.2f}   MISS    ({msg})")
                continue
            print(f"{d:9.2f} {sol.flywheel_rpm:9.0f} {sol.hood_roller_rpm:9.0f} "
                  f"{sol.hood_angle_deg:7.2f} {sol.v_exit_ms:7.2f} "
                  f"{sol.entry_angle_deg:8.2f} "
                  f"{sol.time_of_flight_s:7.3f} {sol.boundary_margin_m:+8.3f} "
                  f"{sol.note:<30}")
        return 0

    sol = solve(distance_m)
    if not sol.success:
        print(f"NO FEASIBLE SHOT for distance {distance_m:.2f} m")
        print(f"  {sol.note}")
        return 1

    print(f"=== Shot for distance {distance_m:.3f} m ({distance_m / 0.3048:.2f} ft) ===")
    print(f"  Flywheel RPM:     {sol.flywheel_rpm:8.1f}")
    print(f"  Hood roller RPM:  {sol.hood_roller_rpm:8.1f}")
    print(f"  Hood angle:       {sol.hood_angle_deg:8.2f}°")
    print(f"  Exit velocity:    {sol.v_exit_ms:8.2f} m/s")
    print(f"  Entry angle:      {sol.entry_angle_deg:8.2f}° (90° = vertical down)")
    print(f"  Entry speed:      {sol.entry_speed_ms:8.2f} m/s")
    print(f"  Time of flight:   {sol.time_of_flight_s:8.3f} s")
    print(f"  Predicted hit x:  {sol.hit_x_m:8.3f} m  (offset {sol.x_offset_m:+.3f} m)")
    print(f"  4-ball margin:    {sol.boundary_margin_m:+.3f} m  "
          f"(>0 = all 4 balls fit; clump half-width assumed "
          f"{robot.shooter.ball_clump_half_width_m:.2f} m)")
    if sol.note:
        print(f"  NOTE: {sol.note}")

    if args.plot:
        import matplotlib.pyplot as plt
        from src.viz.plot import plot_trajectory
        plot_trajectory(
            sol.trajectory, target=target, distance=distance_m,
            exit_height=robot.shooter.exit_height_m,
            ball_radius=game.ball.radius_m,
        )
        plt.title(f"d={distance_m:.2f} m → "
                  f"{sol.flywheel_rpm:.0f}/{sol.hood_roller_rpm:.0f} RPM, "
                  f"{sol.hood_angle_deg:.1f}°")
        plt.show()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
