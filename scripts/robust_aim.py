#!/usr/bin/env python3
"""4414-style robust aim solver.

Usage:
    python scripts/robust_aim.py --distance 3.5
    python scripts/robust_aim.py -d 3.5 --plot
    python scripts/robust_aim.py --sweep            # sweep operating range
    python scripts/robust_aim.py -d 3.5 --compare   # robust vs heuristic
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_game, load_robot
from src.geometry.target import HexTarget
from src.optimizer.robust import solve_shot_robust
from src.optimizer.solve import solve_shot
from src.physics.shooter import DualRoller, Roller


def make_shooter(robot, game):
    return DualRoller(
        top=Roller(diameter_m=robot.flywheel.diameter_m,
                   gear_ratio=robot.flywheel.gear_ratio,
                   motor_free_rpm=robot.flywheel.motor_free_rpm,
                   slip_factor=robot.flywheel.slip_factor),
        bottom=Roller(diameter_m=robot.hood_roller.diameter_m,
                      gear_ratio=robot.hood_roller.gear_ratio,
                      motor_free_rpm=robot.hood_roller.motor_free_rpm,
                      slip_factor=robot.hood_roller.slip_factor),
        ball_radius_m=game.ball.radius_m,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--distance', '-d', type=float, default=None)
    p.add_argument('--unit', choices=['m', 'ft'], default='m')
    p.add_argument('--game-config', default=str(REPO_ROOT / 'config' / 'game_2026.yaml'))
    p.add_argument('--robot-config', default=str(REPO_ROOT / 'config' / 'robot.yaml'))
    p.add_argument('--grid-n', type=int, default=40, help='(v, θ) grid resolution')
    p.add_argument('--plot', action='store_true', help='plot valid region + robust center')
    p.add_argument('--sweep', action='store_true', help='sweep operating distance range')
    p.add_argument('--compare', action='store_true', help='also run heuristic and compare')
    args = p.parse_args()

    game = load_game(args.game_config)
    robot = load_robot(args.robot_config)
    target = HexTarget(across_flats_m=game.hub.target_across_flats_m,
                       height_m=game.hub.target_height_m)
    shooter = make_shooter(robot, game)

    def solve_r(d, seed=None):
        return solve_shot_robust(
            distance=d, exit_height=robot.shooter.exit_height_m,
            ball_mass=game.ball.mass_kg, ball_radius=game.ball.radius_m,
            drag_coefficient=game.ball.drag_coefficient,
            target=target, shooter=shooter,
            theta_range_deg=(robot.shooter.hood_angle_min_deg,
                             robot.shooter.hood_angle_max_deg),
            grid_n=args.grid_n,
            seed=seed,
            air_density=game.environment.air_density_kg_m3,
            gravity=game.environment.gravity_m_s2,
        )

    def solve_h(d):
        return solve_shot(
            distance=d, exit_height=robot.shooter.exit_height_m,
            ball_mass=game.ball.mass_kg, ball_radius=game.ball.radius_m,
            drag_coefficient=game.ball.drag_coefficient,
            target=target, shooter=shooter,
            theta_range_deg=(robot.shooter.hood_angle_min_deg,
                             robot.shooter.hood_angle_max_deg),
            w_entry_angle=robot.cost_weights.w_entry_angle,
            w_v_exit=robot.cost_weights.w_v_exit,
            w_tof=robot.cost_weights.w_tof,
            ball_clump_half_width_m=robot.shooter.ball_clump_half_width_m,
        )

    if args.sweep:
        import numpy as np
        rng = robot.operating_distance_m
        ds = np.arange(rng.min, rng.max + 1e-9, rng.step)
        print(f"{'d (m)':>6} {'hood °':>7} {'v_exit':>7} {'fly RPM':>8} "
              f"{'±Δv':>6} {'±Δθ°':>6} {'TOF':>5} {'entry°':>7}")
        prev_solution = None  # continuity seed for smooth curves
        for d in ds:
            s = solve_r(d, seed=prev_solution)
            if not s.success:
                print(f"{d:6.2f}   MISS")
                prev_solution = None
                continue
            print(f"{d:6.2f} {s.hood_angle_deg:7.2f} {s.v_exit_ms:7.2f} "
                  f"{s.flywheel_rpm:8.0f} {s.margin_v_ms:6.3f} "
                  f"{s.margin_theta_deg:6.2f} {s.time_of_flight_s:5.2f} "
                  f"{s.entry_angle_deg:7.2f}")
            prev_solution = (s.v_exit_ms, s.hood_angle_deg)
        return 0

    if args.distance is None:
        p.error("provide --distance or --sweep")
    distance_m = args.distance * (0.3048 if args.unit == 'ft' else 1.0)

    s = solve_r(distance_m)
    if not s.success:
        print(f"NO ROBUST SOLUTION at d={distance_m:.2f} m\n  {s.note}")
        return 1

    print(f"=== Robust shot for d = {distance_m:.2f} m ===")
    print(f"  Flywheel RPM:        {s.flywheel_rpm:8.1f}")
    print(f"  Hood roller RPM:     {s.hood_roller_rpm:8.1f}")
    print(f"  Hood angle:          {s.hood_angle_deg:8.2f}°")
    print(f"  Exit velocity:       {s.v_exit_ms:8.2f} m/s")
    print(f"  --- tolerance (still scores) ---")
    print(f"  ±Δv (at chosen θ):   {s.margin_v_ms:8.3f} m/s")
    print(f"  ±Δθ (at chosen v):   {s.margin_theta_deg:8.2f}°")
    print(f"  Combined boundary distance (grid units): {s.boundary_distance_grid:.2f}")
    print(f"  --- trajectory ---")
    print(f"  Entry angle:         {s.entry_angle_deg:8.2f}°")
    print(f"  Entry speed:         {s.entry_speed_ms:8.2f} m/s")
    print(f"  Time of flight:      {s.time_of_flight_s:8.3f} s")
    print(f"  Hit x:               {s.hit_x_m:8.3f} m (offset {s.x_offset_m:+.3f})")
    if s.note:
        print(f"  NOTE: {s.note}")

    if args.compare:
        h = solve_h(distance_m)
        print(f"\n=== Heuristic (weighted-cost) for d = {distance_m:.2f} m ===")
        print(f"  Hood angle:    {h.hood_angle_deg:7.2f}°  "
              f"(robust: {s.hood_angle_deg:.2f}°, diff {h.hood_angle_deg - s.hood_angle_deg:+.2f})")
        print(f"  v_exit:        {h.v_exit_ms:7.2f} m/s "
              f"(robust: {s.v_exit_ms:.2f}, diff {h.v_exit_ms - s.v_exit_ms:+.2f})")
        print(f"  Flywheel RPM:  {h.flywheel_rpm:7.0f}     "
              f"(robust: {s.flywheel_rpm:.0f}, diff {h.flywheel_rpm - s.flywheel_rpm:+.0f})")
        print(f"  TOF:           {h.time_of_flight_s:7.2f} s   "
              f"(robust: {s.time_of_flight_s:.2f})")

    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 6.5))
        # Green = valid
        ax.contourf(s.theta_grid, s.v_grid, s.valid_mask.astype(float),
                    levels=[0.5, 1.5], colors=['#90EE90'], alpha=0.55)
        ax.contour(s.theta_grid, s.v_grid, s.valid_mask.astype(float),
                   levels=[0.5], colors=['darkgreen'], linewidths=2)
        # Robust center marker + ± tolerance box
        ax.plot(s.hood_angle_deg, s.v_exit_ms, 'r*', ms=22, mec='black', mew=1.5,
                label=f'robust center: {s.hood_angle_deg:.1f}°, {s.v_exit_ms:.2f} m/s')
        ax.add_patch(plt.Rectangle(
            (s.hood_angle_deg - s.margin_theta_deg, s.v_exit_ms - s.margin_v_ms),
            2 * s.margin_theta_deg, 2 * s.margin_v_ms,
            fill=False, ec='red', lw=2, ls='--',
            label=f'tolerance box: ±{s.margin_theta_deg:.2f}°, ±{s.margin_v_ms:.2f} m/s'))
        if args.compare:
            h = solve_h(distance_m)
            ax.plot(h.hood_angle_deg, h.v_exit_ms, 'b^', ms=14, mec='black',
                    label=f'heuristic: {h.hood_angle_deg:.1f}°, {h.v_exit_ms:.2f} m/s')
        ax.set_xlabel('hood angle (°)')
        ax.set_ylabel('exit velocity (m/s)')
        ax.set_title(f'Valid Shot Region at d = {distance_m:.2f} m')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(REPO_ROOT / 'robust_valid_region.png', dpi=110)
        print(f"\nplot saved: robust_valid_region.png")
        plt.show()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
