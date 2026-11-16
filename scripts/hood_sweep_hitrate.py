#!/usr/bin/env python3
"""Audit one distance: walk every hood angle and show why the pick is the pick.

For each angle on the centred-landing family this prints the speed it needs,
the landing spread each error source produces, the room the opening leaves,
and the scoring probability - once from the analytic model the solver uses and
once from a Monte Carlo burst that makes no linear or normal assumption.

(This is the descendant of the tool that exposed the original V3 w_tof=200
low-arc mistake.)

Run:
    .venv/bin/python scripts/hood_sweep_hitrate.py --distance 3.0
    .venv/bin/python scripts/hood_sweep_hitrate.py -d 5 --n-balls 100000 --step 1
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from src.config import load_game, load_robot  # noqa: E402
from src.montecarlo import simulate_burst  # noqa: E402
from src.optimizer.maxprob import ShotModel, centered_family, choose_angle  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--distance', '-d', type=float, default=3.0)
    ap.add_argument('--step', type=float, default=2.5, help='hood angle step to print (deg)')
    ap.add_argument('--n-balls', type=int, default=20000, help='Monte Carlo balls per angle')
    ap.add_argument('--game-config', default=str(REPO / 'config' / 'game_2026.yaml'))
    ap.add_argument('--robot-config', default=str(REPO / 'config' / 'robot.yaml'))
    args = ap.parse_args()

    game, robot = load_game(args.game_config), load_robot(args.robot_config)
    model = ShotModel.from_configs(game, robot, hex_bearing_deg=robot.aim.hex_bearing_deg)
    budget, d = robot.error_budget, args.distance

    fine = np.arange(model.theta_min, model.theta_max + 1e-9, 0.25)
    fam = centered_family(model, budget, [d], fine)
    pick = choose_angle(fine, fam.speed[0], fam.margin[0], robot.aim.margin_concession)
    if pick is None:
        print(f"no descending shot reaches {d:.2f} m")
        return 1
    th_pick, th_best, m_best, th_slow, v_slow = pick

    print(f"Distance {d:.2f} m, exit {model.exit_height} m, opening at {model.target_height} m")
    print(f"Budget: {budget}\n")
    print(f"{'hood':>6} {'v':>6} {'flight':>6} {'entry':>6} {'room':>6} | "
          f"{'speed':>6} {'angle':>6} {'wake':>5} {'total':>6} | {'model':>8} {'Monte Carlo':>16}")
    print(f"{'deg':>6} {'m/s':>6} {'s':>6} {'deg':>6} {'cm':>6} | "
          f"{'cm':>6} {'cm':>6} {'cm':>5} {'cm':>6} | {'P %':>8} {'P % (+/-)':>16}")
    print("-" * 96)
    for th in np.arange(model.theta_min, model.theta_max + 1e-9, args.step):
        f = centered_family(model, budget, [d], [th])
        if not np.isfinite(f.speed[0, 0]):
            continue
        mc = simulate_burst(model, budget, d, float(th), float(f.speed[0, 0]), n=args.n_balls)
        print(f"{th:6.1f} {f.speed[0, 0]:6.2f} {f.tof[0, 0]:6.2f} {f.entry_angle[0, 0]:6.1f} "
              f"{f.half_width[0, 0] * 100:6.1f} | {f.spread.speed[0, 0] * 100:6.1f} "
              f"{f.spread.angle[0, 0] * 100:6.1f} {f.spread.vortex[0, 0] * 100:5.1f} "
              f"{f.spread.along[0, 0] * 100:6.1f} | {(1 - f.miss[0, 0]) * 100:8.3f} "
              f"{mc.p_score * 100:9.3f} ({mc.stderr * 100:.3f})")
    print()
    print(f"highest probability : {th_best:6.2f} deg  (sigma margin {m_best:.2f})")
    print(f"lowest speed        : {th_slow:6.2f} deg  ({v_slow:.3f} m/s)")
    print(f"V5 pick             : {th_pick:6.2f} deg  "
          f"(margin concession {robot.aim.margin_concession:g})")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
