#!/usr/bin/env python3
"""Sweep hood angle at fixed distance, evaluate 100-ball burst hit rate.

Reproduces Alpha Sim's burst physics (Cd=0.5, exit=0.50m, ±3%v, ±1°θ, seed=42)
to check whether the shipped V3.1 table's (hood, v) is hit-rate-optimal under
mechanical noise. (This is the audit tool that originally exposed the V3
w_tof=200 low-arc mistake; constants/table track shooter_sim.html.)

Run:
    .venv/bin/python scripts/hood_sweep_hitrate.py --distance 3.0
"""
import argparse
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
from scipy.optimize import brentq

from src.physics.trajectory import simulate_trajectory


# Match Alpha Sim constants exactly (shooter_sim.html: HUB/BALL consts + EXIT_H)
EXIT_H = 0.50
TARGET_H = 1.829
CD = 0.50
RHO = 1.225
G = 9.81
BALL_R = 0.075
BALL_M = 0.215
APOTHEM = 0.530
EFF_HALF = APOTHEM - BALL_R  # 0.455m — sim's "in hex" criterion


def mulberry32(seed):
    a = [seed & 0xFFFFFFFF]
    def rng():
        a[0] = (a[0] + 0x6D2B79F5) & 0xFFFFFFFF
        t = a[0]
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + ((t ^ (t >> 7)) * (t | 61))) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0
    return rng


def v_centered(d, hood_deg):
    """Find v ∈ [0.5, 30] s.t. ball hits at x=d for this hood. Returns None if infeasible."""
    th = math.radians(hood_deg)
    if math.sin(th) < 0.02 or math.cos(th) < 0.02:
        return None
    dh = TARGET_H - EXIT_H
    denom = d * math.tan(th) - dh
    if denom <= 0:
        return None
    v_seed = math.sqrt(0.5 * G * d * d / (math.cos(th) ** 2 * denom))

    def offset(v):
        r = simulate_trajectory(v, hood_deg, EXIT_H, TARGET_H,
                                BALL_M, BALL_R, CD, RHO, G)
        return (r.hit_x - d) if r.hit else -d

    lo = max(0.5, v_seed * 0.85)
    hi = v_seed * 2.5
    f_lo, f_hi = offset(lo), offset(hi)
    for _ in range(8):
        if f_hi >= 0: break
        hi *= 1.4
        f_hi = offset(hi)
    if f_hi < 0: return None
    for _ in range(6):
        if f_lo <= 0: break
        lo *= 0.7
        f_lo = offset(lo)
    if f_lo > 0: return None
    try:
        return brentq(offset, lo, hi, xtol=1e-4, maxiter=40)
    except (ValueError, RuntimeError):
        return None


def burst_hit_rate(d, hood_deg, v_center, N=100, v_jit=0.03, ang_jit=1.0, seed=42):
    """Fire N balls with v jitter ±v_jit*v, angle jitter ±ang_jit deg. Match sim seed."""
    rng = mulberry32(seed)
    hits_x = []
    inside = 0
    outside = 0
    no_hit = 0
    tofs = []
    for _ in range(N):
        v_j = (rng() - 0.5) * 2 * v_jit
        a_j = (rng() - 0.5) * 2 * ang_jit
        v = v_center * (1.0 + v_j)
        ang = hood_deg + a_j
        r = simulate_trajectory(v, ang, EXIT_H, TARGET_H,
                                BALL_M, BALL_R, CD, RHO, G)
        if not r.hit:
            no_hit += 1
            continue
        hits_x.append(r.hit_x)
        tofs.append(r.hit_t)
        if abs(r.hit_x - d) <= EFF_HALF:
            inside += 1
        else:
            outside += 1
    total = inside + outside
    rate = inside / total if total > 0 else 0.0
    spread = (max(hits_x) - min(hits_x)) if len(hits_x) > 1 else 0.0
    mean_tof = float(np.mean(tofs)) if tofs else 0.0
    return rate, spread, mean_tof, no_hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--distance', '-d', type=float, default=3.0)
    ap.add_argument('--hood-min', type=float, default=25.0)
    ap.add_argument('--hood-max', type=float, default=80.0)
    ap.add_argument('--n-hood', type=int, default=23)
    ap.add_argument('--n-balls', type=int, default=100)
    args = ap.parse_args()

    d = args.distance
    print(f"Distance = {d:.2f} m, exit = {EXIT_H} m, target = {TARGET_H} m")
    print(f"100-ball burst: ±3% v, ±1° θ, seed=42, eff_half = ±{EFF_HALF:.3f} m")
    print()

    hoods = np.linspace(args.hood_min, args.hood_max, args.n_hood)
    print(f"{'hood':>7} {'v_ctr':>7} {'TOF':>6} {'hit %':>7} "
          f"{'spread':>7} {'no_hit':>6}")
    print("-" * 50)
    best_rate = -1
    best_h = None
    best_v = None
    rows = []
    for h in hoods:
        v = v_centered(d, float(h))
        if v is None:
            print(f"{h:7.2f}    n/a    --     --      --      --")
            continue
        rate, spread, tof, no_hit = burst_hit_rate(d, float(h), v, N=args.n_balls)
        rows.append((h, v, tof, rate, spread, no_hit))
        marker = "  <-- best so far" if rate > best_rate else ""
        if rate > best_rate:
            best_rate = rate
            best_h = float(h)
            best_v = float(v)
        print(f"{h:7.2f} {v:7.3f} {tof:6.3f} {rate*100:6.1f}% "
              f"{spread:7.3f} {no_hit:6d}{marker}")
    print()
    print(f"BEST in sweep:    hood = {best_h:.2f}°, v = {best_v:.3f} m/s,"
          f" hit rate = {best_rate*100:.1f}%")

    # Shipped V3.1 pick (V3_TABLE in shooter_sim.html — centered landing, exit 0.50 m)
    V31 = {
        1.50: (72.807, 6.1751),
        1.75: (70.729, 6.3238),
        2.00: (68.820, 6.4776),
        2.25: (67.073, 6.6362),
        2.50: (65.482, 6.7991),
        2.75: (64.037, 6.9655),
        3.00: (62.726, 7.1345),
        3.25: (61.537, 7.3056),
        3.50: (60.458, 7.4780),
        3.75: (59.467, 7.6509),
        4.00: (58.574, 7.8246),
        4.25: (57.760, 7.9980),
        4.50: (57.014, 8.1715),
        4.75: (56.330, 8.3447),
        5.00: (55.700, 8.5174),
    }
    if d in V31:
        h_v31, v_v31 = V31[d]
        rate_v31, spr_v31, tof_v31, _ = burst_hit_rate(d, h_v31, v_v31, N=args.n_balls)
        print(f"V3.1 table pick:  hood = {h_v31:.2f}°, v = {v_v31:.3f} m/s,"
              f" hit rate = {rate_v31*100:.1f}%, spread = {spr_v31:.3f} m, TOF = {tof_v31:.3f}s")
        gap = best_rate - rate_v31
        if gap > 0.02:
            print(f"\n>>> V3.1 table is suboptimal by {gap*100:.1f} pp"
                  f" (best {best_rate*100:.1f}% vs V3.1 {rate_v31*100:.1f}%)")
        else:
            print(f"\nV3.1 table pick is within {gap*100:.1f} pp of best in sweep.")


if __name__ == '__main__':
    main()
