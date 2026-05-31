"""Trajectory + hub visualization."""
import matplotlib.pyplot as plt
import numpy as np

from src.geometry.target import HexTarget
from src.physics.trajectory import TrajectoryResult


def plot_trajectory(
    traj: TrajectoryResult,
    target: HexTarget,
    distance: float,
    exit_height: float,
    ball_radius: float = 0.075,
    ax=None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 6))

    ax.plot(traj.positions[:, 0], traj.positions[:, 1], 'b-', lw=2, label='trajectory')
    ax.plot(0.0, exit_height, 'g^', ms=14, label=f'shooter exit (h={exit_height:.2f} m)')

    a = target.apothem - ball_radius
    ax.plot([distance - a, distance + a], [target.height_m, target.height_m],
            'r-', lw=4, label=f'effective hex slot (2·{a:.2f} m)')
    ax.plot([distance - target.apothem, distance + target.apothem],
            [target.height_m, target.height_m],
            'r--', lw=1, alpha=0.5, label='hex apothem ±')

    if traj.hit:
        ax.plot(traj.hit_x, target.height_m, 'rx', ms=15, mew=3,
                label=f'entry: {traj.entry_angle_deg:.1f}° @ {traj.entry_speed_ms:.1f} m/s')

    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('horizontal distance (m)')
    ax.set_ylabel('height (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)
    return ax


def plot_feasibility_heatmap(
    distance_range_m,
    flywheel_rpm_range,
    hood_angle_range_deg,
    solve_fn,
    metric: str = 'flywheel_rpm',
):
    """Sweep distances and plot the metric (e.g. RPM, hood angle) vs distance."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    rpms, angles, distances = [], [], []
    for d in distance_range_m:
        sol = solve_fn(d)
        if sol.success:
            rpms.append(sol.flywheel_rpm)
            angles.append(sol.hood_angle_deg)
            distances.append(d)

    axes[0].plot(distances, rpms, 'o-')
    axes[0].set_xlabel('distance (m)')
    axes[0].set_ylabel('flywheel RPM')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(distances, angles, 'o-', color='orange')
    axes[1].set_xlabel('distance (m)')
    axes[1].set_ylabel('hood angle (°)')
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
