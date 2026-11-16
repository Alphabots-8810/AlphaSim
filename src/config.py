"""Config loaders: YAML → typed dataclasses."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import yaml

from src.scoring import ErrorBudget


@dataclass
class HubConfig:
    bottom_hex_across_flats_m: float
    top_height_m: float
    bottom_height_m: float
    target_height_m: float
    target_across_flats_m: float
    base_size_m: float


@dataclass
class BallConfig:
    diameter_m: float
    radius_m: float
    mass_kg: float
    drag_coefficient: float
    magnus_cl: float


@dataclass
class EnvironmentConfig:
    air_density_kg_m3: float
    gravity_m_s2: float


@dataclass
class GameConfig:
    hub: HubConfig
    ball: BallConfig
    environment: EnvironmentConfig


@dataclass
class ShooterMountConfig:
    exit_height_m: float
    exit_forward_offset_m: float
    hood_angle_min_deg: float
    hood_angle_max_deg: float
    ball_clump_half_width_m: float = 0.0


@dataclass
class CostWeights:
    # Legacy V3.1 weighted-cost solver only (src/optimizer/solve.py). The V5
    # solver has no weights; it reads `error_budget` and `aim` instead.
    w_entry_angle: float = 1.0
    w_v_exit: float = 3.0
    w_tof: float = 1.1


@dataclass
class RollerConfig:
    diameter_m: float
    motor: str
    motor_free_rpm: float
    gear_ratio: float
    max_flywheel_rpm: float
    slip_factor: float


@dataclass
class OperatingRange:
    min: float
    max: float
    step: float


@dataclass
class AimConfig:
    """Settings of the V5 (maximum scoring probability) aim solver."""
    # Fraction of the best achievable sigma margin the solver may give up in
    # exchange for a lower flywheel speed. 0 = pure maximum hit probability.
    margin_concession: float = 0.03
    # Angle between the shot line and the nearest hex flat's normal.
    # 0 = a flat faces the robot (narrowest along the shot line; conservative).
    hex_bearing_deg: float = 0.0


@dataclass
class RobotConfig:
    shooter: ShooterMountConfig
    flywheel: RollerConfig
    hood_roller: RollerConfig
    operating_distance_m: OperatingRange
    cost_weights: CostWeights
    error_budget: ErrorBudget = field(default_factory=ErrorBudget)
    aim: AimConfig = field(default_factory=AimConfig)


def load_game(path: Union[Path, str]) -> GameConfig:
    with open(path) as f:
        d = yaml.safe_load(f)
    return GameConfig(
        hub=HubConfig(**d['hub']),
        ball=BallConfig(**d['ball']),
        environment=EnvironmentConfig(**d['environment']),
    )


def load_robot(path: Union[Path, str]) -> RobotConfig:
    with open(path) as f:
        d = yaml.safe_load(f)
    shooter = ShooterMountConfig(**d['shooter'])
    # The dumper width lives with the shooter geometry; every other budget
    # entry is a noise level and lives in its own block.
    budget = ErrorBudget(**{'lateral_half_width_m': shooter.ball_clump_half_width_m,
                            **d.get('error_budget', {})})
    return RobotConfig(
        shooter=shooter,
        flywheel=RollerConfig(**d['flywheel']),
        hood_roller=RollerConfig(**d['hood_roller']),
        operating_distance_m=OperatingRange(**d['operating_distance_m']),
        cost_weights=CostWeights(**d.get('cost_weights', {})),
        error_budget=budget,
        aim=AimConfig(**d.get('aim', {})),
    )
