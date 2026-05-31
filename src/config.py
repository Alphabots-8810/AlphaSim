"""Config loaders: YAML → typed dataclasses."""
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import yaml


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
    w_entry_angle: float = 10.0
    w_v_exit: float = 0.5
    w_tof: float = 70.0


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
class RobotConfig:
    shooter: ShooterMountConfig
    flywheel: RollerConfig
    hood_roller: RollerConfig
    operating_distance_m: OperatingRange
    cost_weights: CostWeights


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
    return RobotConfig(
        shooter=ShooterMountConfig(**d['shooter']),
        flywheel=RollerConfig(**d['flywheel']),
        hood_roller=RollerConfig(**d['hood_roller']),
        operating_distance_m=OperatingRange(**d['operating_distance_m']),
        cost_weights=CostWeights(**d.get('cost_weights', {})),
    )
