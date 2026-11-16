"""Shared fixtures: the configured robot/field model, built once per session."""
from pathlib import Path

import pytest

from src.config import load_game, load_robot
from src.optimizer.maxprob import ShotModel

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def configs():
    return load_game(ROOT / "config" / "game_2026.yaml"), load_robot(ROOT / "config" / "robot.yaml")


@pytest.fixture(scope="session")
def model(configs):
    game, robot = configs
    return ShotModel.from_configs(game, robot)


@pytest.fixture(scope="session")
def budget(configs):
    return configs[1].error_budget
