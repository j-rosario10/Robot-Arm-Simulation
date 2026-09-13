"""Shared fixtures for the puppet_pilot test suite.

Tests are runnable with plain pytest from anywhere - no ROS environment, no
installed package: the workspace src/ directory is put on sys.path directly.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from puppet_pilot.kinematics.collision import CollisionModelConfig, SelfCollisionChecker  # noqa: E402
from puppet_pilot.kinematics.params import DHTable, JointLimits, LinkGeometry  # noqa: E402
from puppet_pilot.spec import SortingSpec, load_spec, repo_root  # noqa: E402


@pytest.fixture(scope="session")
def spec() -> SortingSpec:
    return load_spec()


@pytest.fixture(scope="session")
def spec_path() -> Path:
    return repo_root() / "SPEC.md"


@pytest.fixture(scope="session")
def dh() -> DHTable:
    return DHTable.from_yaml()


@pytest.fixture(scope="session")
def limits() -> JointLimits:
    return JointLimits.from_yaml()


@pytest.fixture(scope="session")
def geometry() -> LinkGeometry:
    return LinkGeometry.from_yaml()


@pytest.fixture(scope="session")
def collision_model() -> CollisionModelConfig:
    return CollisionModelConfig.from_yaml()


@pytest.fixture(scope="session")
def checker(dh, geometry, collision_model) -> SelfCollisionChecker:
    return SelfCollisionChecker(dh, geometry, collision_model)


@pytest.fixture(scope="session")
def kinematic_urdf_path() -> str:
    path = Path(__file__).resolve().parent / "data" / "ur5e_robotiq_kinematic.urdf"
    if not path.exists():
        pytest.fail(
            f"missing test fixture {path}; regenerate with tools/make_test_urdf.py"
        )
    return str(path)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)
