"""Parameter loading for the UR5e kinematics core.

Nothing in this package hardcodes robot numbers. The single sources of truth are
the files Universal Robots ships in ur_description:

    default_kinematics.yaml   -> DH table (same values UR publishes for the UR5e)
    joint_limits.yaml         -> position/velocity/effort limits (uses a custom
                                 ``!degrees`` YAML tag, handled below)
    physical_parameters.yaml  -> link capsule radii/lengths + lateral offsets

The kinematics YAML is in *calibrated* format: one full XYZ/RPY transform per
joint, so a real robot's factory calibration could be dropped in. For the
nominal robot most fields are structural zeros. ``DHTable.from_yaml`` maps the
offsets onto a standard-DH table and then *verifies* the mapping numerically:
it composes the raw YAML transform chain and the DH chain at several joint
configurations and requires them to be identical. A calibrated file that broke
the clean DH structure would fail that check loudly instead of producing a
subtly wrong table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

# Canonical joint order used by every module in this package.
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Order of the per-joint entries in default_kinematics.yaml. Entry i is the
# fixed transform that sits *before* revolute joint i in the URDF chain.
_KINEMATICS_KEYS = ["shoulder", "upper_arm", "forearm", "wrist_1", "wrist_2", "wrist_3"]


def ur5e_config_dir() -> Path:
    """Locate ur_description's ur5e config directory.

    Resolved relative to this file's position inside the workspace
    (src/puppet_pilot/kinematics/ -> src/ur_description/config/ur5e), so tests
    and tools work without a sourced ROS environment.
    """
    src_dir = Path(__file__).resolve().parents[2]
    config = src_dir / "ur_description" / "config" / "ur5e"
    if not config.is_dir():
        raise FileNotFoundError(
            f"ur5e config directory not found at {config}; "
            "is the workspace layout intact?"
        )
    return config


# --------------------------------------------------------------------------
# small transform helpers (module-local; fk.py has the DH-specific ones)
# --------------------------------------------------------------------------

def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF-convention rotation: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _xyzrpy_transform(entry: dict) -> np.ndarray:
    """4x4 homogeneous transform from one YAML kinematics entry."""
    t = np.eye(4)
    t[:3, :3] = _rpy_matrix(entry["roll"], entry["pitch"], entry["yaw"])
    t[:3, 3] = [entry["x"], entry["y"], entry["z"]]
    return t


def _rot_z(q: float) -> np.ndarray:
    t = np.eye(4)
    c, s = math.cos(q), math.sin(q)
    t[:2, :2] = [[c, -s], [s, c]]
    return t


def _pure_x_rotation_angle(r: np.ndarray, context: str) -> float:
    """Extract alpha from a rotation that must be a pure rotation about x.

    The wrist_3 entry encodes its alpha as roll=pi/2, pitch=pi, yaw=pi, which
    composes to the same matrix as Rx(-pi/2) - so we extract the angle from
    the matrix and verify the matrix really is Rx(angle), rather than trusting
    any single RPY field.
    """
    alpha = math.atan2(r[2, 1], r[1, 1])
    ca, sa = math.cos(alpha), math.sin(alpha)
    expected = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    if not np.allclose(r, expected, atol=1e-6):
        raise ValueError(
            f"{context}: rotation is not a pure x-rotation; "
            "kinematics YAML does not match the expected UR structure"
        )
    return alpha


# --------------------------------------------------------------------------
# DH table
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DHTable:
    """Standard (distal) DH parameters, one row per joint.

    Transform for row i: A_i = Rz(theta_i) @ Tz(d_i) @ Tx(a_i) @ Rx(alpha_i).
    """

    a: np.ndarray      # link lengths [m], shape (6,)
    d: np.ndarray      # link offsets [m], shape (6,)
    alpha: np.ndarray  # link twists [rad], shape (6,)

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> "DHTable":
        if path is None:
            path = ur5e_config_dir() / "default_kinematics.yaml"
        with open(path) as f:
            k = yaml.safe_load(f)["kinematics"]

        # Field-by-field mapping from the calibrated-offsets format onto
        # standard DH (documented in docs/phase2-kinematics.md, section 1).
        d1 = k["shoulder"]["z"]
        alpha1 = _pure_x_rotation_angle(
            _rpy_matrix(k["upper_arm"]["roll"], k["upper_arm"]["pitch"], k["upper_arm"]["yaw"]),
            "upper_arm",
        )
        a2 = k["forearm"]["x"]
        a3 = k["wrist_1"]["x"]
        d4 = k["wrist_1"]["z"]
        d5 = -k["wrist_2"]["y"]
        alpha4 = _pure_x_rotation_angle(
            _rpy_matrix(k["wrist_2"]["roll"], k["wrist_2"]["pitch"], k["wrist_2"]["yaw"]),
            "wrist_2",
        )
        d6 = k["wrist_3"]["y"]
        alpha5 = _pure_x_rotation_angle(
            _rpy_matrix(k["wrist_3"]["roll"], k["wrist_3"]["pitch"], k["wrist_3"]["yaw"]),
            "wrist_3",
        )

        table = cls(
            a=np.array([0.0, a2, a3, 0.0, 0.0, 0.0]),
            d=np.array([d1, 0.0, 0.0, d4, d5, d6]),
            alpha=np.array([alpha1, 0.0, 0.0, alpha4, alpha5, 0.0]),
        )
        table._verify_against_yaml_chain(k)
        return table

    def _verify_against_yaml_chain(self, k: dict) -> None:
        """Require DH chain == raw YAML transform chain for several q.

        The YAML chain is exactly what the URDF builds:
            T(q) = T_shoulder @ Rz(q1) @ T_upper_arm @ Rz(q2) @ ... @ Rz(q6)
        If the field mapping above dropped or misplaced anything (e.g. a
        calibrated file with non-structural values), the two chains diverge
        and we refuse to continue.
        """
        from puppet_pilot.kinematics.fk import fk_frames  # late import, avoids cycle

        rng = np.random.default_rng(0)
        configs = [np.zeros(6)] + [rng.uniform(-math.pi, math.pi, 6) for _ in range(3)]
        for q in configs:
            t_yaml = np.eye(4)
            for key, qi in zip(_KINEMATICS_KEYS, q):
                t_yaml = t_yaml @ _xyzrpy_transform(k[key]) @ _rot_z(qi)
            t_dh = fk_frames(self, q)[-1]
            if not np.allclose(t_yaml, t_dh, atol=1e-9):
                raise ValueError(
                    "DH table does not reproduce the YAML kinematic chain; "
                    "the kinematics file does not have the expected structure"
                )


# --------------------------------------------------------------------------
# joint limits
# --------------------------------------------------------------------------

class _DegreesLoader(yaml.SafeLoader):
    """SafeLoader that understands ur_description's ``!degrees`` tag."""


def _degrees_constructor(loader: yaml.Loader, node: yaml.Node) -> float:
    return math.radians(float(loader.construct_scalar(node)))


_DegreesLoader.add_constructor("!degrees", _degrees_constructor)


@dataclass(frozen=True)
class JointLimits:
    """Position/velocity/effort limits in canonical joint order (radians, Nm)."""

    lower: np.ndarray
    upper: np.ndarray
    velocity: np.ndarray
    effort: np.ndarray

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> "JointLimits":
        if path is None:
            path = ur5e_config_dir() / "joint_limits.yaml"
        with open(path) as f:
            limits = yaml.load(f, Loader=_DegreesLoader)["joint_limits"]
        return cls(
            lower=np.array([limits[j]["min_position"] for j in JOINT_NAMES]),
            upper=np.array([limits[j]["max_position"] for j in JOINT_NAMES]),
            velocity=np.array([limits[j]["max_velocity"] for j in JOINT_NAMES]),
            effort=np.array([limits[j]["max_effort"] for j in JOINT_NAMES]),
        )

    def clip(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.lower, self.upper)

    def contains(self, q: np.ndarray) -> bool:
        return bool(np.all(q >= self.lower) and np.all(q <= self.upper))


# --------------------------------------------------------------------------
# link geometry for the collision model
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LinkGeometry:
    """Capsule radii/lengths and lateral link offsets from physical_parameters.yaml.

    ``link_radius``/``link_length`` come from the purpose-built ``links:`` table
    (the file also carries per-part ``*_radius`` scalars intended for inertia
    models; the ``links:`` values are the more conservative set - documented
    choice, see docs/phase2-kinematics.md section 5).
    """

    link_radius: dict
    link_length: dict
    shoulder_offset: float  # lateral offset of the upper-arm tube from its DH segment
    elbow_offset: float     # lateral offset of the forearm tube from its DH segment

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> "LinkGeometry":
        if path is None:
            path = ur5e_config_dir() / "physical_parameters.yaml"
        with open(path) as f:
            phys = yaml.safe_load(f)
        links = phys["inertia_parameters"]["links"]
        return cls(
            link_radius={name: float(v["radius"]) for name, v in links.items()},
            link_length={name: float(v["length"]) for name, v in links.items()},
            shoulder_offset=float(phys["offsets"]["shoulder_offset"]),
            elbow_offset=float(phys["offsets"]["elbow_offset"]),
        )
