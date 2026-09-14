"""Capsule-based self-collision checking for the UR5e + gripper.

The arm is approximated by capsules (line segment + radius). Capsule-capsule
distance is analytic: distance between the two core segments minus the radii.
That gives an O(1) exact test per pair, conservative by construction, cheap
enough to run inside a teleop control loop, and - unlike delegating to the
simulator's collision engine - usable as a command veto *before* a command
reaches the sim (or, one day, real hardware).

Geometry provenance (no invented numbers):
  - capsule radii/lengths: ur_description physical_parameters.yaml (links table)
  - lateral tube offsets:  measured from the collision meshes by
                           tools/fit_collision_model.py, cached in
                           config/collision_model.yaml, cross-checked there
                           against the YAML's shoulder/elbow offsets
  - gripper capsule:       measured from the gripper collision meshes by the
                           same tool, stored in the same file
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from puppet_pilot.kinematics.fk import fk_frames
from puppet_pilot.kinematics.params import DHTable, LinkGeometry


def segment_segment_distance(
    p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray
) -> float:
    """Minimum distance between segments [p1,q1] and [p2,q2].

    Clamped closest-point algorithm (Ericson, Real-Time Collision Detection,
    section 5.1.9): solve the unconstrained closest points of the two infinite
    lines, then clamp each parameter into [0,1], re-projecting the other when
    a clamp changes it. Degenerate (zero-length) segments fall out naturally
    as point cases.
    """
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(d1 @ d1)
    e = float(d2 @ d2)
    f = float(d2 @ r)
    eps = 1e-12

    if a <= eps and e <= eps:  # both segments are points
        return float(np.linalg.norm(r))
    if a <= eps:  # first segment is a point
        t = np.clip(f / e, 0.0, 1.0)
        return float(np.linalg.norm(p1 - (p2 + t * d2)))

    c = float(d1 @ r)
    if e <= eps:  # second segment is a point
        s = np.clip(-c / a, 0.0, 1.0)
        return float(np.linalg.norm((p1 + s * d1) - p2))

    b = float(d1 @ d2)
    denom = a * e - b * b  # always >= 0; zero when the lines are parallel

    # Closest point on line 1 to line 2, clamped; parallel lines pick s=0.
    s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom > eps else 0.0
    t = (b * s + f) / e
    # If t left [0,1], clamp it and recompute s for that endpoint.
    if t < 0.0:
        t = 0.0
        s = np.clip(-c / a, 0.0, 1.0)
    elif t > 1.0:
        t = 1.0
        s = np.clip((b - c) / a, 0.0, 1.0)

    return float(np.linalg.norm((p1 + s * d1) - (p2 + t * d2)))


@dataclass(frozen=True)
class Capsule:
    name: str
    point_a: np.ndarray
    point_b: np.ndarray
    radius: float

    def distance_to(self, other: "Capsule") -> float:
        """Surface-to-surface distance (negative = interpenetration)."""
        core = segment_segment_distance(self.point_a, self.point_b, other.point_a, other.point_b)
        return core - self.radius - other.radius


@dataclass(frozen=True)
class CollisionModelConfig:
    """Generated part of the collision model (see tools/fit_collision_model.py)."""

    lateral_offset_upper_arm: float  # signed, along the shoulder-lift axis [m]
    lateral_offset_forearm: float    # signed, along the elbow axis [m]
    gripper_point_a: np.ndarray      # in the DH tool frame (frame 6)
    gripper_point_b: np.ndarray
    gripper_radius: float

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> "CollisionModelConfig":
        if path is None:
            path = Path(__file__).resolve().parents[1] / "config" / "collision_model.yaml"
        with open(path) as f:
            cfg = yaml.safe_load(f)
        gripper = cfg["gripper_capsule"]
        return cls(
            lateral_offset_upper_arm=float(cfg["lateral_offsets"]["upper_arm"]),
            lateral_offset_forearm=float(cfg["lateral_offsets"]["forearm"]),
            gripper_point_a=np.array(gripper["point_a"], dtype=float),
            gripper_point_b=np.array(gripper["point_b"], dtype=float),
            gripper_radius=float(gripper["radius"]),
        )


@dataclass(frozen=True)
class PairResult:
    name_a: str
    name_b: str
    distance: float  # surface-to-surface, negative = interpenetration

    @property
    def colliding(self) -> bool:
        return self.distance < 0.0


@dataclass(frozen=True)
class CollisionReport:
    pairs: tuple[PairResult, ...]

    @property
    def colliding(self) -> bool:
        return any(p.colliding for p in self.pairs)

    @property
    def worst(self) -> PairResult:
        return min(self.pairs, key=lambda p: p.distance)


class SelfCollisionChecker:
    """Self-collision veto for the UR5e (+ mounted gripper).

    Capsule axes ride on the FK joint origins, so the model follows the arm
    with no extra state. The upper-arm and forearm axes are displaced
    laterally along their joint axes by the measured tube offsets - the UR's
    links do not run through the DH frame origins, and ignoring that (~0.13 m
    for the upper arm) would place the biggest capsule in empty space.
    """

    # Capsule chain indices (order matters for the adjacency rule below):
    # 0 base, 1 shoulder, 2 upper_arm, 3 forearm, 4 wrist_1, 5 wrist_2,
    # 6 wrist_3, 7 gripper.
    #
    # Checked pairs: all (i, j) with j > i+1, minus exclusions inside the
    # wrist cluster. The three wrist links and the gripper base form a compact
    # group whose members sit within ~one link-offset (~0.10-0.14 m) of each
    # other at *every* configuration - comparable to the capsule radius sums -
    # so capsule distances there measure the approximation, not real
    # clearance. The real hardware's designed internal clearances cover these
    # pairs; the genuine self-collision risks (gripper/wrist into forearm,
    # upper arm, shoulder, base) all remain checked. This mirrors the
    # "adjacent / never in collision" pairs a MoveIt SRDF generator disables.
    _EXCLUDED_PAIRS = {
        (4, 6),  # wrist_1 vs wrist_3
        (4, 7),  # wrist_1 vs gripper
        (5, 7),  # wrist_2 vs gripper
    }

    def __init__(self, dh: DHTable, geometry: LinkGeometry, model: CollisionModelConfig):
        self._dh = dh
        self._geometry = geometry
        self._model = model

    def capsules(self, q: np.ndarray) -> list[Capsule]:
        frames = fk_frames(self._dh, q)
        origins = frames[:, :3, 3]          # p[i] = origin of DH frame i
        z_axes = frames[:, :3, 2]           # z[i] = axis of joint i+1
        radius = self._geometry.link_radius
        length = self._geometry.link_length

        # Lateral displacement of the two big tubes off their DH segments.
        shift_upper = self._model.lateral_offset_upper_arm * z_axes[1]
        shift_fore = self._model.lateral_offset_forearm * z_axes[2]

        # Shoulder assembly extends along the shoulder-lift axis around p1.
        shoulder_half = 0.5 * length["shoulder"] * z_axes[1]

        tool = frames[6]
        gripper_a = tool[:3, 3] + tool[:3, :3] @ self._model.gripper_point_a
        gripper_b = tool[:3, 3] + tool[:3, :3] @ self._model.gripper_point_b

        return [
            Capsule("base", origins[0], origins[1], radius["base"]),
            Capsule("shoulder", origins[1] - shoulder_half, origins[1] + shoulder_half, radius["shoulder"]),
            Capsule("upper_arm", origins[1] + shift_upper, origins[2] + shift_upper, radius["upperarm"]),
            Capsule("forearm", origins[2] + shift_fore, origins[3] + shift_fore, radius["forearm"]),
            Capsule("wrist_1", origins[3], origins[4], radius["wrist_1"]),
            Capsule("wrist_2", origins[4], origins[5], radius["wrist_2"]),
            Capsule("wrist_3", origins[5], origins[6], radius["wrist_3"]),
            Capsule("gripper", gripper_a, gripper_b, self._model.gripper_radius),
        ]

    def check(self, q: np.ndarray, margin: float = 0.0) -> CollisionReport:
        """Report every checked pair; ``colliding`` means distance < margin.

        margin=0 means capsules touching. The caller (teleop layer) decides
        its own safety margin; none is baked in here.
        """
        capsules = self.capsules(q)
        results = []
        for i in range(len(capsules)):
            for j in range(i + 2, len(capsules)):
                if (i, j) in self._EXCLUDED_PAIRS:
                    continue
                distance = capsules[i].distance_to(capsules[j]) - margin
                results.append(PairResult(capsules[i].name, capsules[j].name, distance))
        return CollisionReport(pairs=tuple(results))
