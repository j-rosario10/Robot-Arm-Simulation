"""Generate config/collision_model.yaml from the robot's own collision meshes.

Two things in the capsule self-collision model cannot be read directly from
ur_description's YAML files, and inventing them is not allowed:

  1. The *signed* lateral offsets of the upper-arm and forearm tubes from
     their DH frame segments. physical_parameters.yaml gives magnitudes
     (shoulder_offset, elbow_offset) but not directions.
  2. The gripper's bounding capsule (the 2F-85 is not part of ur_description).

This tool measures both from the built URDF's collision geometry in PyBullet
(AABB queries at the zero configuration) and caches the result with
provenance. Rerun after any change to the robot model:

    python tools/fit_collision_model.py --urdf /tmp/ur5e_robotiq_pybullet.urdf
"""

from __future__ import annotations

import argparse
import datetime
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # make puppet_pilot importable

import pybullet as p  # noqa: E402

from puppet_pilot.kinematics.fk import fk_frames  # noqa: E402
from puppet_pilot.kinematics.params import DHTable, LinkGeometry  # noqa: E402

# Radial safety factor for the gripper capsule. The capsule radius is inflated
# by this factor over the measured max radial extent, which buys the axial trim
# below (a fatter capsule needs less axial overshoot to still contain the
# geometry). Pure geometry, no robot-specific tuning.
_RADIAL_INFLATION = 1.1


def _pose_to_matrix(position, quaternion) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
    t[:3, 3] = position
    return t


def _link_indices_by_name(body: int) -> dict[str, int]:
    names = {}
    for i in range(p.getNumJoints(body)):
        names[p.getJointInfo(body, i)[12].decode()] = i
    return names


def _aabb_corners(body: int, link_index: int) -> np.ndarray:
    lo, hi = p.getAABB(body, link_index)
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", required=True, help="PyBullet-loadable URDF (absolute mesh paths)")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "config" / "collision_model.yaml"),
    )
    args = parser.parse_args()

    dh = DHTable.from_yaml()
    geometry = LinkGeometry.from_yaml()

    p.connect(p.DIRECT)
    robot = p.loadURDF(args.urdf, useFixedBase=True)
    links = _link_indices_by_name(robot)
    # Zero configuration everywhere (PyBullet default) - matches fk_frames(q=0).

    # World -> DH base: the DH chain starts at base_link_inertia (see params.py).
    t_world_dh = _pose_to_matrix(*p.getLinkState(robot, links["ur5e_base_link_inertia"], computeForwardKinematics=1)[4:6])
    t_dh_world = np.linalg.inv(t_world_dh)
    frames = fk_frames(dh, np.zeros(6))

    def lateral_offset(link_name: str, frame_index: int) -> float:
        """Signed offset of a link's AABB centre from its DH segment, along the joint axis."""
        corners_world = _aabb_corners(robot, links[link_name])
        centre_dh = (t_dh_world @ np.append(corners_world.mean(axis=0), 1.0))[:3]
        axis = frames[frame_index][:3, 2]
        origin = frames[frame_index][:3, 3]
        return float((centre_dh - origin) @ axis)

    offset_upper = lateral_offset("ur5e_upper_arm_link", 1)
    offset_fore = lateral_offset("ur5e_forearm_link", 2)

    # Cross-check magnitudes against the YAML's published offsets (loose
    # tolerance: an AABB centre is a coarse estimator of a tube axis).
    # Physical layout, confirmed by measurement: the upper-arm tube runs
    # laterally displaced by shoulder_offset; the elbow steps back toward the
    # DH plane, so the forearm tube runs at only ~elbow_offset; the lateral
    # jog out to the wrist happens at the wrist_1 joint (d4).
    expected_upper = geometry.shoulder_offset
    expected_fore = geometry.elbow_offset
    for label, measured, expected in [
        ("upper_arm", offset_upper, expected_upper),
        ("forearm", offset_fore, expected_fore),
    ]:
        if abs(abs(measured) - expected) > 0.05:
            raise RuntimeError(
                f"{label} lateral offset {measured:+.4f} m does not match "
                f"physical_parameters.yaml expectation +/-{expected:.4f} m - "
                "URDF layout may have changed; refusing to write a bad model"
            )

    # Gripper capsule: every robotiq link's AABB corners, in the DH tool frame.
    t_tool_world = np.linalg.inv(t_world_dh @ frames[6])
    corners = []
    for name, index in links.items():
        if "robotiq" in name:
            for corner in _aabb_corners(robot, index):
                corners.append((t_tool_world @ np.append(corner, 1.0))[:3])
    corners = np.array(corners)
    if len(corners) == 0:
        raise RuntimeError("no robotiq links found in URDF")

    radial = float(np.max(np.hypot(corners[:, 0], corners[:, 1])))
    z_min, z_max = float(corners[:, 2].min()), float(corners[:, 2].max())
    radius = _RADIAL_INFLATION * radial
    # Axial trim: a capsule cap (sphere radius R) still covers the cylinder rim
    # (radius r) if the cap centre sits within sqrt(R^2 - r^2) of the end face.
    trim = math.sqrt(radius**2 - radial**2)
    point_a = [0.0, 0.0, z_min + trim]
    point_b = [0.0, 0.0, max(z_min + trim, z_max - trim)]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.date.today().isoformat()
    out_path.write_text(
        f"""# Generated by tools/fit_collision_model.py on {stamp} - do not edit by hand.
# Sources: collision meshes of the built ur5e_robotiq_2f_85 URDF (PyBullet AABB
# queries at q=0) cross-checked against ur_description physical_parameters.yaml.
# Frames: lateral offsets are signed displacements along the respective joint
# axes; the gripper capsule is expressed in the DH tool frame (frame 6).

lateral_offsets:
  upper_arm: {offset_upper:.6f}
  forearm: {offset_fore:.6f}

gripper_capsule:
  point_a: [{point_a[0]:.6f}, {point_a[1]:.6f}, {point_a[2]:.6f}]
  point_b: [{point_b[0]:.6f}, {point_b[1]:.6f}, {point_b[2]:.6f}]
  radius: {radius:.6f}
""",
        encoding="utf-8",
    )
    print(f"wrote {out_path}")
    print(f"  lateral offsets: upper_arm {offset_upper:+.4f} m, forearm {offset_fore:+.4f} m")
    print(f"  gripper capsule: z [{z_min:+.4f}, {z_max:+.4f}], radial {radial:.4f} -> r={radius:.4f}")


if __name__ == "__main__":
    main()
