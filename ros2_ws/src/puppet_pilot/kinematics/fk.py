"""Forward kinematics and the geometric Jacobian for a standard-DH chain.

All poses are 4x4 homogeneous transforms expressed in the DH base frame
(frame 0). Frame 0 sits at the intersection of joint 1's axis with the robot
mounting plane, z up - it is NOT the URDF's ``base_link`` frame, which UR
defines rotated pi about z to match the teach pendant. The unit tests measure
and pin down that fixed correction; consumers that need URDF-world poses apply
it explicitly (see tests/test_fk.py).
"""

from __future__ import annotations

import math

import numpy as np

from puppet_pilot.kinematics.params import DHTable


def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """One standard-DH link transform: A = Rz(theta) @ Tz(d) @ Tx(a) @ Rx(alpha).

    Reading right to left: twist the new frame about the old x-axis (alpha),
    slide it out along x (a), lift it along the joint axis z (d), then rotate
    by the joint variable (theta). Multiplied out (the form below) it is the
    textbook standard-DH matrix.
    """
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def fk_frames(dh: DHTable, q: np.ndarray) -> np.ndarray:
    """All chain frames for joint configuration q.

    Returns an array of shape (7, 4, 4): frames[0] is the base (identity),
    frames[i] is the pose of DH frame i in the base frame. Returning every
    intermediate frame (not just the tool) is deliberate: the Jacobian and the
    collision capsules both need the joint origins and z-axes, and a wrong FK
    is only debuggable when the chain can be inspected link by link.
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (6,):
        raise ValueError(f"expected 6 joint angles, got shape {q.shape}")
    frames = np.empty((7, 4, 4))
    frames[0] = np.eye(4)
    for i in range(6):
        frames[i + 1] = frames[i] @ dh_transform(q[i], dh.d[i], dh.a[i], dh.alpha[i])
    return frames


def tool_pose(dh: DHTable, q: np.ndarray) -> np.ndarray:
    """Pose of DH frame 6 (the tool flange) in the base frame."""
    return fk_frames(dh, q)[-1]


def geometric_jacobian(dh: DHTable, q: np.ndarray, frames: np.ndarray | None = None) -> np.ndarray:
    """6x6 geometric Jacobian mapping joint velocities to the tool twist.

    Column i (revolute joint):
        J_v,i = z_{i-1} x (p_tool - p_{i-1})   [linear velocity from joint i]
        J_w,i = z_{i-1}                        [angular velocity from joint i]

    where z_{i-1} is joint i's rotation axis and p_{i-1} its origin, both in
    the base frame. Physically: turning joint i sweeps the tool about that
    joint's axis, so the tool's linear velocity is the lever arm crossed with
    the axis, and its angular velocity is the axis itself. The twist is
    expressed in the base frame with the reference point at the tool.
    """
    if frames is None:
        frames = fk_frames(dh, q)
    p_tool = frames[-1][:3, 3]
    jac = np.empty((6, 6))
    for i in range(6):
        z_axis = frames[i][:3, 2]
        origin = frames[i][:3, 3]
        jac[:3, i] = np.cross(z_axis, p_tool - origin)
        jac[3:, i] = z_axis
    return jac
