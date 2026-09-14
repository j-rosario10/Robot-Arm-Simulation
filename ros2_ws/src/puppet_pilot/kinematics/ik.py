"""Damped least-squares (DLS) inverse kinematics.

Why DLS and not the plain pseudoinverse: the pseudoinverse step
``dq = pinv(J) @ e`` contains a 1/sigma factor per singular value of J. Near a
singularity the smallest sigma tends to zero and that factor explodes - the
solver demands enormous joint motion to progress in a task direction the
mechanism is losing. DLS solves the regularised problem

    dq = J^T (J J^T + lambda^2 I)^{-1} e

whose per-direction gain is sigma/(sigma^2 + lambda^2): identical to the
pseudoinverse when sigma >> lambda, but rolling smoothly to zero as sigma -> 0,
with a bounded peak of 1/(2*lambda). The arm becomes sluggish along the lost
direction instead of snapping. That trade (bounded commands, temporary
tracking error) is the right one for teleoperation.

The full discussion, including why we iterate a numeric method instead of
using the UR5e's closed-form IK, lives in docs/phase2-kinematics.md section 3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from puppet_pilot.kinematics.fk import fk_frames, geometric_jacobian
from puppet_pilot.kinematics.params import DHTable, JointLimits


def rotation_vector_from_matrix(rot: np.ndarray) -> np.ndarray:
    """Axis-angle (rotation vector) of a rotation matrix, robust at 0 and pi.

    Three regimes, because the generic formula divides by sin(angle):
      - angle ~ 0:   R - R^T is already ~ 2*angle*[axis]_x, use it directly.
      - generic:     axis = vee(R - R^T) / (2 sin(angle)).
      - angle ~ pi:  sin ~ 0 makes the generic form useless; recover the axis
                     from the diagonal of R ~ 2*a*a^T - I instead.
    """
    trace = np.trace(rot)
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle = math.acos(cos_angle)

    skew_vee = 0.5 * np.array(
        [rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]]
    )

    if angle < 1e-9:
        return skew_vee  # first-order accurate for tiny rotations
    if angle < math.pi - 1e-6:
        return angle * skew_vee / math.sin(angle)

    # Near pi: R ~ 2 a a^T - I  =>  a_i = sqrt((R_ii + 1)/2), signs from row k.
    k = int(np.argmax(np.diagonal(rot)))
    axis = np.empty(3)
    axis[k] = math.sqrt(max(0.0, (rot[k, k] + 1.0) / 2.0))
    for j in range(3):
        if j != k:
            axis[j] = rot[k, j] / (2.0 * axis[k])
    axis /= np.linalg.norm(axis)
    return angle * axis


@dataclass
class IKResult:
    q: np.ndarray                 # best joint configuration found
    converged: bool               # both tolerances met
    iterations: int
    position_error: float         # [m] residual at q
    orientation_error: float      # [rad] residual at q
    min_singular_value: float     # of J at q - proximity to singularity


def solve_ik(
    dh: DHTable,
    q_start: np.ndarray,
    target_pose: np.ndarray,
    limits: JointLimits | None = None,
    damping: float = 0.05,
    adaptive_damping: bool = False,
    singular_threshold: float = 0.05,
    max_iterations: int = 200,
    position_tol: float = 1e-4,
    orientation_tol: float = 1e-3,
    step_limit: float = 0.3,
) -> IKResult:
    """Iterative DLS IK from q_start towards target_pose (4x4, DH base frame).

    Warm-starting at the current configuration is what makes an iterative
    method the right fit for teleop: it inherently returns the solution branch
    *nearest to where the arm already is*, so the arm never teleports between
    the UR5e's eight analytic IK branches. For tracking a moving target, call
    with max_iterations=1..3 per control tick.

    Tuning parameters (all overridable, defaults reasoned not magic):
      damping            lambda in the DLS formula. 0.05 bounds the worst-case
                         gain at 1/(2*0.05) = 10, i.e. a 1 cm task error can
                         never demand more than ~0.1 rad in the weakest
                         direction. Larger = calmer near singularities but
                         slower convergence everywhere.
      adaptive_damping   Chiaverini scheme: no damping while the smallest
                         singular value is above singular_threshold, ramped in
                         smoothly below it. Accuracy far from singularities,
                         stability near them.
      step_limit         per-iteration cap on max |dq_i| [rad]; keeps the
                         linearisation honest (J is only valid locally).
      position_tol       0.1 mm - well below anything visible in the sim task.
      orientation_tol    1e-3 rad (~0.06 deg).
    """
    if limits is not None and not limits.contains(np.asarray(q_start, dtype=float)):
        q_start = limits.clip(q_start)

    q = np.array(q_start, dtype=float)
    target_position = target_pose[:3, 3]
    target_rotation = target_pose[:3, :3]

    pos_err = math.inf
    rot_err = math.inf
    sigma_min = 0.0
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        frames = fk_frames(dh, q)
        current = frames[-1]
        error_pos = target_position - current[:3, 3]
        error_rot = rotation_vector_from_matrix(target_rotation @ current[:3, :3].T)
        pos_err = float(np.linalg.norm(error_pos))
        rot_err = float(np.linalg.norm(error_rot))

        jac = geometric_jacobian(dh, q, frames)
        sigma_min = float(np.linalg.svd(jac, compute_uv=False)[-1])

        if pos_err < position_tol and rot_err < orientation_tol:
            return IKResult(q, True, iterations, pos_err, rot_err, sigma_min)

        if adaptive_damping:
            # lambda^2 ramps from 0 (healthy) to damping^2 (singular) as
            # sigma_min falls below the threshold. (Chiaverini 1994.)
            if sigma_min >= singular_threshold:
                lam_sq = 0.0
            else:
                lam_sq = (1.0 - (sigma_min / singular_threshold) ** 2) * damping**2
            lam_sq += 1e-12  # numerical floor: keep the 6x6 solve well-posed
        else:
            lam_sq = damping**2

        error = np.concatenate([error_pos, error_rot])
        # dq = J^T (J J^T + lambda^2 I)^-1 e  - solved, never inverted explicitly
        dq = jac.T @ np.linalg.solve(jac @ jac.T + lam_sq * np.eye(6), error)

        biggest = float(np.max(np.abs(dq)))
        if biggest > step_limit:
            dq *= step_limit / biggest

        q = q + dq
        if limits is not None:
            q = limits.clip(q)

    return IKResult(q, False, iterations, pos_err, rot_err, sigma_min)
