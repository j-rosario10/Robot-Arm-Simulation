"""Robotiq 2F-85 drive model for PyBullet.

The URDF expresses the gripper's coupling with ``<mimic>`` tags: one driven
knuckle plus five joints slaved to it. PyBullet parses the tag and ignores
it, so the coupling has to be recreated. Two ways to do that:

  1. JOINT_GEAR constraints (the usual recipe found in PyBullet examples).
     Tried first here, and rejected: the finger links weigh ~40 g with
     inertias around 1e-5 kg m^2, so the constraint solver's corrective
     torques produce astronomical angular accelerations. The instability
     propagates up the arm and tears the whole robot apart within a second.

  2. Explicit position mirroring, used here: every follower joint is
     position-controlled to track the *measured* driven angle each step. No
     constraint solver involvement, nothing to tune, unconditionally stable.

Tracking the measured angle rather than the commanded one matters: when the
jaws close on a part the driven knuckle stalls part-way, and the followers
stall with it, so the finger pads stay parallel to the part instead of
folding past it.

Command mapping: the task layer speaks in *jaw opening* [m]; the simulator
needs a knuckle angle [rad]. The mapping is linear between the URDF's own
joint limits and SPEC.md's jaw stroke - an approximation (the true relation
is the four-bar linkage's nonlinear one), documented in
docs/phase3-simulation.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybullet as p

DRIVEN_JOINT = "robotiq_85_left_knuckle_joint"

# URDF mimic multipliers, relative to the driven joint (read off the model).
MIMIC_MULTIPLIERS = {
    "robotiq_85_right_knuckle_joint": -1.0,
    "robotiq_85_left_inner_knuckle_joint": 1.0,
    "robotiq_85_right_inner_knuckle_joint": -1.0,
    "robotiq_85_left_finger_tip_joint": -1.0,
    "robotiq_85_right_finger_tip_joint": 1.0,
}


@dataclass(frozen=True)
class GripperModel:
    """Handles and calibration for the mounted gripper."""

    body: int
    driven_index: int
    mimic_indices: dict          # joint index -> URDF mimic multiplier
    closed_angle: float          # rad, jaws touching
    open_angle: float            # rad, jaws at full stroke
    stroke: float                # m, from SPEC
    close_speed: float           # m/s at the jaws, from SPEC
    max_joint_torque: float      # Nm ceiling, derived from SPEC grip force
    lever_arm: float             # m, knuckle -> finger offset used for the conversion

    @property
    def angular_rate_limit(self) -> float:
        """SPEC's jaw close speed [m/s] expressed as a knuckle rate [rad/s].

        Uses the same linear stroke-to-angle mapping as the command path, so
        the two stay consistent: a full stroke takes stroke/close_speed
        seconds and sweeps the whole joint span.
        """
        span = abs(self.open_angle - self.closed_angle)
        return span * self.close_speed / self.stroke

    def angle_for_opening(self, opening: float) -> float:
        """Knuckle angle [rad] for a commanded jaw opening [m]."""
        fraction = float(np.clip(opening / self.stroke, 0.0, 1.0))
        return self.closed_angle + fraction * (self.open_angle - self.closed_angle)

    def opening_for_angle(self, angle: float) -> float:
        """Jaw opening [m] implied by a knuckle angle [rad] (inverse of above)."""
        span = self.open_angle - self.closed_angle
        return float(np.clip((angle - self.closed_angle) / span, 0.0, 1.0)) * self.stroke


def build_gripper(body: int, joint_index: dict, joint_info: dict, spec_gripper) -> GripperModel:
    """Derive the command calibration for the mounted gripper.

    ``joint_index`` maps joint name -> index; ``joint_info`` maps joint name ->
    the tuple from getJointInfo (used for the URDF joint limits).
    """
    # Joint limits come from the model, never invented: the driven knuckle's
    # URDF lower limit is the fully open pose, the upper limit fully closed.
    lower_limit = joint_info[DRIVEN_JOINT][8]
    upper_limit = joint_info[DRIVEN_JOINT][9]

    # Grip force -> knuckle torque. SPEC gives a force at the pads (60 N); the
    # simulator wants a torque ceiling on a revolute joint, so the two are
    # related by the linkage's moment arm. The knuckle-to-finger offset from
    # the URDF is the honest first-order estimate of that arm.
    lever_arm = float(np.linalg.norm(joint_info["robotiq_85_left_finger_joint"][14]))

    return GripperModel(
        body=body,
        driven_index=joint_index[DRIVEN_JOINT],
        mimic_indices={joint_index[name]: mult for name, mult in MIMIC_MULTIPLIERS.items()},
        closed_angle=upper_limit,
        open_angle=lower_limit,
        stroke=spec_gripper.stroke,
        close_speed=spec_gripper.close_speed,
        max_joint_torque=spec_gripper.max_force * lever_arm,
        lever_arm=lever_arm,
    )


def command_opening(model: GripperModel, opening: float) -> None:
    """Drive the jaws toward ``opening`` [m], respecting SPEC's close speed.

    The force ceiling is SPEC's grip force converted to a joint torque - this
    is the *only* place grip force enters, and it is a fixed property, never
    modulated live (vision teleop has no force feedback; see the README's
    limitations section).
    """
    p.setJointMotorControl2(
        model.body,
        model.driven_index,
        p.POSITION_CONTROL,
        targetPosition=model.angle_for_opening(opening),
        force=model.max_joint_torque,
        maxVelocity=model.angular_rate_limit,
    )


def sync_mimic_joints(model: GripperModel) -> None:
    """Make the five slaved joints follow the driven joint's measured angle.

    Called once per simulation step - this is what stands in for PyBullet's
    missing ``<mimic>`` support.
    """
    driven_angle = p.getJointState(model.body, model.driven_index)[0]
    indices = list(model.mimic_indices)
    p.setJointMotorControlArray(
        model.body,
        indices,
        p.POSITION_CONTROL,
        targetPositions=[model.mimic_indices[i] * driven_angle for i in indices],
        forces=[model.max_joint_torque] * len(indices),
    )


def measured_opening(model: GripperModel) -> float:
    """Current jaw opening [m] read back from the driven joint."""
    angle = p.getJointState(model.body, model.driven_index)[0]
    return model.opening_for_angle(angle)
