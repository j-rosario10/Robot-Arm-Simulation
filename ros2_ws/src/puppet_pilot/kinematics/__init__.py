"""From-scratch UR5e kinematics core.

Public API:
    params      - loaders for the DH table, joint limits, link geometry (all
                  sourced from ur_description's YAML files at runtime)
    fk          - forward kinematics and the geometric Jacobian
    ik          - damped least-squares inverse kinematics
    trajectory  - quintic (C2) joint-space trajectories
    collision   - capsule-based self-collision checking
"""

from puppet_pilot.kinematics.params import (
    JOINT_NAMES,
    DHTable,
    JointLimits,
    LinkGeometry,
    ur5e_config_dir,
)
from puppet_pilot.kinematics.fk import dh_transform, fk_frames, tool_pose, geometric_jacobian
from puppet_pilot.kinematics.ik import IKResult, solve_ik
from puppet_pilot.kinematics.trajectory import QuinticSegment, JointTrajectory, plan_rest_to_rest
from puppet_pilot.kinematics.collision import Capsule, segment_segment_distance, SelfCollisionChecker
