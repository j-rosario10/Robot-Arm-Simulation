"""FK validation: DH table vs published values, DH chain vs the URDF's own FK."""

import math

import numpy as np
import pybullet
import pytest

from puppet_pilot.kinematics.fk import fk_frames, geometric_jacobian, tool_pose
from puppet_pilot.kinematics.ik import rotation_vector_from_matrix


def test_dh_table_matches_published_ur5e_values(dh):
    """The loader's mapping must reproduce UR's published UR5e DH table.

    Reference values from Universal Robots, "DH parameters for calculations of
    kinematics and dynamics" (the same numbers ur_description encodes in
    default_kinematics.yaml). Hardcoding them HERE is the point of the test:
    an independent statement of what the loader should produce.
    """
    np.testing.assert_allclose(dh.d, [0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996], atol=1e-6)
    np.testing.assert_allclose(dh.a, [0.0, -0.425, -0.3922, 0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(
        dh.alpha, [math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0], atol=1e-6
    )


def _pose_matrix(position, quaternion):
    t = np.eye(4)
    t[:3, :3] = np.array(pybullet.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
    t[:3, 3] = position
    return t


class TestFKAgainstURDF:
    """DH forward kinematics must agree with PyBullet's FK on the built URDF.

    The two chains use different base and tool frame conventions, so the
    comparison is  T_urdf(q) == A @ T_dh(q) @ B  with A, B *constant*:
      A = world -> base_link_inertia (UR mounts the DH chain rotated pi about z
          relative to base_link, to match the teach pendant frame)
      B = DH flange frame -> tool0 (a pure rotation; tool0 is co-located).
    A is read from the URDF directly; B is solved at q=0 and then required to
    hold for every random configuration - which proves both corrections are
    truly constant and the chains describe the same robot.
    """

    @pytest.fixture(scope="class")
    def sim(self, kinematic_urdf_path):
        client = pybullet.connect(pybullet.DIRECT)
        robot = pybullet.loadURDF(kinematic_urdf_path, useFixedBase=True)
        joints = {}
        links = {}
        for i in range(pybullet.getNumJoints(robot)):
            info = pybullet.getJointInfo(robot, i)
            joints[info[1].decode()] = i
            links[info[12].decode()] = i
        arm = [joints[f"ur5e_{n}"] for n in (
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        )]
        yield robot, arm, links
        pybullet.disconnect(client)

    def _urdf_pose(self, robot, links, link_name):
        state = pybullet.getLinkState(robot, links[link_name], computeForwardKinematics=1)
        return _pose_matrix(state[4], state[5])  # world *frame* pose, not CoM

    def _set_arm(self, robot, arm, q):
        for joint_index, angle in zip(arm, q):
            pybullet.resetJointState(robot, joint_index, angle)

    def test_fk_matches_urdf_over_random_configs(self, dh, sim, rng):
        robot, arm, links = sim

        self._set_arm(robot, arm, np.zeros(6))
        base_correction = self._urdf_pose(robot, links, "ur5e_base_link_inertia")
        tool_correction = (
            np.linalg.inv(tool_pose(dh, np.zeros(6)))
            @ np.linalg.inv(base_correction)
            @ self._urdf_pose(robot, links, "ur5e_tool0")
        )

        # The base correction is the documented pi-flip about z, at the origin.
        np.testing.assert_allclose(base_correction[:3, 3], 0.0, atol=1e-8)
        expected_flip = np.diag([-1.0, -1.0, 1.0])
        np.testing.assert_allclose(base_correction[:3, :3], expected_flip, atol=1e-6)
        # tool0 coincides with the DH flange frame exactly: the URDF's
        # wrist_3->flange rpy(0,-pi/2,-pi/2) and flange->tool0 rpy(pi/2,0,pi/2)
        # compose to identity.
        np.testing.assert_allclose(tool_correction[:3, 3], 0.0, atol=1e-8)
        np.testing.assert_allclose(tool_correction[:3, :3], np.eye(3), atol=1e-6)

        for _ in range(100):
            q = rng.uniform(-math.pi, math.pi, 6)
            self._set_arm(robot, arm, q)
            expected = self._urdf_pose(robot, links, "ur5e_tool0")
            actual = base_correction @ tool_pose(dh, q) @ tool_correction
            np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_jacobian_matches_finite_differences(dh, rng):
    """The geometric Jacobian must predict the pose change of a small joint step."""
    for _ in range(20):
        q = rng.uniform(-math.pi, math.pi, 6)
        dq = rng.normal(0.0, 1.0, 6) * 1e-6
        jac = geometric_jacobian(dh, q)

        before = fk_frames(dh, q)[-1]
        after = fk_frames(dh, q + dq)[-1]
        measured_linear = after[:3, 3] - before[:3, 3]
        measured_angular = rotation_vector_from_matrix(after[:3, :3] @ before[:3, :3].T)

        predicted = jac @ dq
        np.testing.assert_allclose(predicted[:3], measured_linear, atol=1e-9)
        np.testing.assert_allclose(predicted[3:], measured_angular, atol=1e-9)
