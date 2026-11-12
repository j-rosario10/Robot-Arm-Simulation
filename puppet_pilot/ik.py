"""Inverse kinematics: "which joint angles put the gripper HERE?"

The arm has six joints, but we think in terms of where the fingertips are.
IK converts one to the other. The method is damped least squares:

  1. Ask MuJoCo how the fingertip would move if each joint turned a little.
     That table of numbers is called the Jacobian (J).
  2. Measure the error: how far, and how rotated, the fingertip is from
     where we want it.
  3. Solve for the small joint change that shrinks that error:
         step = J^T (J J^T + damping^2 I)^-1 error
  4. Repeat a few times.

The "damping" stops the maths blowing up when the arm is stretched out or
folded on itself (a singularity). Without it the arm would jerk violently.
"""

import mujoco
import numpy as np

from puppet_pilot.scene import ARM_JOINTS, TCP_SITE


class ArmIK:
    def __init__(self, model: mujoco.MjModel, damping: float = 0.05):
        self.model = model
        # A private copy of the simulation state to do trial calculations on,
        # so solving IK never disturbs the real physics.
        self._scratch = mujoco.MjData(model)
        self._site = model.site(TCP_SITE).id
        self._qadr = [model.joint(j).qposadr[0] for j in ARM_JOINTS]   # where each joint lives in qpos
        self._dadr = [model.joint(j).dofadr[0] for j in ARM_JOINTS]    # ...and in the Jacobian columns
        self._lo = np.array([model.joint(j).range[0] for j in ARM_JOINTS])
        self._hi = np.array([model.joint(j).range[1] for j in ARM_JOINTS])
        self._damping_sq = damping ** 2
        self._jac_pos = np.zeros((3, model.nv))
        self._jac_rot = np.zeros((3, model.nv))
        self.down_quat = None     # the "pointing straight down" orientation

    def lock_orientation(self, data: mujoco.MjData) -> None:
        """Remember the gripper's current orientation as the one to keep."""
        self.down_quat = np.zeros(4)
        mujoco.mju_mat2Quat(self.down_quat, data.site_xmat[self._site])

    def solve(self, q_start, target_pos, iterations: int = 4, max_step: float = 0.15):
        """Returns joint angles that move the fingertip toward target_pos while
        keeping it pointing down. Starts from q_start (the current command),
        so the answer is always close to where the arm already is."""
        d = self._scratch
        q = np.array(q_start, dtype=float)
        target_pos = np.asarray(target_pos, dtype=float)
        error = np.zeros(6)
        site_quat, quat_conj, quat_err = np.zeros(4), np.zeros(4), np.zeros(4)

        for _ in range(iterations):
            d.qpos[self._qadr] = q
            mujoco.mj_kinematics(self.model, d)      # where is everything for these angles?
            mujoco.mj_comPos(self.model, d)

            # position error (metres)
            error[:3] = target_pos - d.site_xpos[self._site]
            # orientation error (as a small rotation vector)
            mujoco.mju_mat2Quat(site_quat, d.site_xmat[self._site])
            mujoco.mju_negQuat(quat_conj, site_quat)
            mujoco.mju_mulQuat(quat_err, self.down_quat, quat_conj)
            mujoco.mju_quat2Vel(error[3:], quat_err, 1.0)

            mujoco.mj_jacSite(self.model, d, self._jac_pos, self._jac_rot, self._site)
            jac = np.vstack([self._jac_pos[:, self._dadr], self._jac_rot[:, self._dadr]])

            # damped least squares step
            step = jac.T @ np.linalg.solve(jac @ jac.T + self._damping_sq * np.eye(6), error)
            biggest = np.max(np.abs(step))
            if biggest > max_step:                   # never turn a joint too far in one go
                step *= max_step / biggest
            q = np.clip(q + step, self._lo, self._hi)
        return q
