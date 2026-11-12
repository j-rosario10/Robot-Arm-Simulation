"""The simulation as one object: give it a target, step it, read the result.

This ties together the scene, the IK solver and the grasp logic. It knows
nothing about ROS or the webcam, so it can be driven by a test, a script or
the ROS node equally well.
"""

import mujoco
import numpy as np

from puppet_pilot import scene
from puppet_pilot.grasp import Grasper
from puppet_pilot.ik import ArmIK

CONTROL_EVERY = 4     # run IK once per 4 physics steps (physics 500 Hz, control 125 Hz)


class ArmSim:
    def __init__(self):
        self.model = scene.build_model()
        self.data = mujoco.MjData(self.model)
        self.ik = ArmIK(self.model)
        self.grasper = Grasper(self.model)
        self._gripper_ctrl = self.model.actuator(scene.GRIPPER_ACTUATOR).id
        self._tcp = self.model.site(scene.TCP_SITE).id
        self._steps = 0
        self.reset()

    def reset(self) -> None:
        """Arm to home, objects back on the table, counter to zero."""
        scene.reset(self.model, self.data)
        self.ik.lock_orientation(self.data)          # "down" = how the gripper sits at home
        self.target = self.data.site_xpos[self._tcp].copy()
        self.closed = False
        self.grasper.reset()

    def set_command(self, target_xyz, closed: bool) -> None:
        self.target = np.asarray(target_xyz, dtype=float)
        self.closed = bool(closed)

    def step(self) -> None:
        """Advance the physics by one small time step (2 ms)."""
        if self._steps % CONTROL_EVERY == 0:
            # steer: new joint targets from IK, starting from the last ones
            self.data.ctrl[:6] = self.ik.solve(self.data.ctrl[:6], self.target)
            self.data.ctrl[self._gripper_ctrl] = 255 if self.closed else 0
            self.grasper.update(self.data, self.closed)
        mujoco.mj_step(self.model, self.data)
        self._steps += 1

    @property
    def tip_position(self) -> np.ndarray:
        return self.data.site_xpos[self._tcp]

    @property
    def timestep(self) -> float:
        return self.model.opt.timestep
