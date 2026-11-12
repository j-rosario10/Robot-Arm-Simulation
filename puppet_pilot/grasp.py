"""Sticky grasping and the bin counter.

Real friction grasping is hard to get reliable in a simulator. So instead:
when the gripper closes with an object between its fingers, the object is
"welded" to the gripper. When the gripper opens, the weld is switched off and
the object falls. Simple and dependable.
"""

import mujoco
import numpy as np

from puppet_pilot.scene import (BIN_CENTRE, BIN_HALF_SIZE, BIN_HEIGHT, GRIPPER_BODY,
                                OBJECTS, TCP_SITE)

GRAB_RADIUS = 0.045   # an object this close to the fingertips can be grabbed (metres)


class Grasper:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self._site = model.site(TCP_SITE).id
        self._gripper = model.body(GRIPPER_BODY).id
        self._bodies = {o.name: model.body(o.name).id for o in OBJECTS}
        self._welds = {o.name: model.equality(f"hold_{o.name}").id for o in OBJECTS}
        self.held = None          # name of the object being carried, or None
        self.in_bin = set()       # names of objects that have landed in the bin
        self._was_closed = False

    def reset(self) -> None:
        self.held, self._was_closed = None, False
        self.in_bin.clear()

    def update(self, data: mujoco.MjData, closed: bool) -> None:
        """Call every control tick with the current gripper command."""
        if closed and not self._was_closed and self.held is None:
            self._try_grab(data)              # the moment the gripper closes
        elif not closed and self.held is not None:
            data.eq_active[self._welds[self.held]] = 0   # let go
            self.held = None
        self._was_closed = closed
        self._count_bin(data)

    def _try_grab(self, data: mujoco.MjData) -> None:
        tip = data.site_xpos[self._site]
        # pick the nearest object, if it is close enough
        name, dist = min(((n, np.linalg.norm(data.xpos[b] - tip)) for n, b in self._bodies.items()),
                         key=lambda item: item[1])
        if dist > GRAB_RADIUS:
            return

        # Record where the object is RIGHT NOW relative to the gripper, so the
        # weld holds it exactly there instead of snapping it somewhere else.
        body = self._bodies[name]
        neg_pos, neg_quat = np.zeros(3), np.zeros(4)
        rel_pos, rel_quat = np.zeros(3), np.zeros(4)
        mujoco.mju_negPose(neg_pos, neg_quat, data.xpos[self._gripper], data.xquat[self._gripper])
        mujoco.mju_mulPose(rel_pos, rel_quat, neg_pos, neg_quat, data.xpos[body], data.xquat[body])

        weld = self._welds[name]
        eq_data = self.model.eq_data[weld]
        eq_data[0:3] = 0.0          # anchor point (in the object)
        eq_data[3:6] = rel_pos      # object position relative to the gripper
        eq_data[6:10] = rel_quat    # object orientation relative to the gripper
        eq_data[10] = 1.0           # how strongly rotation is held too
        data.eq_active[weld] = 1
        self.held = name
        self.in_bin.discard(name)   # picked back out of the bin

    def _count_bin(self, data: mujoco.MjData) -> None:
        for name, body in self._bodies.items():
            if name == self.held:
                continue
            x, y, z = data.xpos[body]
            inside = (abs(x - BIN_CENTRE[0]) < BIN_HALF_SIZE and
                      abs(y - BIN_CENTRE[1]) < BIN_HALF_SIZE and z < BIN_HEIGHT)
            if inside:
                self.in_bin.add(name)
            else:
                self.in_bin.discard(name)
