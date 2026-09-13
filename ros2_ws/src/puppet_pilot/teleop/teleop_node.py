"""Teleop controller: operator topic -> retarget -> IK -> joint commands.

The control chain per tick (SPEC's 30 Hz):
    operator sample -> EMA smooth -> axis map -> workspace map (Retargeter)
    -> warm-started DLS IK (<=3 iterations, adaptive damping)
    -> capsule self-collision veto
    -> /puppet_pilot/joint_command (consumed by the sim node/session)

Safety behaviours:
  - stale operator data (>0.3 s): hold the last commanded pose
  - IK not fully converged: command the bounded best-effort step anyway
    (that is what DLS is *for* - graceful degradation near singularities)
  - self-collision predicted: hold the last safe command instead

Run:  python -m puppet_pilot.teleop.teleop_node
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from puppet_pilot.kinematics.collision import CollisionModelConfig, SelfCollisionChecker
from puppet_pilot.kinematics.ik import solve_ik
from puppet_pilot.kinematics.params import DHTable, JointLimits, LinkGeometry
from puppet_pilot.spec import load_spec, repo_root
from puppet_pilot.teleop.bridge_node import OPERATOR_TOPIC
from puppet_pilot.teleop.retarget import Retargeter, RetargetConfig

# These mirror sim/world.py; restated here so the teleop node stays free of
# any PyBullet import (it may run against real hardware one day).
ARM_JOINT_NAMES = [
    "ur5e_shoulder_pan_joint", "ur5e_shoulder_lift_joint", "ur5e_elbow_joint",
    "ur5e_wrist_1_joint", "ur5e_wrist_2_joint", "ur5e_wrist_3_joint",
]
GRIPPER_JOINT = "robotiq_85_left_knuckle_joint"
JOINT_COMMAND_TOPIC = "/puppet_pilot/joint_command"
JOINT_STATE_TOPIC = "/puppet_pilot/joint_states"

WORLD_FROM_DH = np.diag([-1.0, -1.0, 1.0, 1.0])
TOOL_POINTING_DOWN = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])

# tool0 -> pad-tip distance. The simulator measures this from the loaded
# model and reports it in its startup log; until a live value arrives on
# /puppet_pilot/robot_info this default (the measured 2F-85 value) is used.
DEFAULT_JAW_OFFSET = 0.152

_COLLISION_MARGIN = 0.005  # m of clearance the veto insists on


class TeleopNode(Node):
    def __init__(self, jaw_offset: float):
        super().__init__("puppet_pilot_teleop")
        self.spec = load_spec()
        self.retargeter = Retargeter(self.spec, RetargetConfig())
        self.dh = DHTable.from_yaml()
        self.limits = JointLimits.from_yaml()
        self.checker = SelfCollisionChecker(
            self.dh, LinkGeometry.from_yaml(), CollisionModelConfig.from_yaml()
        )
        self.jaw_offset = jaw_offset

        self.q_command: np.ndarray | None = None   # set from the first joint state
        self.opening = self.spec.gripper.stroke
        self.sample: np.ndarray | None = None
        self.sample_time = 0.0
        self.vetoed = 0

        self.create_subscription(Float64MultiArray, OPERATOR_TOPIC, self._on_operator, 10)
        self.create_subscription(JointState, JOINT_STATE_TOPIC, self._on_joint_state, 10)
        self.command_pub = self.create_publisher(JointState, JOINT_COMMAND_TOPIC, 10)
        self.create_timer(1.0 / self.spec.teleop.publish_rate, self._tick)

        # Live axis flips from the control panel: the operator clicks a button
        # while driving and the mapping changes within a second. Which way
        # "feels right" depends on where the camera sits relative to the
        # operator, so this is a runtime setting, not code.
        self._base_axis_map = self.retargeter.config.axis_map.copy()
        self._flips = None
        self._smoothness = None
        self._max_step = None   # per-tick joint delta cap [rad], from the slider
        self._config_path = repo_root() / "control_panel" / "teleop_config.json"
        self.create_timer(1.0, self._reload_config)
        self._reload_config()
        self.get_logger().info(
            f"teleop up: {self.spec.teleop.publish_rate:.0f} Hz, "
            f"workspace {np.round(self.retargeter.workspace_min, 2)} .. "
            f"{np.round(self.retargeter.workspace_max, 2)}, jaw offset {jaw_offset:.3f} m"
        )

    # -- inputs ------------------------------------------------------------

    def _on_operator(self, message: Float64MultiArray) -> None:
        self.sample = np.array(message.data)
        self.sample_time = time.monotonic()

    def _on_joint_state(self, message: JointState) -> None:
        if self.q_command is not None:
            return  # only needed once, to seed the command state
        positions = dict(zip(message.name, message.position))
        try:
            self.q_command = np.array([positions[name] for name in ARM_JOINT_NAMES])
            self.get_logger().info("seeded from current joint state")
        except KeyError:
            pass

    def _reload_config(self) -> None:
        try:
            cfg = json.loads(self._config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cfg = {}

        flips = tuple(bool(cfg.get(f"flip_{axis}", False)) for axis in ("x", "y", "z"))
        if flips != self._flips:
            self._flips = flips
            signs = np.diag([-1.0 if f else 1.0 for f in flips])
            # Rows of the axis map are robot axes: flipping one is a row sign.
            self.retargeter.config.axis_map = signs @ self._base_axis_map
            self.get_logger().info(
                "axis flips: forward/back=%s left/right=%s up/down=%s" % flips
            )

        smoothness = float(np.clip(cfg.get("smoothness", 50), 0, 100)) / 100.0
        if smoothness != self._smoothness:
            self._smoothness = smoothness
            self._apply_smoothness(smoothness)

    def _apply_smoothness(self, s: float) -> None:
        """One slider, two knobs, both anchored on SPEC at the midpoint.

        s = 0.5 reproduces SPEC exactly (EMA alpha as written, 62% of the
        datasheet joint speed). Sliding up lowers alpha (heavier filtering of
        MediaPipe jitter) and tightens the per-tick joint speed cap (no
        snapping between IK solutions); sliding down does the opposite.
        """
        spec_alpha = self.spec.teleop.ema_alpha
        self.retargeter.alpha = float(np.clip(spec_alpha * 2.0 * (1.0 - s), 0.05, 0.9))
        velocity_fraction = 1.0 - 0.75 * s
        self._max_step = velocity_fraction * self.limits.velocity / self.spec.teleop.publish_rate
        self.get_logger().info(
            f"smoothness {s * 100:.0f}%: ema alpha {self.retargeter.alpha:.3f}, "
            f"joint speed cap {velocity_fraction * 100:.0f}% of limit"
        )

    # -- control tick --------------------------------------------------------

    def _tick(self) -> None:
        if self.q_command is None or self.sample is None:
            return
        stale = (time.monotonic() - self.sample_time) > self.retargeter.config.stale_after
        if not stale:
            wrist, pinch = self.sample[:3], self.sample[3]
            if self.sample[4] > 0.5:  # pose_ok: only move the arm on real data
                target, opening = self._solve(wrist, pinch)
                if target is not None:
                    self.q_command = target
            if self.sample[5] > 0.5:  # hand_ok
                self.opening = self.retargeter.opening_from_pinch(pinch)

        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = ARM_JOINT_NAMES + [GRIPPER_JOINT]
        message.position = list(self.q_command) + [self.opening]
        self.command_pub.publish(message)

    def _solve(self, wrist: np.ndarray, pinch: float) -> tuple:
        pad_target, _ = self.retargeter.update(wrist, pinch)

        flange = pad_target - TOOL_POINTING_DOWN[:, 2] * self.jaw_offset
        world_target = np.eye(4)
        world_target[:3, :3] = TOOL_POINTING_DOWN
        world_target[:3, 3] = flange
        dh_target = np.linalg.inv(WORLD_FROM_DH) @ world_target

        # Warm start from the last command; <=3 iterations keeps the tick
        # cheap and the motion continuous. Non-convergence is fine - the
        # bounded DLS step *is* the graceful behaviour near singularities
        # and workspace edges.
        result = solve_ik(
            self.dh, self.q_command, dh_target, limits=self.limits,
            max_iterations=3, adaptive_damping=True,
        )

        # Speed limit: the IK solution can jump (branch changes, singularity
        # escapes); the command may only move a bounded amount per tick, so
        # the arm glides rather than snaps. This is also what gets recorded as
        # the action label, which keeps the demonstrations smooth for learning.
        target = result.q
        if self._max_step is not None:
            delta = np.clip(target - self.q_command, -self._max_step, self._max_step)
            target = self.q_command + delta

        if self.checker.check(target, margin=_COLLISION_MARGIN).colliding:
            self.vetoed += 1
            if self.vetoed % 30 == 1:
                self.get_logger().warning("self-collision veto active - holding safe pose")
            return None, None
        return target, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jaw-offset", type=float, default=DEFAULT_JAW_OFFSET)
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = TeleopNode(jaw_offset=args.jaw_offset)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
