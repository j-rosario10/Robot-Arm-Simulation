"""ROS 2 node wrapping the simulation world.

Topics
    /puppet_pilot/joint_states   (sensor_msgs/JointState, published)
        Six arm joints plus the gripper's driven joint. ``position`` for the
        gripper entry is its jaw opening in metres, not the knuckle angle -
        the task layer thinks in stroke, and SPEC's numbers are in stroke.

    /puppet_pilot/joint_command  (sensor_msgs/JointState, subscribed)
        Same layout. Any subset of the named joints may be commanded; the
        gripper entry is again a jaw opening in metres.

The node owns the physics clock: it steps the simulator on a wall-clock timer
rather than stepping per received command, so the world keeps running (and
parts keep settling) whether or not anything is commanding the arm.

Run it with the project venv, which has both rclpy (system site-packages) and
pybullet:

    source ~/venvs/puppet_pilot/bin/activate
    source /opt/ros/jazzy/setup.bash
    python -m puppet_pilot.sim.sim_node --gui
"""

from __future__ import annotations

import argparse

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from puppet_pilot.sim.gripper import DRIVEN_JOINT
from puppet_pilot.sim.world import ARM_JOINT_NAMES, SortingWorld

JOINT_STATE_TOPIC = "/puppet_pilot/joint_states"
JOINT_COMMAND_TOPIC = "/puppet_pilot/joint_command"

# Publishing joint state at SPEC's demonstration log rate keeps the recorded
# demonstrations and the topic stream on the same cadence.
_PUBLISH_QUEUE = 10


class SimNode(Node):
    def __init__(self, gui: bool, seed: int):
        super().__init__("puppet_pilot_sim")
        self.world = SortingWorld.create(gui=gui, seed=seed)
        self.joint_names = ARM_JOINT_NAMES + [DRIVEN_JOINT]

        self.publisher = self.create_publisher(JointState, JOINT_STATE_TOPIC, _PUBLISH_QUEUE)
        self.create_subscription(
            JointState, JOINT_COMMAND_TOPIC, self._on_command, _PUBLISH_QUEUE
        )

        # Physics runs at the simulator's own timestep; state publishes at the
        # rate SPEC specifies for logging.
        # Last received commands, kept for recording (Phase 5): the commanded
        # joint positions are the action supervision for behaviour cloning.
        self.last_arm_command = None
        self.last_gripper_command = self.world.spec.gripper.stroke

        self.create_timer(self.world.timestep, self._step_physics)
        self.create_timer(1.0 / self.world.spec.recording.log_rate, self._publish_state)

        self.get_logger().info(
            f"simulation ready: {len(self.world.parts)} parts, seed {self.world.seed}"
        )

    def _step_physics(self) -> None:
        self.world.step()

    def _publish_state(self) -> None:
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = self.joint_names
        message.position = list(self.world.joint_positions()) + [self.world.gripper_opening()]
        message.velocity = list(self.world.joint_velocities()) + [0.0]
        self.publisher.publish(message)

    def _on_command(self, message: JointState) -> None:
        """Apply a command. Unknown joint names are ignored with a warning
        rather than dropping the whole message - a partial command (arm only,
        or gripper only) is a normal thing for the teleop layer to send."""
        arm_targets = dict(zip(ARM_JOINT_NAMES, self.world.joint_positions()))
        gripper_target = None

        for name, position in zip(message.name, message.position):
            if name in arm_targets:
                arm_targets[name] = position
            elif name == DRIVEN_JOINT:
                gripper_target = position
            else:
                self.get_logger().warning(f"ignoring unknown joint {name!r}")

        self.last_arm_command = [arm_targets[name] for name in ARM_JOINT_NAMES]
        self.world.command_joints(self.last_arm_command)
        if gripper_target is not None:
            self.last_gripper_command = gripper_target
            self.world.command_gripper(gripper_target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true", help="open the PyBullet window")
    parser.add_argument("--seed", type=int, default=0)
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = SimNode(gui=args.gui, seed=args.seed)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.world.disconnect()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
