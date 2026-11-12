"""ROS 2 node 2 of 2: arm_sim.

A thin wrapper around puppet_pilot/sim.py. It receives the target from the
hand tracker, runs the physics in real time, and shows the MuJoCo window.

Three things run at their own pace so none can stall the others:
    ROS messages   arrive in a background thread and are just stored
    physics        500 steps per second, kept in step with the real clock
    drawing        60 times per second

Listens to:  /hand/target, /hand/gripper_closed, /sim/reset
Publishes:   /sim/status   (bin count, what is held, speed, delay)
"""

import json
import threading
import time

import mujoco
import mujoco.viewer
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from std_msgs.msg import Bool, Empty, String

from puppet_pilot import scene
from puppet_pilot.sim import ArmSim

DRAW_HZ = 60
STATUS_HZ = 10
KEY_R = 82   # key code the viewer reports for the R key


class ArmSimNode(Node):
    def __init__(self):
        super().__init__("arm_sim")
        self.sim = ArmSim()
        self._lock = threading.Lock()     # protects the three values below
        self._target = None
        self._closed = False
        self._reset_requested = False
        self._latency_ms = 0.0
        self._commands = 0                # how many targets have arrived

        self.create_subscription(PointStamped, "/hand/target", self._on_target, 1)
        self.create_subscription(Bool, "/hand/gripper_closed", self._on_gripper, 1)
        self.create_subscription(Empty, "/sim/reset", self._on_reset, 1)
        self.status_pub = self.create_publisher(String, "/sim/status", 1)

    # These three run in the ROS thread. They only store the newest value.
    def _on_target(self, msg: PointStamped) -> None:
        sent = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self._lock:
            self._target = (msg.point.x, msg.point.y, msg.point.z)
            self._commands += 1
            self._latency_ms = 0.9 * self._latency_ms + 0.1 * (time.time() - sent) * 1000

    def _on_gripper(self, msg: Bool) -> None:
        with self._lock:
            self._closed = msg.data

    def _on_reset(self, _msg=None) -> None:
        with self._lock:
            self._reset_requested = True

    def _on_key(self, key: int) -> None:
        if key == KEY_R:
            self._on_reset()

    def run(self) -> None:
        sim = self.sim
        with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=self._on_key,
                                          show_left_ui=False, show_right_ui=False) as viewer:
            # Camera in front of the robot, looking back at it and the table.
            viewer.cam.lookat[:] = [0.45, 0.05, 0.12]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.5, 180, -28

            start = time.perf_counter()
            sim_start = sim.data.time
            next_draw = next_status = start
            steps_done, fps_mark = 0, start

            while viewer.is_running() and rclpy.ok():
                with self._lock:              # take the newest command
                    target, closed = self._target, self._closed
                    reset, self._reset_requested = self._reset_requested, False

                with viewer.lock():           # don't let the window draw mid-step
                    if reset:
                        sim.reset()
                        with self._lock:
                            self._target = None   # hold home until the hand moves it
                        start, sim_start = time.perf_counter(), sim.data.time
                    elif target is not None:
                        sim.set_command(target, closed)

                    # Step until the simulation clock catches up with the real
                    # clock. Capped, so one slow moment can't snowball.
                    behind = (time.perf_counter() - start) - (sim.data.time - sim_start)
                    for _ in range(min(int(behind / sim.timestep), 20)):
                        sim.step()
                        steps_done += 1

                now = time.perf_counter()
                if now >= next_draw:
                    viewer.sync()             # redraw the window
                    next_draw = now + 1.0 / DRAW_HZ
                if now >= next_status:
                    with self._lock:
                        commands, self._commands = self._commands, 0
                    self._publish_status(steps_done / (now - fps_mark), commands / (now - fps_mark))
                    steps_done, fps_mark = 0, now
                    next_status = now + 1.0 / STATUS_HZ
                time.sleep(0.0005)            # give other threads a turn

    def _publish_status(self, sim_fps: float, command_hz: float) -> None:
        g = self.sim.grasper
        with self._lock:
            latency = self._latency_ms
        self.status_pub.publish(String(data=json.dumps({
            "in_bin": len(g.in_bin), "objects": len(scene.OBJECTS), "held": g.held, "near": g.near,
            "sim_fps": sim_fps, "command_hz": command_hz, "latency_ms": latency,
        })))


def main() -> None:
    rclpy.init()
    node = ArmSimNode()
    # ROS message handling gets its own thread, so the physics loop never waits on it.
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
