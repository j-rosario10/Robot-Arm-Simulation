"""ROS 2 node 1 of 2: hand_tracker.

A thin wrapper. All the real work is in puppet_pilot/tracker.py. This file
publishes the results as ROS messages and shows the preview window.

Two loops run side by side:
    tracking thread   camera frame -> hand -> publish. Never waits for the screen.
    main thread       draws the preview window and reads the keyboard.
Drawing a window can stall (Windows slows it right down when the window is
covered up), so it must not sit between the camera and the robot.

Publishes:
    /hand/target          where the gripper should go (x, y, z in metres)
    /hand/gripper_closed  True = close the gripper
    /sim/reset            sent when you press R
Listens to:
    /sim/status           bin count and speed, shown on the preview
"""

import json
import threading
import time

import cv2
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from std_msgs.msg import Bool, Empty, String

from puppet_pilot.tracker import WINDOW, HandPipeline, draw_status


class HandTrackerNode(Node):
    def __init__(self):
        super().__init__("hand_tracker")
        self.pipeline = HandPipeline()
        self.target_pub = self.create_publisher(PointStamped, "/hand/target", 1)
        self.gripper_pub = self.create_publisher(Bool, "/hand/gripper_closed", 1)
        self.reset_pub = self.create_publisher(Empty, "/sim/reset", 1)
        self.create_subscription(String, "/sim/status", self._on_status, 1)
        self.status = {}          # latest numbers from the simulator
        self._latest = None       # newest (preview image, gripper closed), for the window
        self._running = True

    def _on_status(self, msg: String) -> None:
        self.status = json.loads(msg.data)

    def _track_loop(self) -> None:
        """Tracking thread: turn every new camera frame into a command, at once."""
        while self._running and rclpy.ok():
            out = self.pipeline.tick()
            if out is None:
                time.sleep(0.001)                 # no new frame yet
                continue
            frame, (x, y, z), closed, captured_at = out

            # Stamp the message with when the camera frame was taken, so the
            # simulator can measure the total camera-to-arm delay.
            msg = PointStamped()
            msg.header.stamp.sec = int(captured_at)
            msg.header.stamp.nanosec = int((captured_at % 1) * 1e9)
            msg.point.x, msg.point.y, msg.point.z = x, y, z
            self.target_pub.publish(msg)
            self.gripper_pub.publish(Bool(data=closed))
            self._latest = (frame, closed)        # hand the picture to the window

    def _status_lines(self) -> list:
        """The simulator's numbers, as text for the preview window."""
        s = self.status
        if not s:
            return ["waiting for the simulator..."]
        if s["held"]:
            doing = f"holding {s['held'].replace('_', ' ')} - open hand to drop"
        elif s["near"]:
            doing = f"pinch to grab {s['near'].replace('_', ' ')}"
        else:
            doing = "move over an object"
        return [f"IN BIN {s['in_bin']} / {s['objects']}", doing,
                f"sim {s['sim_fps']:.0f} steps/s   camera-to-arm {s['latency_ms']:.0f} ms"]

    def run(self) -> None:
        """Main thread: the preview window and the keyboard."""
        tracking = threading.Thread(target=self._track_loop, daemon=True)
        tracking.start()

        # Small window in the top-left corner that stays above the simulator.
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(WINDOW, 10, 10)
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0)      # pick up any waiting messages
            latest, self._latest = self._latest, None
            if latest is not None:
                frame, closed = latest
                draw_status(frame, self.pipeline, closed, self._status_lines())
                cv2.imshow(WINDOW, cv2.resize(frame, (480, 360)))

            key = cv2.waitKey(5) & 0xFF               # also lets the window redraw
            if key == ord("c"):
                self.pipeline.mapping.recentre()
            elif key == ord("r"):
                self.reset_pub.publish(Empty())
            elif key == ord("q"):
                break

        self._running = False
        tracking.join(timeout=1.0)


def main() -> None:
    rclpy.init()
    node = HandTrackerNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node._running = False
        node.pipeline.close()
        cv2.destroyAllWindows()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
