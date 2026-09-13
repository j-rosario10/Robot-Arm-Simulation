"""Socket -> ROS bridge: republishes the Windows pose stream as a topic.

This is the WSL half of the architectural seam: it knows about TCP and JSON
on one side and ROS on the other, and nothing else. Swapping the perception
source (depth camera, glove, a synthetic test driver) means pointing this
node at a different socket - the control side never notices.

Topic:  /puppet_pilot/operator  (std_msgs/Float64MultiArray)
        data = [wrist_x, wrist_y, wrist_z, pinch, pose_ok, hand_ok]
        (wrist in arm-lengths, camera axes; pinch 0..1 calibrated)

The Windows host's IP is discovered automatically (WSL2 NAT puts it in
/etc/resolv.conf); override with --host or PUPPET_PILOT_HOST.

Run:  python -m puppet_pilot.teleop.bridge_node [--host H] [--port 8877]
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

OPERATOR_TOPIC = "/puppet_pilot/operator"
DEFAULT_PORT = 8877  # must match windows_side/pose_publisher.py


def host_candidates() -> list:
    """Where the Windows host might be, as seen from WSL2, in trial order.

    1. PUPPET_PILOT_HOST if set (explicit override).
    2. The default-route gateway. Under WSL2 NAT networking this IS the
       Windows host (verified: 172.23.224.1 connects).
    3. resolv.conf's nameserver - the classic recipe, but newer WSL writes a
       DNS-tunnelling address there (10.255.255.254) that is NOT the host,
       which is why it is only a fallback now.
    4. Loopback - mirrored networking mode, or a local synthetic operator.
    """
    candidates: list = []
    override = os.environ.get("PUPPET_PILOT_HOST")
    if override:
        candidates.append(override)
    try:
        route = subprocess.run(["ip", "route", "show", "default"],
                               capture_output=True, text=True, timeout=2).stdout.split()
        if "via" in route:
            candidates.append(route[route.index("via") + 1])
    except (OSError, subprocess.SubprocessError):
        pass
    resolv = Path("/etc/resolv.conf")
    if resolv.exists():
        for line in resolv.read_text().splitlines():
            if line.startswith("nameserver"):
                candidates.append(line.split()[1])
    candidates.append("127.0.0.1")
    # de-duplicate, keeping order
    return list(dict.fromkeys(candidates))


class BridgeNode(Node):
    def __init__(self, hosts: list, port: int):
        super().__init__("puppet_pilot_bridge")
        self.hosts, self.port = hosts, port
        self.host = None  # the candidate that actually answered
        self.publisher = self.create_publisher(Float64MultiArray, OPERATOR_TOPIC, 10)
        self.socket = None
        self.buffer = b""
        self.published = 0
        self._connect_attempts = 0
        self.create_timer(0.005, self._poll)
        self.create_timer(5.0, self._report)
        self.get_logger().info(f"bridging {hosts} port {port} -> {OPERATOR_TOPIC}")

    def _connect(self) -> None:
        """Try every host candidate in order; keep the first that answers."""
        self._connect_attempts += 1
        for host in self.hosts:
            try:
                candidate = socket.create_connection((host, self.port), timeout=0.7)
            except OSError:
                continue
            candidate.setblocking(False)
            self.socket = candidate
            self.host = host
            self.buffer = b""
            self.get_logger().info(f"connected to pose publisher at {host}:{self.port}")
            return
        self.socket = None
        if self._connect_attempts % 10 == 1:
            self.get_logger().info(
                f"waiting for a pose publisher on {self.hosts} port {self.port} "
                "(is the camera started and past calibration?)"
            )
        time.sleep(0.5)  # do not hammer hosts that are not up yet

    def _poll(self) -> None:
        if self.socket is None:
            self._connect()
            return
        try:
            chunk = self.socket.recv(65536)
        except BlockingIOError:
            return
        except OSError:
            chunk = b""
        if not chunk:
            self.get_logger().warning("pose publisher disconnected; retrying")
            self.socket = None
            return

        self.buffer += chunk
        # Publish every complete line; a partial line waits for its rest.
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = Float64MultiArray()
            message.data = [
                float(sample["wrist"][0]), float(sample["wrist"][1]),
                float(sample["wrist"][2]), float(sample["pinch"]),
                1.0 if sample.get("pose_ok") else 0.0,
                1.0 if sample.get("hand_ok") else 0.0,
            ]
            self.publisher.publish(message)
            self.published += 1

    def _report(self) -> None:
        self.get_logger().info(f"{self.published / 5.0:.1f} msg/s")
        self.published = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = BridgeNode([args.host] if args.host else host_candidates(), args.port)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
