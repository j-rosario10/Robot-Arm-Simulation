"""Synthetic operator: speaks the pose_publisher wire protocol without a camera.

Two jobs:
  - end-to-end testing of the bridge -> teleop -> sim chain on WSL alone
    (point the bridge at 127.0.0.1)
  - debugging the control side when the camera side is the suspect

Publishes a slow figure-of-eight wrist motion plus a periodic pinch, in the
same shoulder-relative arm-length units the real publisher uses.

Run:  python -m puppet_pilot.tools.synthetic_operator [--port 8877] [--rate 30]
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import threading
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--period", type=float, default=12.0, help="s per motion cycle")
    args = parser.parse_args()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", args.port))
    server.listen(1)
    clients: list = []

    def accept_loop() -> None:
        while True:
            client, address = server.accept()
            clients.append(client)
            print(f"client connected from {address[0]}")

    threading.Thread(target=accept_loop, daemon=True).start()
    print(f"synthetic operator on port {args.port} at {args.rate:.0f} Hz")

    start = time.monotonic()
    while True:
        phase = 2 * math.pi * ((time.monotonic() - start) / args.period)
        # Figure-of-eight in the camera plane, gentle push in depth. Camera
        # axes: x right, y down, z away - matches the real publisher.
        sample = {
            "t": time.time(),
            "pose_ok": True,
            "hand_ok": True,
            "wrist": [
                round(0.35 * math.sin(2 * phase), 4),           # x: side to side
                round(-0.15 + 0.20 * math.sin(phase), 4),       # y: up and down
                round(-0.45 - 0.15 * math.cos(phase), 4),       # z: toward camera
            ],
            "pinch": round(0.5 + 0.5 * math.sin(phase / 2), 4),
        }
        line = (json.dumps(sample) + "\n").encode()
        for client in clients[:]:
            try:
                client.sendall(line)
            except OSError:
                clients.remove(client)
        time.sleep(1.0 / args.rate)


if __name__ == "__main__":
    main()
