"""Teleop session: the sim node plus episode lifecycle and recording.

This is the process a demonstration-collection session runs. On top of the
plain sim node it adds, per SPEC:
  - episode management: spawn -> record -> detect success / timeout /
    failure -> save -> respawn with the next seed
  - the 30 Hz demonstration log (recording/recorder.py)
  - the "currently targeted part" annotation: the nearest unsorted part to
    the gripper's pad tip (a heuristic, recorded so the learner does not
    have to rediscover intent from scratch)

Terminal keys while running:  n = abort episode and respawn,  q = quit.

Run (WSL, venv + ROS sourced):
    python -m puppet_pilot.sim.session --gui --record
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pybullet as p
import rclpy

from puppet_pilot.recording.recorder import EpisodeRecorder
from puppet_pilot.sim.sim_node import SimNode

_OUT_OF_PLAY_Z = -0.05  # below the table top = fell off the world

# Live view for the web panel: rendered with PyBullet's software renderer
# (no GPU on this machine), so keep it small and infrequent - each frame
# costs the physics thread tens of milliseconds.
_STREAM_SIZE = (480, 360)
_STREAM_EVERY_N_FRAMES = 6            # 30 Hz / 6 = 5 fps
_STREAM_CAMERA = dict(eye=[1.25, -1.05, 0.85], target=[0.38, 0.02, 0.08])


class TeleopSession(SimNode):
    def __init__(self, gui: bool, seed: int, record: bool, source: str = "teleop",
                 demos_dir=None, stream_dir=None):
        super().__init__(gui=gui, seed=seed)
        self.record = record
        self.stream_dir = Path(stream_dir) if stream_dir else None
        self._frame_count = 0
        self._last_outcome = None
        if self.stream_dir:
            self.stream_dir.mkdir(parents=True, exist_ok=True)
        self.recorder = EpisodeRecorder(self.world.spec, directory=demos_dir, source=source)
        self.episode_seed = seed
        self.elapsed = 0.0
        self._pending_abort = False
        self._begin_episode(first=True)

        period = 1.0 / self.world.spec.recording.log_rate
        self.create_timer(period, self._frame)
        self._frame_period = period

        # Only start the keyboard listener when we actually have a
        # terminal. Launched from the web control panel there is no
        # stdin, and iterating a closed stdin would raise in the thread.
        if sys.stdin is not None and sys.stdin.isatty():
            threading.Thread(target=self._keyboard_loop, daemon=True).start()
        self.get_logger().info(
            f"session ready (recording={'ON' if record else 'off'}); "
            f"jaw offset {self.world.jaw_offset:.3f} m - pass --jaw-offset to "
            "teleop_node if it logs a different default. Keys: n=respawn, q=quit"
        )

    # -- episode lifecycle ---------------------------------------------------

    def _begin_episode(self, first: bool = False) -> None:
        if not first:
            self.episode_seed += 1
            self.world.reset_episode(self.episode_seed)
        self.elapsed = 0.0
        if self.record:
            self.recorder.start(
                self.episode_seed,
                [(part.kind, part.is_reject) for part in self.world.parts],
            )
        self.get_logger().info(
            f"episode seed {self.episode_seed}: "
            + ", ".join(("REJECT " if part.is_reject else "") + part.kind
                        for part in self.world.parts)
        )

    def _end_episode(self, outcome: str) -> None:
        sorted_count = self.world.parts_sorted_correctly()
        if self.record:
            path = self.recorder.finish(outcome, sorted_count)
            saved = f" -> {path.name}" if path else ""
        else:
            saved = ""
        self.get_logger().info(
            f"episode over: {outcome}, {sorted_count}/{len(self.world.parts)} sorted{saved}"
        )
        self._last_outcome = f"{outcome} ({sorted_count}/{len(self.world.parts)})"
        self._begin_episode()

    def _outcome(self) -> str | None:
        """SPEC's episode termination conditions, checked every frame."""
        spec = self.world.spec
        done = 0
        for part in self.world.parts:
            position = part.pose()[0]
            if position[2] < _OUT_OF_PLAY_Z:
                return "failure_dropped"
            location = self.world.part_location(part)
            resting = np.linalg.norm(part.velocity()[0]) < spec.success.settle_velocity
            if location in ("good", "reject") and resting:
                wanted = "reject" if part.is_reject else "good"
                if location != wanted:
                    return "failure_wrong_shelf"
                done += 1
        if done == len(self.world.parts):
            return "success"
        if self.elapsed > spec.episode.timeout:
            return "timeout"
        return None

    # -- per-frame work --------------------------------------------------------

    def _frame(self) -> None:
        self.elapsed += self._frame_period
        self._frame_count += 1
        if self.stream_dir and self._frame_count % _STREAM_EVERY_N_FRAMES == 0:
            self._write_stream_frame()

        if self.record and self.recorder.recording:
            poses = np.array([np.concatenate(part.pose()) for part in self.world.parts])
            q = self.world.joint_positions()
            q_cmd = np.array(self.last_arm_command) if self.last_arm_command is not None else q
            self.recorder.add_frame(
                q=q,
                qd=self.world.joint_velocities(),
                q_cmd=q_cmd,
                gripper_cmd=self.last_gripper_command,
                gripper_state=self.world.gripper_opening(),
                object_poses=poses,
                target_index=self.target_part_index(),
            )

        if self._pending_abort:
            self._pending_abort = False
            self._end_episode("aborted")
            return
        outcome = self._outcome()
        if outcome is not None:
            self._end_episode(outcome)

    def _write_stream_frame(self) -> None:
        """Render one frame and publish it (plus a status file) atomically."""
        from PIL import Image

        width, height = _STREAM_SIZE
        view = p.computeViewMatrix(_STREAM_CAMERA["eye"], _STREAM_CAMERA["target"], [0, 0, 1])
        projection = p.computeProjectionMatrixFOV(50, width / height, 0.05, 6)
        _, _, rgb, _, _ = p.getCameraImage(
            width, height, view, projection, renderer=p.ER_TINY_RENDERER,
            lightDirection=[1.0, -1.0, 1.4],
        )
        image = np.reshape(np.array(rgb, dtype=np.uint8), (height, width, 4))[:, :, :3]
        tmp = self.stream_dir / "frame.tmp.jpg"
        Image.fromarray(image).save(tmp, format="JPEG", quality=70)
        os.replace(tmp, self.stream_dir / "frame.jpg")

        status = {
            "seed": self.episode_seed,
            "elapsed": round(self.elapsed, 1),
            "timeout": self.world.spec.episode.timeout,
            "sorted": self.world.parts_sorted_correctly(),
            "parts": len(self.world.parts),
            "recording": bool(self.record),
            "last_outcome": self._last_outcome,
            "gripper_opening_mm": round(self.world.gripper_opening() * 1000, 1),
            "updated": time.time(),
        }
        tmp_status = self.stream_dir / "status.tmp.json"
        tmp_status.write_text(json.dumps(status), encoding="utf-8")
        os.replace(tmp_status, self.stream_dir / "status.json")

    def target_part_index(self) -> int:
        """Nearest unsorted part to the pad tip (the intent heuristic)."""
        tool_position, tool_orientation = self.world.tool_pose()
        rotation = np.array(p.getMatrixFromQuaternion(tool_orientation)).reshape(3, 3)
        pad_tip = np.array(tool_position) + rotation[:, 2] * self.world.jaw_offset

        best_index, best_distance = 0, float("inf")
        for index, part in enumerate(self.world.parts):
            location = self.world.part_location(part)
            wanted = "reject" if part.is_reject else "good"
            if location == wanted:
                continue  # already sorted - no longer anyone's target
            distance = float(np.linalg.norm(np.array(part.pose()[0]) - pad_tip))
            if distance < best_distance:
                best_index, best_distance = index, distance
        return best_index

    # -- console ---------------------------------------------------------------

    def _keyboard_loop(self) -> None:
        for line in sys.stdin:
            key = line.strip().lower()
            if key == "n":
                self._pending_abort = True
            elif key == "q":
                rclpy.try_shutdown()
                return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--demos-dir", type=Path, default=None,
                        help="override data/demos (used by smoke tests)")
    parser.add_argument("--stream", type=Path, default=None,
                        help="directory to write frame.jpg + status.json for the web panel")
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = TeleopSession(gui=args.gui, seed=args.seed, record=args.record,
                         demos_dir=args.demos_dir, stream_dir=args.stream)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.recorder.recording:
            node.recorder.finish("aborted", node.world.parts_sorted_correctly())
        node.world.disconnect()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
