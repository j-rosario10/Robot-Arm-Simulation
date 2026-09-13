"""Replay a recorded demonstration through the simulator.

Kinematic playback: each logged frame's joint angles and object poses are
*set*, not re-simulated. That is the point - replay verifies the recorded
DATA is clean (SPEC's reason for this tool), independent of whether physics
would reproduce it. If a replay looks wrong, the recording is wrong; if a
replay looks right but training goes badly, the data is fine and the problem
is elsewhere. That separation is worth a lot when debugging Phase 6.

Run (WSL, venv):  python -m puppet_pilot.recording.replay data/demos/<file>.npz --gui
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pybullet as p

from puppet_pilot.recording.recorder import load_episode
from puppet_pilot.sim.world import SortingWorld


def replay(path: Path, gui: bool, speed: float) -> None:
    episode = load_episode(path)
    meta = episode["meta"]
    print(f"episode {path.name}: seed {meta['seed']}, outcome {meta['outcome']}, "
          f"{meta['frames']} frames, {meta['duration']:.1f} s, "
          f"{meta['sorted_count']}/{len(meta['kinds'])} sorted")

    world = SortingWorld.create(gui=gui, seed=meta["seed"])
    if [part.kind for part in world.parts] != meta["kinds"]:
        raise RuntimeError(
            "seed reproduced a different part list than the recording - "
            "the world code has changed since this demo was recorded"
        )

    period = 1.0 / (meta["log_rate"] * speed)
    for i in range(len(episode["t"])):
        for joint_index, angle in zip(world.arm_indices, episode["q"][i]):
            p.resetJointState(world.robot, joint_index, float(angle))
        knuckle = world.gripper.angle_for_opening(float(episode["gripper_state"][i]))
        p.resetJointState(world.robot, world.gripper.driven_index, knuckle)
        for follower, multiplier in world.gripper.mimic_indices.items():
            p.resetJointState(world.robot, follower, multiplier * knuckle)
        for part, pose in zip(world.parts, episode["object_poses"][i]):
            p.resetBasePositionAndOrientation(part.body, pose[:3].tolist(), pose[3:].tolist())
        if gui:
            time.sleep(period)

    print("replay finished cleanly (all frames applied)")
    world.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    args = parser.parse_args()
    replay(args.episode, args.gui, args.speed)


if __name__ == "__main__":
    main()
