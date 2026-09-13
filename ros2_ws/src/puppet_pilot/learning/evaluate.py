"""Evaluation harness: run the policy autonomously, score it per SPEC.

The policy gets exactly the observation it was trained on, built from sim
state the same way the recorder built it (same target-part heuristic). Its
action passes through the same safety layer as a human command: joint
limits, then the capsule self-collision veto. Autonomy does not get to skip
the guards teleop had.

Reported per run: SPEC success rate, partial credit (parts correctly sorted
out of 8 - SPEC prefers this for learning curves; binary success is too
coarse early on), mean cycle time of successful episodes, and per-episode
outcomes.

Run:  python -m puppet_pilot.learning.evaluate data/policies/bc_v1.pt --episodes 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from puppet_pilot.kinematics.collision import CollisionModelConfig, SelfCollisionChecker
from puppet_pilot.kinematics.params import LinkGeometry
from puppet_pilot.learning.model import load_policy
from puppet_pilot.sim.world import SortingWorld

_OUT_OF_PLAY_Z = -0.05


def target_pose(world: SortingWorld) -> np.ndarray:
    """Same intent heuristic as the recorder: nearest unsorted part to the
    pad tip. Falls back to the last part when everything is sorted."""
    import pybullet as p

    tool_position, tool_orientation = world.tool_pose()
    rotation = np.array(p.getMatrixFromQuaternion(tool_orientation)).reshape(3, 3)
    pad_tip = np.array(tool_position) + rotation[:, 2] * world.jaw_offset

    best, best_distance = None, float("inf")
    for part in world.parts:
        wanted = "reject" if part.is_reject else "good"
        if world.part_location(part) == wanted:
            continue
        position, orientation = part.pose()
        distance = float(np.linalg.norm(np.array(position) - pad_tip))
        if distance < best_distance:
            best, best_distance = np.concatenate([position, orientation]), distance
    if best is None:
        position, orientation = world.parts[-1].pose()
        best = np.concatenate([position, orientation])
    return best


def run_episode(world: SortingWorld, policy, checker, seed: int) -> dict:
    spec = world.spec
    world.reset_episode(seed)
    control_period = 1.0 / spec.recording.log_rate
    physics_per_control = int(round(control_period / world.timestep))
    stroke = spec.gripper.stroke

    q_command = world.home_configuration.copy()
    elapsed, outcome = 0.0, None
    while outcome is None:
        obs = np.concatenate([
            world.joint_positions(),
            world.joint_velocities(),
            [world.gripper_opening() / stroke],
            target_pose(world),
        ]).astype(np.float32)

        action = policy.act(obs).numpy()
        candidate = world.limits.clip(world.joint_positions() + action[:6])
        if not checker.check(candidate, margin=0.005).colliding:
            q_command = candidate
        world.command_joints(q_command)
        world.command_gripper(float(np.clip(action[6], 0.0, 1.0)) * stroke)
        world.step(physics_per_control)
        elapsed += control_period

        outcome = _outcome(world, elapsed)

    return {
        "seed": seed,
        "outcome": outcome,
        "sorted_count": world.parts_sorted_correctly(),
        "duration": round(elapsed, 1),
    }


def _outcome(world: SortingWorld, elapsed: float) -> str | None:
    spec = world.spec
    done = 0
    for part in world.parts:
        if part.pose()[0][2] < _OUT_OF_PLAY_Z:
            return "failure_dropped"
        location = world.part_location(part)
        resting = np.linalg.norm(part.velocity()[0]) < spec.success.settle_velocity
        if location in ("good", "reject") and resting:
            wanted = "reject" if part.is_reject else "good"
            if location != wanted:
                return "failure_wrong_shelf"
            done += 1
    if done == len(world.parts):
        return "success"
    if elapsed > spec.episode.timeout:
        return "timeout"
    return None


def evaluate(checkpoint: Path, episodes: int, base_seed: int, gui: bool = False,
             out: Path | None = None, quiet: bool = False) -> dict:
    policy, info = load_policy(checkpoint)
    world = SortingWorld.create(gui=gui, seed=base_seed)
    checker = SelfCollisionChecker(
        world.dh, LinkGeometry.from_yaml(), CollisionModelConfig.from_yaml()
    )

    results = []
    started = time.monotonic()
    for i in range(episodes):
        result = run_episode(world, policy, checker, base_seed + 1000 + i)
        results.append(result)
        if not quiet:
            print(f"  episode {i + 1}/{episodes} seed {result['seed']}: "
                  f"{result['outcome']}, {result['sorted_count']}/8, {result['duration']}s")
    world.disconnect()

    successes = [r for r in results if r["outcome"] == "success"]
    report = {
        "checkpoint": str(checkpoint),
        "train_episodes": len(info.get("train_episodes", [])),
        "episodes": episodes,
        "success_rate": round(len(successes) / episodes, 3),
        "mean_sorted": round(float(np.mean([r["sorted_count"] for r in results])), 2),
        "mean_cycle_time": round(float(np.mean([r["duration"] for r in successes])), 1)
        if successes else None,
        "wall_seconds": round(time.monotonic() - started, 1),
        "results": results,
    }
    if not quiet:
        print(f"success {report['success_rate'] * 100:.0f}%  "
              f"partial credit {report['mean_sorted']}/8  "
              f"cycle {report['mean_cycle_time']} s")
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.episodes, args.seed, args.gui, args.out)


if __name__ == "__main__":
    main()
