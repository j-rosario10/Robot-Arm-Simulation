"""Render demonstrations and policy runs to video (Phase 7 deliverables).

Subcommands:
    demo <episode.npz> --out media/teleop.mp4       replay a recorded demo
    policy <checkpoint.pt> --seed N --out media/auto.mp4   autonomous run
    side-by-side <left.mp4> <right.mp4> --out media/compare.mp4

Rendering uses PyBullet's software renderer (no GPU on this machine), so
expect a few frames per second of wall time - fine for offline video export.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pybullet as p

WIDTH, HEIGHT, FPS = 640, 480, 30
_CAMERA = dict(eye=[1.25, -1.05, 0.85], target=[0.38, 0.02, 0.08])


def _writer(path: Path):
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(path, fps=FPS, codec="libx264", quality=7)


def _frame() -> np.ndarray:
    view = p.computeViewMatrix(_CAMERA["eye"], _CAMERA["target"], [0, 0, 1])
    projection = p.computeProjectionMatrixFOV(50, WIDTH / HEIGHT, 0.05, 6)
    _, _, rgb, _, _ = p.getCameraImage(
        WIDTH, HEIGHT, view, projection, renderer=p.ER_TINY_RENDERER,
        lightDirection=[1.0, -1.0, 1.4],
    )
    return np.reshape(np.array(rgb, dtype=np.uint8), (HEIGHT, WIDTH, 4))[:, :, :3]


def render_demo(episode_path: Path, out: Path) -> None:
    from puppet_pilot.recording.recorder import load_episode
    from puppet_pilot.sim.world import SortingWorld

    episode = load_episode(episode_path)
    meta = episode["meta"]
    world = SortingWorld.create(gui=False, seed=meta["seed"])
    writer = _writer(out)
    stride = max(1, int(round(meta["log_rate"] / FPS)))
    for i in range(0, len(episode["t"]), stride):
        for joint_index, angle in zip(world.arm_indices, episode["q"][i]):
            p.resetJointState(world.robot, joint_index, float(angle))
        knuckle = world.gripper.angle_for_opening(float(episode["gripper_state"][i]))
        p.resetJointState(world.robot, world.gripper.driven_index, knuckle)
        for follower, multiplier in world.gripper.mimic_indices.items():
            p.resetJointState(world.robot, follower, multiplier * knuckle)
        for part, pose in zip(world.parts, episode["object_poses"][i]):
            p.resetBasePositionAndOrientation(part.body, pose[:3].tolist(), pose[3:].tolist())
        writer.append_data(_frame())
    writer.close()
    world.disconnect()
    print(f"wrote {out}")


def render_policy(checkpoint: Path, seed: int, out: Path, max_seconds: float | None) -> None:
    from puppet_pilot.kinematics.collision import CollisionModelConfig, SelfCollisionChecker
    from puppet_pilot.kinematics.params import LinkGeometry
    from puppet_pilot.learning.evaluate import _outcome, target_pose
    from puppet_pilot.learning.model import load_policy
    from puppet_pilot.sim.world import SortingWorld

    policy, _ = load_policy(checkpoint)
    world = SortingWorld.create(gui=False, seed=seed)
    checker = SelfCollisionChecker(
        world.dh, LinkGeometry.from_yaml(), CollisionModelConfig.from_yaml()
    )
    spec = world.spec
    limit = max_seconds if max_seconds is not None else spec.episode.timeout

    writer = _writer(out)
    control_period = 1.0 / spec.recording.log_rate
    physics_per_control = int(round(control_period / world.timestep))
    q_command = world.home_configuration.copy()
    elapsed = 0.0
    while elapsed < limit:
        obs = np.concatenate([
            world.joint_positions(), world.joint_velocities(),
            [world.gripper_opening() / spec.gripper.stroke], target_pose(world),
        ]).astype(np.float32)
        action = policy.act(obs).numpy()
        candidate = world.limits.clip(world.joint_positions() + action[:6])
        if not checker.check(candidate, margin=0.005).colliding:
            q_command = candidate
        world.command_joints(q_command)
        world.command_gripper(float(np.clip(action[6], 0.0, 1.0)) * spec.gripper.stroke)
        world.step(physics_per_control)
        writer.append_data(_frame())
        elapsed += control_period
        if _outcome(world, elapsed) is not None:
            break
    writer.close()
    world.disconnect()
    print(f"wrote {out} ({elapsed:.0f} s of episode)")


def side_by_side(left: Path, right: Path, out: Path) -> None:
    import imageio

    left_reader = imageio.get_reader(left)
    right_reader = imageio.get_reader(right)
    writer = _writer(out)
    for frame_left, frame_right in zip(left_reader, right_reader):
        height = min(frame_left.shape[0], frame_right.shape[0])
        writer.append_data(np.hstack([frame_left[:height], frame_right[:height]]))
    writer.close()
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("demo")
    demo.add_argument("episode", type=Path)
    demo.add_argument("--out", type=Path, required=True)

    policy = commands.add_parser("policy")
    policy.add_argument("checkpoint", type=Path)
    policy.add_argument("--seed", type=int, default=1000)
    policy.add_argument("--out", type=Path, required=True)
    policy.add_argument("--max-seconds", type=float, default=None)

    compose = commands.add_parser("side-by-side")
    compose.add_argument("left", type=Path)
    compose.add_argument("right", type=Path)
    compose.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "demo":
        render_demo(args.episode, args.out)
    elif args.command == "policy":
        render_policy(args.checkpoint, args.seed, args.out, args.max_seconds)
    else:
        side_by_side(args.left, args.right, args.out)


if __name__ == "__main__":
    main()
