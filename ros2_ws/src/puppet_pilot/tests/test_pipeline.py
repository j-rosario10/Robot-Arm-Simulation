"""End-to-end pipeline smoke test: sim -> record -> dataset -> train -> eval.

One scripted pick-and-place is recorded through the real EpisodeRecorder
(exercising the same data path a human demonstration takes), a policy is
trained on it, and the evaluation harness runs it autonomously for a short
budget. Asserts the PIPELINE - files, shapes, reports - not policy quality:
one scripted episode teaches nothing (and per the brief, scripted
trajectories are never used as real training data).
"""

import numpy as np
import pytest

from puppet_pilot.kinematics.collision import CollisionModelConfig, SelfCollisionChecker
from puppet_pilot.kinematics.params import LinkGeometry
from puppet_pilot.learning.evaluate import run_episode
from puppet_pilot.learning.model import load_policy
from puppet_pilot.learning.train import train
from puppet_pilot.recording.recorder import EpisodeRecorder, load_manifest
from puppet_pilot.recording.replay import replay
from puppet_pilot.sim.world import SortingWorld


@pytest.fixture(scope="module")
def world():
    sim = SortingWorld.create(gui=False, seed=7)  # seed 7: bushing spawn verified grasp-clean (open-loop scripts cannot correct a nudged part the way a closed-loop human does)
    yield sim
    import pybullet as p
    if p.isConnected():  # the test disconnects it mid-flow to make room
        sim.disconnect()


def _record_scripted_pick(world, spec, directory) -> None:
    """Drive one bushing to the good shelf, logging frames as the session would."""
    recorder = EpisodeRecorder(spec, directory=directory, source="pipeline-test")
    recorder.start(world.seed, [(p.kind, p.is_reject) for p in world.parts])

    control_period = 1.0 / spec.recording.log_rate
    physics_per_control = int(round(control_period / world.timestep))
    state = {"q_cmd": world.home_configuration.copy(), "grip": spec.gripper.stroke}

    def log_while(seconds):
        for _ in range(int(seconds / control_period)):
            world.command_joints(state["q_cmd"])
            world.command_gripper(state["grip"])
            world.step(physics_per_control)
            recorder.add_frame(
                q=world.joint_positions(), qd=world.joint_velocities(),
                q_cmd=state["q_cmd"], gripper_cmd=state["grip"],
                gripper_state=world.gripper_opening(),
                object_poses=np.array([np.concatenate(p.pose()) for p in world.parts]),
                target_index=0,
            )

    def go(position, seconds=1.5):
        result = world.solve_pose(position, seed=state["q_cmd"])
        assert result.converged
        state["q_cmd"] = result.q
        log_while(seconds)

    bushing = next(p for p in world.parts if p.kind == "bushing")
    start = np.array(bushing.pose()[0])
    grasp = np.array([start[0], start[1], world.grasp_height(spec.scene.bin_thickness)])
    go(grasp + [0, 0, 0.10])
    go(grasp, 1.0)
    state["grip"] = 0.0
    log_while(1.5)
    go(grasp + [0, 0, 0.12], 1.0)
    shelf = spec.scene.good_shelf
    drop = np.array([shelf.center[0], shelf.center[1],
                     shelf.pad_thickness + world.pad_descent + 0.02])
    go(drop + [0, 0, 0.10], 2.0)
    go(drop, 1.0)
    state["grip"] = spec.gripper.stroke
    log_while(1.0)
    go(drop + [0, 0, 0.10], 1.0)

    lifted_ok = world.part_location(bushing) == "good"
    recorder.finish("success" if lifted_ok else "failure", world.parts_sorted_correctly())
    assert lifted_ok, "scripted pick-and-place itself failed - sim regression"


def test_full_pipeline(world, spec, tmp_path):
    # 1. Record through the real recorder.
    _record_scripted_pick(world, spec, tmp_path)
    manifest = load_manifest(tmp_path)
    assert manifest["summary"]["episodes"] == 1
    assert manifest["summary"]["successes"] == 1

    # 2. Replay verifies the file cleanly (headless, full speed). Replay
    #    builds its own world, so release this one first.
    entry = manifest["episodes"][0]
    world.disconnect()
    replay(tmp_path / entry["file"], gui=False, speed=1000.0)

    # 3. Train on it (a single episode becomes the train split; validation
    #    is empty, so train loss stands in - fine for a plumbing test).
    checkpoint = tmp_path / "pipeline.pt"
    info = train(tmp_path, checkpoint, epochs=8, quiet=True)
    assert checkpoint.exists() and np.isfinite(info["best_val_loss"])

    # 4. Evaluate autonomously for a short budget; assert report integrity.
    policy, _ = load_policy(checkpoint)
    eval_world = SortingWorld.create(gui=False, seed=77)
    # Shrink the timeout so the smoke test stays fast: this tests plumbing,
    # not performance. (spec objects are frozen; build a short-lived copy.)
    import dataclasses
    short_spec = dataclasses.replace(
        eval_world.spec, episode=dataclasses.replace(eval_world.spec.episode, timeout=6.0)
    )
    eval_world.spec = short_spec
    checker = SelfCollisionChecker(
        eval_world.dh, LinkGeometry.from_yaml(), CollisionModelConfig.from_yaml()
    )
    result = run_episode(eval_world, policy, checker, seed=78)
    eval_world.disconnect()

    assert result["outcome"] in ("success", "timeout", "failure_dropped", "failure_wrong_shelf")
    assert 0 <= result["sorted_count"] <= spec.episode.parts_per_episode
    assert np.isfinite(result["duration"])
