"""Learning stack on synthetic episodes: dataset semantics, training, I/O.

Synthetic data here tests the PIPELINE - shapes, splits, normalisation,
checkpointing - not policy quality. The brief forbids substituting scripted
trajectories for human demonstrations, and nothing here does: these arrays
never claim to be demonstrations of the task.
"""

import numpy as np
import pytest
import torch

from puppet_pilot.learning.dataset import (
    ACT_DIM, OBS_DIM, episode_to_transitions, load_dataset, normalisation_stats,
)
from puppet_pilot.learning.model import Policy, load_policy, save_policy
from puppet_pilot.learning.train import train
from puppet_pilot.recording.recorder import EpisodeRecorder, load_episode


def _synthesise_demos(spec, directory, count=6, frames=30):
    """A learnable synthetic relation: q_cmd = q + fixed delta."""
    delta = np.linspace(-0.02, 0.03, 6)
    for seed in range(count):
        recorder = EpisodeRecorder(spec, directory=directory, source="synthetic-test")
        recorder.start(seed=seed, parts=[("bushing", False), ("ring", True)])
        rng = np.random.default_rng(seed)
        for _ in range(frames):
            q = rng.normal(0, 0.8, 6)
            recorder.add_frame(
                q=q, qd=rng.normal(0, 0.05, 6), q_cmd=q + delta,
                gripper_cmd=0.06, gripper_state=0.055,
                object_poses=rng.normal(0, 0.2, (2, 7)), target_index=0,
            )
        recorder.finish("success", 2)
    return delta


@pytest.fixture()
def demo_dir(spec, tmp_path):
    _synthesise_demos(spec, tmp_path)
    return tmp_path


def test_transition_semantics(spec, demo_dir):
    """obs = [q, qd, grip/stroke, target pose]; act = [q_cmd - q, cmd/stroke]."""
    from puppet_pilot.recording.recorder import load_manifest

    entry = load_manifest(demo_dir)["episodes"][0]
    episode = load_episode(demo_dir / entry["file"])
    obs, act = episode_to_transitions(episode, spec)

    assert obs.shape[1] == OBS_DIM and act.shape[1] == ACT_DIM
    i = 7
    np.testing.assert_allclose(obs[i, :6], episode["q"][i], atol=1e-6)
    np.testing.assert_allclose(obs[i, 12], episode["gripper_state"][i] / spec.gripper.stroke)
    target = episode["object_poses"][i, episode["target_index"][i]]
    np.testing.assert_allclose(obs[i, 13:], target, atol=1e-6)
    np.testing.assert_allclose(act[i, :6], episode["q_cmd"][i] - episode["q"][i], atol=1e-6)


def test_split_is_by_episode(spec, demo_dir):
    train_split, val_split = load_dataset(demo_dir, spec)
    assert len(val_split.episodes) >= 1, "SPEC: always a validation episode"
    assert not set(train_split.episodes) & set(val_split.episodes)
    # 80/20 of 6 episodes -> 5/1 (rounded, min 1 held out)
    assert len(train_split.episodes) == 5


def test_training_learns_the_synthetic_relation(spec, tmp_path):
    """On q_cmd = q + delta the MLP must recover delta almost exactly -
    if it cannot fit THIS, the pipeline is broken somewhere."""
    delta = _synthesise_demos(spec, tmp_path, count=6, frames=60)
    checkpoint = tmp_path / "policy.pt"
    info = train(tmp_path, checkpoint, epochs=40, quiet=True)
    assert info["best_val_loss"] < 1e-3

    policy, _ = load_policy(checkpoint)
    rng = np.random.default_rng(99)
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[:6] = rng.normal(0, 0.8, 6)
    action = policy.act(obs).numpy()
    np.testing.assert_allclose(action[:6], delta, atol=0.01)


def test_checkpoint_round_trip(tmp_path):
    policy = Policy()
    stats = {
        "obs_mean": np.arange(OBS_DIM, dtype=np.float32),
        "obs_std": np.full(OBS_DIM, 2.0, dtype=np.float32),
        "act_mean": np.zeros(ACT_DIM, dtype=np.float32),
        "act_std": np.ones(ACT_DIM, dtype=np.float32),
    }
    policy.set_normalisation(stats)
    save_policy(policy, tmp_path / "p.pt", {"note": "test"})

    loaded, info = load_policy(tmp_path / "p.pt")
    assert info["note"] == "test"
    obs = torch.randn(4, OBS_DIM)
    torch.testing.assert_close(loaded(obs), policy(obs))
