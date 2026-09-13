"""Dataset assembly from recorded demonstrations.

Observation (20 dims) and action (7 dims), exactly as SPEC's recording table
defines them:
    obs    = joint positions (6) + joint velocities (6) + gripper state (1,
             jaw opening normalised by stroke) + target object pose (7)
    action = joint position deltas (6, commanded minus actual at that frame)
             + gripper command (1, normalised opening)

Two decisions worth defending:
  - Split by EPISODE, never by frame (SPEC is emphatic): frames within an
    episode are highly correlated, so a frame-level split leaks training
    data into validation and produces meaninglessly good numbers.
  - Train on successful episodes only, by default: failed demonstrations
    supervise the policy toward the mistake. (--include-all exists for
    experiments.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from puppet_pilot.recording.recorder import load_episode, load_manifest
from puppet_pilot.spec import SortingSpec, load_spec

OBS_DIM = 20
ACT_DIM = 7


@dataclass
class Arrays:
    """One split's tensors plus the episode files they came from."""

    obs: np.ndarray
    act: np.ndarray
    episodes: list


def episode_to_transitions(episode: dict, spec: SortingSpec) -> tuple[np.ndarray, np.ndarray]:
    """Flatten one recorded episode into (obs, act) rows."""
    stroke = spec.gripper.stroke
    frames = len(episode["t"])
    target = episode["object_poses"][np.arange(frames), episode["target_index"]]

    obs = np.concatenate(
        [
            episode["q"],
            episode["qd"],
            (episode["gripper_state"] / stroke)[:, None],
            target,
        ],
        axis=1,
    ).astype(np.float32)
    act = np.concatenate(
        [
            episode["q_cmd"] - episode["q"],
            (episode["gripper_cmd"] / stroke)[:, None],
        ],
        axis=1,
    ).astype(np.float32)
    return obs, act


def load_dataset(
    directory: Path,
    spec: SortingSpec | None = None,
    include_all: bool = False,
    max_episodes: int | None = None,
    split_seed: int = 0,
) -> tuple[Arrays, Arrays]:
    """Load, filter, and split the demonstrations by episode (SPEC's 80/20)."""
    spec = spec or load_spec()
    manifest = load_manifest(directory)
    entries = manifest["episodes"]
    if not include_all:
        entries = [e for e in entries if e["outcome"] == "success"]
    if not entries:
        kind = "usable" if include_all else "successful"
        raise RuntimeError(
            f"no {kind} episodes in {directory} - record demonstrations "
            "first (python -m puppet_pilot.sim.session --record)"
        )

    rng = np.random.default_rng(split_seed)
    order = rng.permutation(len(entries))
    if max_episodes is not None:
        order = order[:max_episodes]
    picked = [entries[i] for i in order]

    split_at = max(1, int(round(spec.recording.train_fraction * len(picked))))
    if len(picked) > 1:
        split_at = min(split_at, len(picked) - 1)  # always keep >=1 validation episode

    def build(chosen: list) -> Arrays:
        observations, actions = [], []
        for entry in chosen:
            episode = load_episode(directory / entry["file"])
            obs, act = episode_to_transitions(episode, spec)
            observations.append(obs)
            actions.append(act)
        return Arrays(
            obs=np.concatenate(observations) if observations else np.zeros((0, OBS_DIM), np.float32),
            act=np.concatenate(actions) if actions else np.zeros((0, ACT_DIM), np.float32),
            episodes=[entry["file"] for entry in chosen],
        )

    return build(picked[:split_at]), build(picked[split_at:])


def normalisation_stats(train: Arrays) -> dict:
    """Per-dimension standardisation, computed on the training split ONLY
    (computing on everything would leak validation statistics)."""
    return {
        "obs_mean": train.obs.mean(axis=0),
        "obs_std": train.obs.std(axis=0) + 1e-6,
        "act_mean": train.act.mean(axis=0),
        "act_std": train.act.std(axis=0) + 1e-6,
    }
