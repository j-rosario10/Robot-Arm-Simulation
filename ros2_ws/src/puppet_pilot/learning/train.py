"""Behaviour-cloning training loop. CPU-only, minutes not hours.

Run (WSL, venv):
    python -m puppet_pilot.learning.train --out data/policies/bc_v1.pt
Options: --demos DIR, --epochs N, --max-episodes N (for the learning curve),
--include-all (also train on failed demos - normally a bad idea).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from puppet_pilot.learning.dataset import load_dataset, normalisation_stats
from puppet_pilot.learning.model import Policy, save_policy
from puppet_pilot.recording.recorder import demos_dir
from puppet_pilot.spec import load_spec

BATCH_SIZE = 256
LEARNING_RATE = 1e-3


def train(
    demos: Path,
    out: Path,
    epochs: int = 60,
    max_episodes: int | None = None,
    include_all: bool = False,
    seed: int = 0,
    quiet: bool = False,
) -> dict:
    torch.manual_seed(seed)
    spec = load_spec()
    train_split, val_split = load_dataset(
        demos, spec, include_all=include_all, max_episodes=max_episodes, split_seed=seed
    )
    stats = normalisation_stats(train_split)

    policy = Policy()
    policy.set_normalisation(stats)
    optimiser = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    obs = torch.as_tensor(train_split.obs)
    act = torch.as_tensor(train_split.act)
    val_obs = torch.as_tensor(val_split.obs)
    val_act = torch.as_tensor(val_split.act)

    if not quiet:
        print(f"train: {len(train_split.episodes)} episodes / {len(obs)} frames | "
              f"val: {len(val_split.episodes)} episodes / {len(val_obs)} frames")

    best_val = float("inf")
    best_state = None
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        policy.train()
        order = torch.randperm(len(obs))
        epoch_loss = 0.0
        for start in range(0, len(obs), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            optimiser.zero_grad()
            loss = loss_fn(policy(obs[batch]), act[batch])
            loss.backward()
            optimiser.step()
            epoch_loss += float(loss.detach()) * len(batch)
        epoch_loss /= len(obs)

        policy.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(policy(val_obs), val_act)) if len(val_obs) else epoch_loss
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in policy.state_dict().items()}
        if not quiet and (epoch % 10 == 0 or epoch == 1):
            print(f"epoch {epoch:3d}  train {epoch_loss:.5f}  val {val_loss:.5f}")

    if best_state is not None:
        policy.load_state_dict(best_state)  # keep the best-validation weights
    info = {
        "train_episodes": train_split.episodes,
        "val_episodes": val_split.episodes,
        "epochs": epochs,
        "best_val_loss": best_val,
        "train_seconds": round(time.monotonic() - started, 1),
        "seed": seed,
    }
    save_policy(policy, out, info)
    if not quiet:
        print(f"saved {out}  (best val {best_val:.5f}, {info['train_seconds']} s on CPU)")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", type=Path, default=demos_dir())
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--include-all", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(args.demos, args.out, args.epochs, args.max_episodes, args.include_all, args.seed)


if __name__ == "__main__":
    main()
