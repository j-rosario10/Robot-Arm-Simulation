"""The behaviour-cloning policy: a deliberately small MLP.

Why an MLP and not an LSTM (the reasoning the brief asks for): the
observation already contains the temporal information this task needs -
joint *velocities* say where the arm is going, and the target object pose
says what it is going for. Sorting is close to Markovian in that state; the
classic reason to reach for recurrence (partial observability, e.g. "which
part was I carrying?") is largely covered by the gripper-state dimension.
An LSTM would add training fragility and CPU cost for information the state
mostly already carries. If evaluation shows failures that look like
forgotten context (dropping a part and freezing mid-transfer), that is the
evidence that would justify revisiting this.

Why small (2x128): CPU-only training in minutes is a project constraint,
~50 demonstrations is a small dataset, and a bigger network would memorise
it rather than generalise.

Normalisation lives inside the module (buffers, saved with the weights), so
a checkpoint is self-contained: raw observation in, raw action out.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from puppet_pilot.learning.dataset import ACT_DIM, OBS_DIM

HIDDEN = 128


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, ACT_DIM),
        )
        self.register_buffer("obs_mean", torch.zeros(OBS_DIM))
        self.register_buffer("obs_std", torch.ones(OBS_DIM))
        self.register_buffer("act_mean", torch.zeros(ACT_DIM))
        self.register_buffer("act_std", torch.ones(ACT_DIM))

    def set_normalisation(self, stats: dict) -> None:
        self.obs_mean.copy_(torch.as_tensor(stats["obs_mean"], dtype=torch.float32))
        self.obs_std.copy_(torch.as_tensor(stats["obs_std"], dtype=torch.float32))
        self.act_mean.copy_(torch.as_tensor(stats["act_mean"], dtype=torch.float32))
        self.act_std.copy_(torch.as_tensor(stats["act_std"], dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Raw observation -> raw action (normalisation handled internally)."""
        normalised = (obs - self.obs_mean) / self.obs_std
        return self.net(normalised) * self.act_std + self.act_mean

    @torch.no_grad()
    def act(self, obs) -> torch.Tensor:
        return self(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0)


def save_policy(policy: Policy, path: Path, info: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": policy.state_dict(), "info": info}, path)


def load_policy(path: Path) -> tuple[Policy, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    policy = Policy()
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    return policy, checkpoint["info"]
