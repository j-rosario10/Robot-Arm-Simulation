"""Episode recording: one .npz per episode plus a manifest.

What gets logged per frame, at SPEC's 30 Hz (all in SI):
    q (6), qd (6)          actual joint positions / velocities
    q_cmd (6)              commanded joint positions (the teleop's output -
                           the action supervision for behaviour cloning)
    gripper_cmd, gripper_state   jaw opening command / measurement [m]
    object_poses (N, 7)    position + quaternion of every part
    target_index           which part the operator is currently working on
                           (nearest unsorted part to the pad tip)

Episode-level: the RNG seed (SPEC requires it logged - it makes the exact
part layout reproducible), part kinds and reject flags, the outcome, partial
credit, and duration. The manifest aggregates all episodes so Phase 6 can
load a dataset without touching every file.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from puppet_pilot.spec import SortingSpec, repo_root


def demos_dir() -> Path:
    return repo_root() / "data" / "demos"


class EpisodeRecorder:
    """Accumulates frames for one episode at a time and writes them out."""

    def __init__(self, spec: SortingSpec, directory: Path | None = None, source: str = "teleop"):
        self.spec = spec
        self.directory = directory or demos_dir()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.source = source
        self._frames: list = []
        self._meta: dict = {}
        self._started_at = 0.0

    @property
    def recording(self) -> bool:
        return bool(self._meta)

    def start(self, seed: int, parts: list) -> None:
        """``parts`` is [(kind, is_reject), ...] in spawn order."""
        self._frames = []
        self._started_at = time.monotonic()
        self._meta = {
            "seed": int(seed),
            "kinds": [kind for kind, _ in parts],
            "is_reject": [bool(flag) for _, flag in parts],
            "log_rate": self.spec.recording.log_rate,
            "source": self.source,
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def add_frame(self, q, qd, q_cmd, gripper_cmd, gripper_state, object_poses, target_index) -> None:
        self._frames.append((
            time.monotonic() - self._started_at,
            np.asarray(q, dtype=np.float32),
            np.asarray(qd, dtype=np.float32),
            np.asarray(q_cmd, dtype=np.float32),
            float(gripper_cmd),
            float(gripper_state),
            np.asarray(object_poses, dtype=np.float32),
            int(target_index),
        ))

    def finish(self, outcome: str, sorted_count: int) -> Path | None:
        """Write the episode; returns the file path (None if nothing recorded)."""
        if not self._meta or not self._frames:
            self._meta = {}
            return None
        meta = dict(self._meta)
        meta.update(
            outcome=outcome,
            sorted_count=int(sorted_count),
            duration=self._frames[-1][0],
            frames=len(self._frames),
        )

        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = self.directory / f"episode_{stamp}_{meta['seed']}.npz"
        columns = list(zip(*self._frames))
        np.savez_compressed(
            path,
            t=np.array(columns[0], dtype=np.float32),
            q=np.stack(columns[1]),
            qd=np.stack(columns[2]),
            q_cmd=np.stack(columns[3]),
            gripper_cmd=np.array(columns[4], dtype=np.float32),
            gripper_state=np.array(columns[5], dtype=np.float32),
            object_poses=np.stack(columns[6]),
            target_index=np.array(columns[7], dtype=np.int32),
            meta=json.dumps(meta),
        )
        self._update_manifest(path.name, meta)
        self._meta = {}
        self._frames = []
        return path

    def _update_manifest(self, filename: str, meta: dict) -> None:
        manifest = load_manifest(self.directory)
        manifest["episodes"] = [e for e in manifest["episodes"] if e["file"] != filename]
        manifest["episodes"].append({
            "file": filename,
            "seed": meta["seed"],
            "outcome": meta["outcome"],
            "sorted_count": meta["sorted_count"],
            "duration": round(meta["duration"], 2),
            "frames": meta["frames"],
            "source": meta["source"],
        })
        episodes = manifest["episodes"]
        successes = [e for e in episodes if e["outcome"] == "success"]
        manifest["summary"] = {
            "episodes": len(episodes),
            "successes": len(successes),
            "success_rate": round(len(successes) / len(episodes), 3),
            "mean_sorted": round(float(np.mean([e["sorted_count"] for e in episodes])), 2),
        }
        (self.directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )


def load_manifest(directory: Path | None = None) -> dict:
    directory = directory or demos_dir()
    path = directory / "manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"episodes": [], "summary": {}}


def load_episode(path: Path) -> dict:
    """Load one episode back as {arrays..., meta: dict}."""
    with np.load(path, allow_pickle=False) as data:
        episode = {key: data[key] for key in data.files if key != "meta"}
        episode["meta"] = json.loads(str(data["meta"]))
    return episode
