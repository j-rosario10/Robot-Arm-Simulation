"""Recorder round trip: frames in, npz + manifest out, arrays intact."""

import numpy as np

from puppet_pilot.recording.recorder import EpisodeRecorder, load_episode, load_manifest


def _record_one(spec, directory, seed, outcome, frames=40):
    recorder = EpisodeRecorder(spec, directory=directory, source="test")
    parts = [("bushing", False), ("dowel", True), ("ring", False)]
    recorder.start(seed=seed, parts=parts)
    rng = np.random.default_rng(seed)
    for i in range(frames):
        recorder.add_frame(
            q=rng.normal(0, 1, 6), qd=rng.normal(0, 0.1, 6),
            q_cmd=rng.normal(0, 1, 6),
            gripper_cmd=0.05, gripper_state=0.04,
            object_poses=rng.normal(0, 0.3, (3, 7)),
            target_index=i % 3,
        )
    return recorder.finish(outcome, sorted_count=2)


def test_round_trip(spec, tmp_path):
    path = _record_one(spec, tmp_path, seed=5, outcome="success")
    episode = load_episode(path)

    assert episode["q"].shape == (40, 6)
    assert episode["q_cmd"].shape == (40, 6)
    assert episode["object_poses"].shape == (40, 3, 7)
    assert episode["meta"]["seed"] == 5
    assert episode["meta"]["outcome"] == "success"
    assert episode["meta"]["kinds"] == ["bushing", "dowel", "ring"]
    assert episode["meta"]["is_reject"] == [False, True, False]
    assert episode["meta"]["log_rate"] == spec.recording.log_rate
    assert np.all(np.diff(episode["t"]) >= 0), "timestamps must be monotonic"


def test_manifest_aggregates(spec, tmp_path):
    _record_one(spec, tmp_path, seed=1, outcome="success")
    _record_one(spec, tmp_path, seed=2, outcome="timeout")
    _record_one(spec, tmp_path, seed=3, outcome="success")

    manifest = load_manifest(tmp_path)
    assert manifest["summary"]["episodes"] == 3
    assert manifest["summary"]["successes"] == 2
    assert manifest["summary"]["success_rate"] == 0.667
    assert manifest["summary"]["mean_sorted"] == 2.0


def test_finish_without_start_is_harmless(spec, tmp_path):
    recorder = EpisodeRecorder(spec, directory=tmp_path)
    assert recorder.finish("aborted", 0) is None
    assert load_manifest(tmp_path)["episodes"] == []
