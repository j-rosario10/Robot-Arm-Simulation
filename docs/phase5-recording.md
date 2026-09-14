# Phase 5: Demonstration recording

## What was built

| Piece | Role |
|---|---|
| `recording/recorder.py` | 30 Hz episode logger → one `.npz` per episode + `data/demos/manifest.json` |
| `recording/replay.py` | plays a recorded episode back through the simulator |
| `sim/session.py` | the demo-collection process: sim node + episode lifecycle + recorder |

## What one frame contains (SPEC's recording table, plus the action label)

`q`, `qd` (actual joints), **`q_cmd` (commanded joints, this is the
behaviour-cloning supervision, so it is recorded explicitly rather than
reconstructed later)**, gripper command and measured opening, all eight
object poses, and `target_index`, the nearest unsorted part to the pad tip,
a recorded statement of operator intent so the learner doesn't have to
rediscover it. Episode level: the RNG seed (SPEC: logged with every episode, it reproduces the exact part layout), part kinds/reject flags, outcome,
partial credit, duration.

## Episode lifecycle (in `session.py`)

spawn → record → automatic termination per SPEC (all 8 sorted = success;
180 s = timeout; part dropped off-world or resting on the wrong shelf =
failure) → save → respawn with the next seed. `n` aborts an episode (saved
as `aborted`, excluded from training by default), `q` quits.

## Why replay is kinematic, not physical

Replay *sets* each logged frame's joint angles and object poses rather than
re-simulating. Deliberate: replay's job (per the brief) is verifying the
**data** is clean, independent of whether physics would reproduce it. If a
replay looks wrong, the recording is wrong; if replays look right but
training goes badly, the data is exonerated and the bug hunt moves on. That
separation is exactly what you want when Phase 6 misbehaves.

## Failure modes

- Aborted/failed episodes still get saved (with their outcome), the
  manifest's success rate is the honest teleop success rate, and the
  dataset loader filters on outcome rather than the recorder hiding data.
- If the world code changes (part generation, seeds), old demos' seeds no
  longer reproduce their layouts, replay detects this and refuses, rather
  than silently replaying against a wrong scene.
