# Phase 7 — Evaluation and demo deliverables

## Tooling built (all runnable now; numbers await real demonstrations)

| Deliverable | Tool | Output |
|---|---|---|
| Success rate + cycle time, autonomous | `python -m puppet_pilot.learning.evaluate <ckpt> --episodes 20 --out media/eval.json` | JSON report |
| Human-piloted baseline | `data/demos/manifest.json` summary (success rate + durations accumulate as you record) | manifest |
| Success vs #demonstrations plot | `python -m puppet_pilot.learning.curve` | `media/learning_curve.png/.json` |
| Teleop video | `python -m puppet_pilot.tools.record_video demo <episode.npz> --out media/teleop.mp4` | mp4 |
| Autonomous video | `python -m puppet_pilot.tools.record_video policy <ckpt> --seed 1000 --out media/auto.mp4` | mp4 |
| Side-by-side | `python -m puppet_pilot.tools.record_video side-by-side media/teleop.mp4 media/auto.mp4 --out media/side_by_side.mp4` | mp4 |

Fair-comparison notes baked into the tools: evaluation seeds start at 1000,
disjoint from demo-collection seeds, so the policy is never scored on a
layout it trained on; the human baseline comes from the same manifest the
demos were recorded into (no cherry-picking a separate "good run" set); and
partial credit is reported alongside binary success because SPEC says the
latter is too coarse to show early learning progress.

## The suggested end-game sequence (after recording ≥50 demos)

```bash
python -m puppet_pilot.learning.train --out data/policies/bc_v1.pt
python -m puppet_pilot.learning.evaluate data/policies/bc_v1.pt --episodes 20 --out media/eval.json
python -m puppet_pilot.learning.curve --sizes 5 10 20 35 50
python -m puppet_pilot.tools.record_video demo data/demos/<best>.npz --out media/teleop.mp4
python -m puppet_pilot.tools.record_video policy data/policies/bc_v1.pt --out media/auto.mp4
python -m puppet_pilot.tools.record_video side-by-side media/teleop.mp4 media/auto.mp4 --out media/side_by_side.mp4
```

Then paste `media/eval.json` numbers and the manifest summary into the
README's results table.

## Honest limitations (also in the README — do not soften these)

1. **No force feedback.** Vision teleop is position-controlled; the operator
   cannot feel grip force. Grip force is a *property* (a motor force
   ceiling derived from the gripper's drive train), never modulated live.
   On real hardware: current sensing on the gripper motor or a wrist F/T
   sensor, plus haptic feedback to the operator.
2. **v1 "inspection" is a colour cue** standing in for measurement — the
   Phase 8 upgrade replaces it with camera measurement against SPEC's
   tolerance bands.
3. **Fixed top-down tool orientation** — a deliberate retargeting choice
   (docs/phase4-teleop.md); no tilted grasps.
4. **The policy sees ground-truth object poses** from the simulator, not
   estimated ones — sim-to-real would need a perception stack.
5. **Software-rendered GUI** (no GPU): the live view is not smooth; the
   physics and the recorded data are unaffected.
