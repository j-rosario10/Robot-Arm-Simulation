# Phase 6 — Imitation learning (behaviour cloning)

## The setup

Supervised learning on (observation, action) pairs sliced from the recorded
demonstrations, exactly as SPEC defines them:

- **obs (20)** = joint positions (6) + joint velocities (6) + gripper state
  (1, opening normalised by stroke) + target object pose (7). The target is
  the recorded `target_index` part — v1's "detection" is the colour-cue
  world's ground truth pose, per SPEC.
- **action (7)** = commanded-minus-actual joint deltas (6) + gripper command
  (1, normalised). At evaluation the policy's delta is added to the current
  joints and passed through the *same* safety layer teleop used: joint
  limits, then the capsule self-collision veto. Autonomy doesn't skip guards.

Pipeline: `learning/dataset.py` → `model.py` → `train.py` → `evaluate.py`
(→ `curve.py` for the success-vs-demos plot).

## Decisions to defend

- **MLP (20→128→128→7), not an LSTM.** The observation is close to
  Markovian for this task: velocities carry the short-term temporal
  information, gripper state carries "am I holding something", and the
  target pose carries the goal. Recurrence buys robustness to *partial
  observability* — which this observation mostly doesn't have — at the cost
  of CPU training time and fragility on a ~50-episode dataset. The evidence
  that would reverse this call: evaluation failures that look like forgotten
  context (freezing mid-transfer with a part in hand).
- **Small on purpose**: CPU-only minutes-scale training is a project
  constraint, and with ≤50 episodes a bigger net memorises.
- **Split by episode, never by frame** (SPEC is emphatic, and right):
  consecutive frames are nearly identical, so a frame split leaks the
  validation set into training and reports fantasy numbers.
- **Successful episodes only** by default — a failed demo supervises the
  policy toward the failure. `--include-all` exists for experiments.
- **Normalisation stats from the training split only**, stored inside the
  checkpoint as module buffers: a checkpoint is self-contained (raw obs in,
  raw action out) and can never be paired with the wrong stats file.

## Distribution shift — the failure mode this phase is *about*

Behaviour cloning learns π(obs) on the states the demonstrator visited.
The moment the policy makes a small error, it lands in a state slightly off
the demonstration manifold; its prediction there is worse, the next state is
further off, and errors **compound** — Ross & Bagnell's classic quadratic-
in-horizon result versus the linear error of ordinary supervised learning.

**What it will look like here** (expect to see these, and say so out loud):
- The arm drifts a few cm from any demonstrated approach path, and instead
  of correcting, wanders or parks — the demos contain almost no "recover
  from being wrong" data, because the human was rarely wrong.
- A missed grasp leaves the gripper closed on air near the part — a state
  that essentially never appears in successful demos — and the policy has
  no idea what to do with it.
- Long stretches of near-zero action (the mean of disagreeing demos).

**Mitigations, in the order worth trying:**
1. **More and more-varied demonstrations** — the first hypothesis for a bad
   policy is "not enough data", per the brief. Vary approach directions,
   speeds, part layouts (seeds do this automatically).
2. **Deliberately imperfect demonstrations**: make small mistakes and
   correct them while recording. Recovery behaviour can only be learned if
   it is demonstrated.
3. **Noise injection during collection** (DART-style): perturb the executed
   command slightly while logging the *clean* command as the label, so the
   dataset covers a tube around the ideal trajectory, not a filament.
4. **DAgger** — run the policy, let the expert relabel visited states, and
   retrain. The principled fix, and the natural v2 here since the teleop rig
   *is* the query-the-expert mechanism; costs interactive expert time.

Being able to diagnose this failure out loud is worth more in an interview
than a policy that happens to work — the brief says so, and it is true.

## Evaluation

`evaluate.py` runs SPEC's episode protocol (fresh seeds the demos never
used) and reports binary success, **partial credit (parts correctly sorted
of 8)** — SPEC prefers it for learning curves because binary success is too
coarse early — and mean cycle time of successful episodes. `curve.py`
retrains on growing demo subsets to produce the success-vs-demonstrations
plot for the README.

## Verified (pipeline, not performance)

- Dataset semantics unit-tested against hand-computed obs/action rows.
- On a synthetic `q_cmd = q + δ` relation the trainer recovers δ to 0.01 rad
  (if it can't fit that, the plumbing is broken — this is a pipeline test,
  not a claim about task learning).
- Full chain smoke: scripted pick → recorded → replayed → trained →
  evaluated autonomously, files and reports intact. Real training awaits
  real human demonstrations, which nothing here substitutes for.
