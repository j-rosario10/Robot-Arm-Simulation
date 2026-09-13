# Phase 4 — Teleoperation (design written before implementation)

## The pipeline

```
Windows                                   WSL2
┌─────────────────────────────┐   TCP    ┌──────────────┐  ROS topic  ┌───────────────┐  /puppet_pilot/   ┌───────────┐
│ pose_publisher.py           │  JSON    │ bridge_node  │  operator   │ teleop_node   │  joint_command    │ sim node/ │
│ webcam → MediaPipe Tasks    │ ───────► │ socket →     │ ──────────► │ EMA → retarget│ ────────────────► │ session   │
│ pose + hands, calibration,  │  30 Hz   │ ROS republish│             │ → IK → veto   │                   │ (PyBullet)│
│ pinch normalisation         │          └──────────────┘             └───────────────┘                   └───────────┘
└─────────────────────────────┘
```

The webcam never crosses into WSL2 (Phase 0 decision). What crosses the wire
is ~40 bytes of *meaning* per frame — a shoulder-relative wrist vector and a
calibrated pinch value — not video. Swap the Windows script for a depth
camera or a glove and nothing on the control side changes.

## The retargeting problem, and how information is deliberately lost

A human arm has ~7 usable DOF (3 shoulder, 1 elbow, 3 wrist) plus the hand.
The UR5e has 6. Any mapping is lossy; the design decision is *choosing what
to lose*. The options considered:

1. **Joint-space mimicry** — extract human joint angles, map onto robot
   joints. Rejected: the kinematic structures are dissimilar (the UR5e's
   joint axes do not correspond to a human arm's), landmark-derived joint
   angles compound MediaPipe noise through every link, and the operator
   would have to learn the robot's body instead of just doing the task.

2. **Full 6-DOF end-effector tracking** — wrist position *and* orientation
   from vision. Rejected for v1: single-RGB wrist orientation is the
   noisiest thing MediaPipe produces (self-occlusion, depth ambiguity), and
   this task never needs it — every grasp in the sorting task is top-down.

3. **Position-only end-effector retargeting with a fixed tool orientation**
   — **chosen**. The shoulder-relative wrist position (3 DOF) drives the
   tool position; the tool always points straight down; the pinch distance
   (1 DOF) drives the jaws. The human's elbow orbit angle — the true "7th
   DOF" — is discarded entirely: the robot's posture is instead determined
   by warm-started IK, which keeps each solution continuous with the last.

Why 3 is right for this project: the task defines the useful control
dimensions (3 position + 1 grip for top-down pick-and-place), and every
channel we keep is one MediaPipe estimates *well*. Keeping more DOF than the
task needs would spend operator concentration and policy capacity on the
noisiest channels. The loss is not a compromise smuggled in — it is the
design: *retargeting should preserve task-relevant intent, not body pose.*

What this costs, honestly: no tilted grasps (irrelevant here), and the
robot's elbow does its own thing (harmless — self-collision is vetoed).

## The mapping, step by step

1. **Shoulder-relative wrist**: `w = (wrist − shoulder) / arm_length` from
   MediaPipe world landmarks, arm length = |shoulder→elbow| + |elbow→wrist|
   (running median). Normalising by the operator's own arm makes the map
   invariant to distance from the camera and to who is operating.
2. **Axis mapping** (camera frame → robot frame): a sign/permutation matrix,
   a config value, not code. The camera convention (x right, y down, z away)
   and how it "feels" depend on where the camera physically sits — the brief
   is explicit that this needs live iteration; defaults are documented in
   `teleop/retarget.py`.
3. **Workspace map**: an affine map from a configured human reach box (in
   arm-lengths) onto the robot task box, then clamped. The task box is
   *derived from SPEC's scene* — the bounding box of bin + both shelves plus
   a margin, floor to home height — so if SPEC's layout moves, the teleop
   workspace follows.
4. **Smoothing**: exponential moving average with SPEC's α = 0.35 on the
   wrist vector, applied *before* IK so joint commands inherit the smoothing.
5. **IK**: Phase 2 damped least squares, warm-started from the current
   configuration, ≤3 iterations per tick, adaptive damping on. Warm-starting
   is what makes iterative IK *the* teleop method: it returns the branch
   nearest the arm's current state, so the arm never teleports.
6. **Veto**: the Phase 2 capsule self-collision checker runs on each
   candidate command; a colliding command is replaced by the last safe one.
   The sim never sees an unsafe command (and real hardware wouldn't either).
7. **Pinch → jaws**: pinch = thumb-tip↔index-tip distance normalised by hand
   span (wrist↔middle-MCP), calibrated at startup (hold open, hold pinched —
   SPEC requires it; hands differ). SPEC's two thresholds define a linear
   band: fully closed below 0.25, fully open above 0.75, proportional
   between — continuous control with dead zones at both ends, which reads as
   hysteresis-like stability at the extremes without a state machine.

## Latency and rate budget

30 Hz publish (SPEC). The socket is newline-delimited JSON — at 40 bytes per
message this is nothing; the brief's "don't let the bridge become a
bottleneck" concern is about *video*, which never crosses. The teleop node
ticks at the same 30 Hz; the sim consumes the latest command at 240 Hz. If
the operator feed stalls (>0.3 s stale), the teleop node holds pose.

## Live tuning from the control panel (added after the first live session)

Two things turned out to need adjusting *while driving*, not in code:

- **Axis flips** (left/right, forward/back, up/down): which way feels natural
  depends on where the camera sits relative to the operator. The panel writes
  `control_panel/teleop_config.json`; the teleop node re-reads it every
  second and applies a sign per robot axis (a row sign on the axis map).
- **Smoothness slider**: one knob, two mechanisms, anchored on SPEC at the
  midpoint. Sliding toward "steadier" lowers the EMA alpha (heavier filtering
  of MediaPipe jitter) *and* tightens a per-tick joint speed cap in the teleop
  node, so the arm glides between IK solutions instead of snapping. The cap
  also shapes what gets recorded as the action label, which keeps
  demonstrations smooth for learning. Camera-side, the hand detector runs on
  every other frame (~17 -> ~25 fps on CPU); pinch changes slowly, so nothing
  is lost.

The camera view is shown **mirrored** (selfie style) with only the tracked
arm chain (shoulder-elbow-wrist) and the thumb-index pinch line drawn -
MediaPipe sees the whole body, but this is all the controller uses.

## Failure modes to expect live (budgeted, per the brief)

- Axis signs feel wrong → flip entries in `AXIS_MAP` (config, one line each).
- Jitter in depth (camera z) is larger than x/y → the EMA handles most; if
  not, α is in SPEC.md and there is a per-axis scale.
- Wrist near workspace edges saturates (clamped) — by design.
- Wrist singularity: q5 near 0 makes the arm sluggish along one direction
  (DLS damping) — expected physics, not a bug; move through it.
- MediaPipe losing the hand (out of frame) → pinch holds last value; pose
  lost → wrist holds; both flagged in the on-screen overlay.

## What I should be able to answer

- What exactly is lost in 7→6 DOF retargeting here, and why is that the
  right information to lose *for this task*?
- Why warm-started iterative IK instead of analytic IK for teleop?
- Why does normalising by arm length and hand span matter?
- Why smooth before IK rather than after?
- Where would force feedback enter if we had it, and why can't we? (We
  command position; grip force is a fixed ceiling — see README limitations.)
