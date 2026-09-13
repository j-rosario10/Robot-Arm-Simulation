# Phase 2: Kinematics core (written *before* the implementation)

Everything here is implemented from scratch in `ros2_ws/src/puppet_pilot/kinematics/`, no KDL, no MoveIt, no IKFast. The point of this phase is that I can derive, defend, and debug every line of the motion math in an interview.

---

## 1. Frames and the DH table

### Convention chosen: standard (distal) Denavit-Hartenberg

Each joint-to-joint transform is `A_i = Rz(θ_i) · Tz(d_i) · Tx(a_i) · Rx(α_i)`.

**Alternatives rejected:**

- **Modified DH (Craig's convention).** Equally expressive, different frame placement (α, a applied before the joint rotation). Rejected because Universal Robots publishes its official tables in *standard* DH, using the same convention makes validation a direct number-for-number comparison instead of a conversion exercise.
- **Product of exponentials (POE / screw theory).** Mathematically cleaner (no intermediate frame gymnastics, just twists in a common frame), and arguably the more modern formulation. Rejected because the brief explicitly asks for DH, DH remains the interview lingua franca, and the validation source (UR's config files) is DH-shaped.

### The UR5e table and where every number comes from

Source of truth: `ur_description/config/ur5e/default_kinematics.yaml`, the same numbers UR publishes as the UR5e DH parameters. The loader reads that YAML at runtime (nothing hardcoded) and maps its per-joint XYZ/RPY offsets onto the table:

| i | joint | a_i [m] | d_i [m] | α_i | from YAML |
|---|---|---|---|---|---|
| 1 | shoulder_pan | 0 | 0.1625 | +π/2 | `shoulder.z` → d1; `upper_arm.roll` → α1 |
| 2 | shoulder_lift | −0.425 | 0 | 0 | `forearm.x` → a2 |
| 3 | elbow | −0.3922 | 0 | 0 | `wrist_1.x` → a3 |
| 4 | wrist_1 | 0 | 0.1333 | +π/2 | `wrist_1.z` → d4; `wrist_2.roll` → α4 |
| 5 | wrist_2 | 0 | 0.0997 | −π/2 | `wrist_2.y` → d5; `wrist_3.{roll,pitch,yaw}` → α5 |
| 6 | wrist_3 | 0 | 0.0996 | 0 | `wrist_3.y` → d6 |

Two things worth being able to explain about this mapping:

- The YAML is in *calibrated-kinematics* format (per-joint full 6-DOF offsets, so a real robot's factory calibration can be dropped in). For the nominal robot most entries are structural zeros; the loader **asserts** those zeros (tolerance 1e-6) so a calibrated file that silently breaks the clean DH structure fails loudly instead of producing a subtly wrong table.
- The negative signs on a2/a3 are UR's convention (elbow "behind" the shoulder at zero pose); they fall straight out of the YAML, and flipping them is the classic way hand-built UR tables go wrong.

The URDF chain carries two extra *fixed* transforms my DH chain does not: `base_link → base_link_inertia` (a π flip about z, UR's URDF matches the teach-pendant "Base" frame, which points opposite to the internal chain) and `wrist_3 → flange → tool0`. The FK validation test measures both corrections once and asserts they are constant over hundreds of random configurations, that proves the DH table and the URDF describe the same robot, with the frame bookkeeping made explicit rather than fudged. (Exact correction values recorded in §6 after the tests ran.)

## 2. Forward kinematics

Plain chaining of the six `A_i` as 4×4 homogeneous matrices in NumPy, returning **all** intermediate frames, not just the tool pose. Reasons: the Jacobian needs every frame origin and z-axis anyway; the collision capsules hang off those same frames; and debugging a wrong FK is only possible when you can print the chain link by link.

## 3. Jacobian and inverse kinematics

### Geometric Jacobian

For revolute joint i: `J_v,i = z_{i−1} × (p_e − p_{i−1})`, `J_w,i = z_{i−1}`, columns assembled from the FK frames. Chosen over numeric differentiation because it is exact, cheaper (one FK pass), and the derivation is the thing an interviewer will actually probe.

### IK: iterative damped least-squares (DLS)

Update rule per iteration: `Δq = Jᵀ (J Jᵀ + λ² I)⁻¹ e`, where `e` is the 6-vector task error: position error stacked with orientation error expressed as the rotation vector (axis·angle) of `R_target · R_currentᵀ`. Step-capped, iterated to tolerance, joints clamped to limits from `joint_limits.yaml`.

**Why DLS and not the plain pseudoinverse** (the question this project must answer): the pseudoinverse solution `Δq = J⁺e` decomposes as `Σ (uᵢᵀe / σᵢ) vᵢ`. Near a singularity, the smallest singular value σ→0 and its 1/σ factor blows up, the arm demands near-infinite joint velocity to make progress in a task direction the mechanism is losing. DLS replaces `1/σ` with `σ/(σ² + λ²)`, which behaves like the pseudoinverse when σ ≫ λ but rolls off to zero as σ → 0 (peak gain 1/2λ). The trade: bounded, stable joint commands *at the cost of tracking error* near singular poses, the arm "goes reluctant" instead of snapping. That is the correct trade for teleoperation, where a wild joint spike is far worse than temporary sluggishness. λ is a tunable; the implementation also supports adaptive damping (λ ramped up only when σ_min drops below a threshold, Chiaverini's scheme) so the arm is undamped-accurate far from singularities.

**Why not the UR5e's analytic (closed-form) IK**, a real alternative, and worth knowing: a 6R arm with a spherical-ish wrist admits up to 8 closed-form solutions; analytic IK is exact and microseconds-fast, and an industrial controller would use it. Rejected here because (a) teleop tracks a *continuously moving* target from the current configuration, an iterative method warm-started at q_now inherently returns the nearest solution and never teleports between the 8 branches, which analytic IK must handle with explicit branch-selection logic; (b) DLS carries over unchanged to any future chain (custom gripper TCP offset, different arm), while the analytic solution is UR-geometry-specific; (c) the differential-kinematics route is the one that demonstrates Jacobian understanding, which is the point of this project.

**Failure modes to be able to narrate:** unreachable target → converges to workspace boundary with nonzero residual (reported, not hidden); joint-limit saturation → clamped, solver may stall against the limit (reported); singular neighbourhood → bounded but slow progress along the lost direction. The wrist singularity (q5 = 0, z4 ∥ z6) is the one the teleop phase will actually hit.

The IK target is the `tool0` flange frame. The gripper TCP offset is deliberately *not* baked in here, it belongs to the task layer (Phase 4), sourced from SPEC.md, so the kinematics core stays gripper-agnostic.

## 4. Trajectories: quintic segments

Per-joint quintic polynomial `q(t) = Σ cᵢ tⁱ` per segment, coefficients solved from six boundary conditions (position, velocity, acceleration at both ends). Default: zero boundary velocity/acceleration ("rest-to-rest"), which for a point-to-point move is exactly the minimum-jerk profile.

**Why quintic:** it is the lowest polynomial order that lets you pin acceleration at both ends, giving C²-continuous trajectories, no acceleration steps to excite the sim's contact solver (or a real arm's structure). **Rejected:** cubic (acceleration jumps at segment ends, visible as jerks), trapezoidal velocity profiles (the industrial standard for time-optimality, but only C¹ and piecewise logic-heavy; nothing in this project is time-critical), full min-jerk optimization through via-points (adds a solver for marginal benefit; rest-to-rest quintics already stop at each waypoint, which suits pick-approach-grasp-retreat structure).

Segment durations are computed from the joint velocity limits in `joint_limits.yaml`: a rest-to-rest quintic's peak velocity is `(15/8)·Δq/T`, so `T ≥ (15/8)·max_i(|Δq_i|)/v_max,i`, scaled by a safety fraction. Acceleration limits are *not* published for the e-series (the YAML says so explicitly), so acceleration is left uncapped by default and accepted as an optional caller-supplied bound, no invented numbers.

## 5. Self-collision checking

Capsule (line-segment + radius) approximation per link, pairwise analytic segment-segment distance, non-adjacent pairs only.

**Where the geometry comes from (no invented numbers):** radii and lengths from the `links:` table in `ur_description/config/ur5e/physical_parameters.yaml` (note: that file also has per-part `*_radius` scalars meant for inertia models; the `links:` table is the more conservative and purpose-built set, documented choice). Capsule endpoints are the FK joint origins, so the capsules follow the arm automatically. The gripper's bounding capsule is *measured from its own collision meshes* by a one-off script (`tools/fit_gripper_capsule.py`, PyBullet AABB query at open pose) and cached to `config/gripper_capsule.yaml` with provenance comments.

**Why capsules:** O(1) exact distance per pair, conservative by construction, standard practice in industrial self-collision guards, and, critically, implementable from scratch in ~40 lines that I can walk through. **Rejected:** full-mesh checks via PyBullet (accurate, but this checker's job is to veto commands *inside the controller* before they reach the sim/hardware, so it must not depend on the sim being up; also GJK-on-meshes is not "from scratch"); sphere trees (strictly more spheres to place and tune for no benefit at this geometry).

The checker exposes a distance margin parameter (default 0 = touching); the teleop layer decides its own safety margin later.

## 6. Validation results

All 22 tests pass (`python -m pytest tests` from `src/puppet_pilot`, < 1 s).

- **yaml→DH**: loader mapping reproduces the raw YAML transform chain to 1e-9 at q=0 and random configs (checked inside the loader itself), and the resulting table equals UR's published UR5e values, restated independently in the test.
- **FK vs the URDF's own FK** (PyBullet as the independent oracle, kinematic-skeleton URDF fixture): agreement to 1e-5 over 100 random configurations.
- **Frame corrections, now pinned down**: `world/base_link → DH base` is exactly a π flip about z (`diag(−1,−1,1)`, zero translation), UR's URDF matches the teach-pendant Base frame, which points opposite the internal chain. `tool0` coincides with DH frame 6 *exactly*: the URDF's `wrist_3→flange` rpy(0,−π/2,−π/2) and `flange→tool0` rpy(π/2,0,π/2) compose to identity. So: `T_world→tool0(q) = Rz(π) · T_dh(q)`, one correction, total.
- **IK round-trip**: 25 random poses, solved from 0.3-rad-perturbed starts: all converge, position residual < 0.1 mm, limits respected. Unreachable target (2 m out) honestly reports `converged=False` with the residual.
- **Singularity behaviour**: at q5 = 1e-6 (wrist singularity) σ_min < 1e-4; for the same task error along the lost direction, the pseudoinverse step is > 50× the DLS step, and the DLS step respects its theoretical bound ‖Δq‖ ≤ ‖e‖/(2λ). IK *to* a near-singular pose stays finite and lands within 1 mm.
- **Trajectories**: boundary conditions exact to 1e-9, v = a = 0 at interior waypoints (C²), sampled peak velocity ≤ the requested fraction of the datasheet limit, duration formula verified.
- **Self-collision**: zero pose and three representative working poses clear; fully folded elbow (q3 = 3.1 rad) flagged. Fitted collision model: upper-arm tube offset +0.135 m (yaml: 0.138), forearm +0.0055 m, gripper capsule z ∈ [−0.006, 0.152] m, radius 0.089 m.

**A finding worth retelling**: the collision-fit cross-check *falsified my initial mental model* of the elbow. I expected the forearm tube to inherit the upper arm's 0.138 m lateral offset (minus 7 mm); the mesh measurement said +0.0055 m. The real layout: the elbow steps back laterally, the forearm runs almost in the DH plane (offset ≈ elbow_offset), and the lateral jog to the wrist happens at d4. The tool refused to write the model until the expectation matched reality, exactly what a provenance check is for.

## 7. What I should be able to answer after this phase

- Derive `A_i` for one DH row on a whiteboard; explain each of the four factors and why the order matters.
- Why is the UR5e's a2 negative? What breaks if you flip it?
- Walk through the geometric Jacobian column formula and say what each term means physically.
- What exactly blows up at a singularity, in SVD terms, and what does DLS change about it? What is the cost?
- Why iterative IK for teleop when an analytic solution exists for this arm?
- Why quintic and not cubic or trapezoidal? What does C² buy?
- Where do the capsule radii come from, and what happens to the checker if the custom gripper replaces the 2F-85?
