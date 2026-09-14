# Phase 3: Simulation world

## What was built

| Module | Role |
|---|---|
| `puppet_pilot/spec.py` | Parses SPEC.md into typed values. **Every** task number in the project comes through here. |
| `sim/parts.py` | Procedural part geometry, analytic mass and inertia |
| `sim/gripper.py` | 2F-85 drive model: opening ↔ knuckle angle, mimic handling, force ceiling |
| `sim/world.py` | Scene assembly, episode randomisation, physics stepping, task evaluation |
| `sim/sim_node.py` | ROS 2 node: publishes joint state, subscribes to joint commands |
| `tools/make_pybullet_urdf.py` | Rewrites the xacro output's mesh URIs to absolute paths for PyBullet |

49 tests pass, covering SPEC parsing, part inertia, scene composition, settling, robot control, gripper behaviour, and a full pick-and-place.

## SPEC.md as the single source of truth

`spec.py` is deliberately **strict**: a value it cannot find raises `SpecError` naming the section and the pattern it was looking for. There are no fallback defaults anywhere. `test_spec.py` restates every parsed value independently, so editing SPEC.md fails the tests and forces a conscious decision rather than silently changing the task.

Millimetres in the document, SI everywhere in the code, the conversion happens once, at the parser boundary.

## Part geometry: what is a primitive and what is not

- **Bushing, ring**, collision is a solid cylinder primitive. Nothing in this task passes through the bore, so modelling the hole in the *collision* shape buys nothing and costs contact-solver time. The bore appears in the visual mesh only (a trimesh annulus).
- **Dowel**, cylinder primitive.
- **Hex bolt**, the only part needing real geometry, because the jaws grip across the head flats. Built as a **compound**: a hex-prism convex hull (written to a small OBJ at build time) plus a cylinder shank. A single convex hull of the whole bolt would fill the neck between head and shank with phantom material and make the part rest on a slope.

Everything is convex: primitives by definition, the bolt head by construction. PyBullet also treats a mesh collision shape on a dynamic body as its convex hull, so the arm's decimated collision STLs are convex hulls for free, the dense visual `.dae` meshes never touch the contact solver.

## Inertia: computed, never defaulted

Every part's inertia tensor is computed analytically from nominal geometry and SPEC's steel density, then written in explicitly. PyBullet's defaults would be wrong twice over: derived from the *collision* shape (solid where a bushing is hollow) and at the wrong density.

The closed forms are validated in `test_parts.py` against trimesh's exact polyhedral mass-property integration, an entirely independent computation path. The hex prism's is the one worth checking:

- Decompose the hexagon into six equilateral triangles of side `s = f/√3`
- Summing triangle polar moments with the parallel-axis theorem: `J = (5√3/8)s⁴`, area `A = (3√3/2)s²`
- So `Izz = m·J/A = (5/36)·m·f²`, and `Ixx = Iyy = m·((5/72)f² + h²/12)`

That `5/36` coefficient is easy to get wrong, and wrong inertia presents as parts that jitter, creep or refuse to settle, a physics bug that is miserable to trace back to its cause.

Resulting masses: bolt 19.8 g, bushing 49.3 g, ring 34.0 g, dowel 27.7 g. All comfortably inside the 2F-85's 5 kg payload and inside its 85 mm stroke (both asserted in tests).

## Gripper drive

**Mimic joints.** The URDF has one driven knuckle and five `<mimic>` joints. PyBullet parses the tag and ignores it. Two ways to restore the coupling:

1. `JOINT_GEAR` constraints, the usual recipe in PyBullet examples. **Tried and rejected**: the finger links weigh ~40 g with inertias around 1e-5 kg·m², so the constraint solver's corrective torques produce astronomical angular accelerations. The instability propagated up the arm and tore the whole robot apart within a second.
2. **Explicit position mirroring**, what is used. Every follower tracks the *measured* driven angle each step, via one `setJointMotorControlArray` inside `world.step()`. No constraint solver involvement, nothing to tune, unconditionally stable.

Tracking the **measured** angle rather than the commanded one matters: when the jaws close on a part the driven knuckle stalls part-way, and the followers stall with it, keeping the pads parallel to the part instead of folding past it.

**Opening ↔ angle.** The task layer speaks in jaw opening (metres, as SPEC does); the simulator needs a knuckle angle. The map is linear between the URDF's own joint limits (0 rad open, 0.8 rad closed) and SPEC's 85 mm stroke. Measured against the real pad geometry, the linear approximation is good to about 6 mm across the stroke (true gap at full open is 79 mm against the nominal 85 mm), the real relation is the four-bar linkage's nonlinear one. Acceptable for this task; worth restating if precision grasping ever matters.

**Grip force.** SPEC gives 60 N at the pads; a revolute joint needs a torque ceiling. The two are related by the linkage's moment arm, taken from the URDF's knuckle-to-finger offset (19.65 mm), giving ≈1.18 N·m. This is the only place grip force enters, and it is a fixed property, never modulated live, because vision teleop has no force feedback.

## Grasp assist (added after the first live teleop session)

Lining a 25 mm part up between the pads by eye, at ~5 fps of software
rendering and with no force or tactile feedback, proved unforgiving. The
simulator now gives an explicit, bounded helping hand (SPEC.md "Grasp
assist"): while the jaws are closing, a part within 40 mm of the jaw centre
is pulled toward it (1.5 N, lightly damped); once inside 12 mm with the jaws
closed on it, it is welded to the tool with a fixed constraint until the jaws
open 15 mm wider than the part. It stands in for the compliant fingers and
haptics a real teleop rig would have. It applies identically to human and
autonomous runs, so the comparison stays fair - and the README lists it as a
limitation, because grasp success with it on overstates what raw physics
would give.

## Self-collision policy

Self-collision is **not** enabled in the simulator. On the 2F-85's faked closed loop it makes the gripper fight itself, and the project already has a purpose-built veto: the Phase 2 capsule checker, which runs on the *command* before it ever reaches the sim (and would work identically against real hardware). Simulator collision detection is for part-to-gripper and part-to-world contact, which is what it is good at.

## The one-world-per-process guard

PyBullet's module-level API implicitly targets client 0. A second `p.connect()` therefore does **not** give you an isolated world, new bodies silently pile into the first one. This presented as an unexplained physics explosion in the test suite, with no error anywhere. `SortingWorld.create` now raises if a client is already connected, and there is a test for that guard.

## Debugging log (the most useful interview material in this phase)

Six real bugs, each with a different signature. Worth being able to tell as stories:

1. **Arm collapses, parts explode.** The URDF's zero configuration has the arm stretched out horizontally. Left there with no position command it sags under gravity into the table and bin. *Fix:* park at SPEC's home pose before anything else moves.
2. **Gear constraints tear the robot apart.** Described above, a general lesson about constraint stiffness on very light links.
3. **Second world corrupts the first.** The PyBullet client bug above.
4. **Dowel pins never settle.** With PyBullet's default rolling friction of zero, a cylinder that starts rolling never stops: 46 to 119 mm of travel in 3 s, still spinning at 36 rad/s, while bushings, rings and bolts held position to under a millimetre. SPEC's friction section covered only lateral friction, so **rolling and spinning friction (0.001, steel on steel) were added to SPEC.md** rather than patched into the code.
5. **Grasp drives the pads through the floor.** I had taken the finger-tip *link origin* as the jaw reference. The pads actually hang from that origin and extend 89 to 152 mm below tool0, a 54 mm error, enough to bury the pads 29 mm under the bin floor. *Fix:* reference the **pad tip**, measured from the collision AABB.
6. **Jaws jam before reaching the part.** The pads swing outward *and downward* as they close, 13.4 mm of extra reach over the closing stroke. Staging a grasp at floor level therefore drives the pads into the floor mid-close. *Fix:* measure the descent at calibration time and expose `world.grasp_height(floor_z)`.

Bugs 5 and 6 are the same underlying lesson: **a gripper's "where it grabs" is not a single point, and it moves during the grasp.** Both values are now measured from the loaded model at startup, so the Phase 8 gripper swap needs no code change here.

## Verification

- **Scene**: 8 parts, 3 rejects, at least one of each kind, all spawned inside the bin, all resting on the floor.
- **Settling**: after settle, one further second of physics moves nothing more than 1 mm.
- **Reproducibility**: same seed → identical layout, logged with the episode.
- **Robot**: tracks joint commands to 0.02 rad; simulated tool pose agrees with the Phase 2 FK to 1e-4 m through the pinned `Rz(π)` correction; holds home against gravity.
- **Gripper**: opens and closes across the full stroke; all five mimic joints track the driven joint to 0.05 rad at both extremes.
- **Grasp**: bushing picked from the bin, lifted 99 mm, jaws settling at 24.5 mm on the Ø25 part, then placed on the good shelf and classified there by `part_location`.

## Failure modes to watch

- **Grasping near a bin wall.** The gripper base is ~91 mm across and the bin walls are only 70 mm tall; a part hard against a wall may not be reachable top-down. Expect this during teleop, and treat it as a task constraint rather than a bug.
- **The linear opening map** drifts by up to ~6 mm from true jaw gap. Fine for grasp/release; do not use it as a measurement device (relevant to the Phase 8 dimensional-inspection upgrade, which should measure from the camera, not the jaws).
- **Rolling friction is a tuning knob.** If dowels feel sticky during teleop, that is the value to revisit, in SPEC.md.
- **`/mnt/c` filesystem speed**, world creation loads meshes across the WSL boundary; it takes a second or two.

## What I should be able to answer

- Why is the bushing's collision shape solid when the real part is hollow, and why is its *inertia* still the hollow one?
- Derive the hex prism's `Izz` and say why analytic inertia beats letting the engine default it.
- Why did gear constraints fail where explicit position mirroring worked? What property of the links made the difference?
- Why do the followers track the measured angle rather than the commanded one?
- How does a 60 N grip force become a joint torque, and what is the approximation in that conversion?
- Why is self-collision off in the simulator when the project has a self-collision checker?
- What is the difference between the jaw offset and the pad descent, and why does a grasp fail if you ignore either?
