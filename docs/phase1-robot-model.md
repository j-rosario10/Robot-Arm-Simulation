# Phase 1: Robot model (UR5e + Robotiq 2F-85)

## What was built

A colcon workspace containing four packages:

| Package | Origin | Role |
|---|---|---|
| `ur_description` | cloned, `UniversalRobots/Universal_Robots_ROS2_Description`, `ros2` branch, shallow | Official UR arm xacro macros, meshes, and per-model parameter YAMLs |
| `robotiq_description` | cloned (sparse) from `PickNikRobotics/ros2_robotiq_gripper` | 2F-85 meshes + ros2_control xacro that the combo package references |
| `ur_robotiq_description` | cloned, `automaticaddison/ur_robotiq`, then modified | The combined arm+gripper model, retargeted ur3e → ur5e |
| `ur_robotiq` | same clone | Empty metapackage, untouched |

Top-level model file: `ur_robotiq_description/urdf/ur5e_robotiq_2f_85_urdf.xacro`.

## How the model composes (worth being able to sketch)

```
ur5e_robotiq_2f_85_urdf.xacro          (robot name, includes both halves)
├── ur5e_urdf.xacro                    (wrapper: declares args, ur_type=ur5e)
│   └── ur_macro.xacro                 (official UR macro from ur_description)
│       └── config/ur5e/*.yaml         (joint limits, DH-ish kinematics, masses, meshes)
└── robotiq_2f_85_urdf.xacro           (gripper links/joints defined inline)
    ├── meshes from robotiq_description        (visual .dae / collision .stl)
    └── 2f_85.ros2_control.xacro from robotiq_description
```

Key mechanism: the official UR macro is *fully parameterised by YAML files*, `ur_type` only selects which config directory (`config/ur5e/`) feeds the macro. That is why retargeting ur3e→ur5e is a config change, not a modelling change. The UR5e's joint limits, link masses, and kinematic offsets all come from Universal Robots' own published values.

The gripper attaches with a single **fixed joint**: `robotiq_85_base_joint`, parent `ur5e_tool0`, origin identity. `tool0` is UR's convention for the mounting flange frame ("all-zeros tool"); the flange geometry is already part of the arm meshes, so no offsets need measuring.

The gripper's 6 moving joints form one closed-loop-ish linkage driven by a single input: `robotiq_85_left_knuckle_joint` is the only independent joint; the other five carry `<mimic>` tags slaved to it (right side mirrored with multiplier −1). This matches SPEC.md's driven-left / mimic-right convention. Note for Phase 3: PyBullet ignores `<mimic>`, the mimic behaviour will need explicit gear constraints there.

## Deviations from the brief, and why

1. **A third repo had to be cloned.** `ur_robotiq_description/package.xml` declares `robotiq_description` as a dependency, the gripper meshes and ros2_control xacro live there, not in ur_robotiq. `automaticaddison/robotiq_description` does not exist on GitHub; the mesh filenames and macro name match PickNik Robotics' `ros2_robotiq_gripper` (the de-facto ROS 2 Robotiq description, which Addison's tutorial series builds against). Sparse-cloned so only the `robotiq_description` package is present, the driver/controller packages in that repo are irrelevant here and would drag in ros2_control build deps.

2. **The UR wrapper was rebuilt against the current macro signature, not just renamed.** The shipped `ur3e_urdf.xacro` was written against the *Iron* branch of ur_description. The current `ros2` branch renamed macro params `use_fake_hardware`→`use_mock_hardware` and `fake_sensor_commands`→`mock_sensor_commands`; passing the old names is a hard xacro error (unknown macro parameter). So `ur5e_urdf.xacro` is a copy of the *cloned branch's own* `ur.urdf.xacro` wrapper with three deltas: `ur_type` default `ur5e`, `tf_prefix` default `ur5e_`, provenance comment. This guarantees wrapper and macro can never drift apart in this workspace.

3. **Gripper xacro needed path adaptation for the same reason (upstream drift).** Current robotiq_description moved 2F-85 meshes into a `2f_85/` subdirectory (when 2F-140 support landed) and renamed the control xacro `robotiq_gripper.ros2_control.xacro` → `2f_85.ros2_control.xacro` (same macro name inside, two changed params). The combo package predates that. Edits: 18 mesh paths, 1 include path, 2 params at the call site, and parent `ur3e_tool0` → `ur5e_tool0`.

## Decisions I should be able to defend

- **Why keep the `ur5e_` tf_prefix instead of dropping it?** It's the repo's existing convention, and it makes frame provenance explicit (`ur5e_tool0` vs a bare `tool0`). Cost: every downstream reference must include the prefix. Consistency with the cloned code won over brevity.
- **Why keep automaticaddison's inline gripper model instead of switching to PickNik's own 2F-85 macro?** The brief chose this repo deliberately; its gripper definition carries explicit per-link inertials and a simple driven/mimic joint layout that maps directly onto SPEC.md's gripper table and onto PyBullet's constraint-based mimic implementation later. Switching to PickNik's macro would change every joint name for zero functional gain here.
- **Why shallow/sparse clones?** These are read-only vendored dependencies; history is dead weight (~100 MB+ for UR meshes). Cloned with WSL git, not Windows git, so line endings stay LF.

## Failure modes / gotchas hit or anticipated

- Old tutorials' xacro args (`use_fake_hardware`) are silently incompatible with the current UR macro, the error surfaces as "unknown macro parameter" at xacro time, not at build time. colcon builds description packages happily even if the xacro inside is broken; only *processing* the xacro catches it.
- `<mimic>` is honoured by robot_state_publisher/RViz (joint_state_publisher_gui shows one gripper slider) but **not** by PyBullet, Phase 3 must recreate it with a gear constraint or explicit position mirroring.
- The `world` root link is massless and intentional: it pins the arm base for RViz (fixed frame `world`) and later gives PyBullet a fixed base.

## Verified

- `colcon build` succeeds on the workspace; `xacro` processes the top-level file to flat URDF; `check_urdf` parses it (link/joint tree well-formed).
- Generated URDF references `meshes/ur5e/...` (proof `ur_type` took effect) and `meshes/visual/2f_85/...` (proof gripper meshes resolve).
- Assembled model rendered in headless PyBullet (internal check): complete arm chain, gripper seated flush at `ur5e_tool0`, fingers close symmetrically when both knuckle joints are set. Two PyBullet quirks surfaced for Phase 3: it resolves neither `package://` nor `file://` URIs (paths must be rewritten to absolute), and pure-frame links without `<inertial>` (`ur5e_base`, `ur5e_flange`, `ur5e_tool0`) get defaulted to mass = 1 kg, harmless for rendering, but Phase 3 should keep them welded by fixed joints so the phantom mass never enters dynamics.
- _RViz acceptance (user-verified): pending._

## How to view it (acceptance check)

```bash
source /opt/ros/jazzy/setup.bash
source "/mnt/c/Robot Arm Project/ros2_ws/install/setup.bash"
ros2 launch urdf_tutorial display.launch.py model:="$(ros2 pkg prefix ur_robotiq_description)/share/ur_robotiq_description/urdf/ur5e_robotiq_2f_85_urdf.xacro"
```

In RViz: set Fixed Frame to `world` if it isn't already. The joint_state_publisher_gui window has sliders for the six `ur5e_*` arm joints and one `robotiq_85_left_knuckle_joint` slider, the mimic joints follow it, closing both fingers symmetrically.
