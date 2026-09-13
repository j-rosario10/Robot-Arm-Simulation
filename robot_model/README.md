# xacro overlay

`scripts/setup.sh` clones three upstream packages and then copies these three
files over the top of `ur_robotiq/ur_robotiq_description/urdf/`. They're kept
here rather than vendored in place so it's obvious which bits are mine.

The upstream `ur_robotiq` package combines a UR arm with a Robotiq 2F-85. It
ships targeting a UR3e, and it was written against an older `ur_description`,
so a straight retarget wasn't enough, two things had drifted underneath it.

### `ur5e_urdf.xacro`

The upstream wrapper passes `use_fake_hardware` / `fake_sensor_commands` to
the UR macro. The current `ros2` branch renamed those to `use_mock_hardware` /
`mock_sensor_commands`, and passing the old names is a hard xacro error rather
than a warning. So instead of editing the old wrapper I copied the *cloned
branch's own* `ur.urdf.xacro` and changed three things:

- `ur_type` default → `ur5e`
- `tf_prefix` default → `ur5e_`
- a comment saying where it came from

Doing it this way means the wrapper and the macro can't drift apart again.

Worth knowing: the UR macro is fully parameterised by YAML, so retargeting
UR3e → UR5e is a config selection, not a modelling change. The joint limits,
link masses and kinematic offsets all come from Universal Robots' own
published values in `ur_description/config/ur5e/`.

### `robotiq_2f_85_urdf.xacro`

Same problem, different package. Current `robotiq_description` moved the 2F-85
meshes into a `2f_85/` subdirectory when 2F-140 support landed, and renamed
`robotiq_gripper.ros2_control.xacro` to `2f_85.ros2_control.xacro` (same macro
name inside, two changed parameters). Changes here:

- 18 mesh paths → `meshes/{visual,collision}/2f_85/...`
- the ros2_control include path and two of its arguments
- mount parent `ur3e_tool0` → `ur5e_tool0`

### `ur5e_robotiq_2f_85_urdf.xacro`

The top-level file that pulls both halves together. Replaces the upstream
`ur3e_robotiq_2f_85_urdf.xacro`, which `setup.sh` deletes along with
`ur3e_urdf.xacro`.

---

The gripper mounts to the arm with a single fixed joint at `ur5e_tool0`, UR's convention for the mounting flange, so no offsets need measuring. The
2F-85's six moving joints are driven by one input: `robotiq_85_left_knuckle_joint`
is the only independent joint and the other five carry `<mimic>` tags slaved
to it. RViz honours those. PyBullet does not, which is dealt with in
`sim/gripper.py` (and is a good story, see the README).
