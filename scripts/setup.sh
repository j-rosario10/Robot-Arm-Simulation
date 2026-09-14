#!/bin/bash
# Fetch the upstream robot description packages and apply our xacro overlay.
#
# These three packages aren't vendored into the repo: ur_description alone is
# ~86 MB of meshes, and they belong to their authors. This clones them at the
# versions the project was built against and drops our modified xacros on top.
#
# Run from WSL, once, after cloning:
#     bash scripts/setup.sh
#
# Then build:
#     cd ros2_ws && colcon build

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$HERE/ros2_ws/src"
mkdir -p "$SRC"
cd "$SRC"

echo "==> Universal Robots description (ros2 branch)"
if [ ! -d ur_description ]; then
  git clone --depth 1 -b ros2 \
    https://github.com/UniversalRobots/Universal_Robots_ROS2_Description.git ur_description
else
  echo "    already present, skipping"
fi

echo "==> Robotiq description (sparse: we only need the description package)"
if [ ! -d ros2_robotiq_gripper ]; then
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/PickNikRobotics/ros2_robotiq_gripper.git
  (cd ros2_robotiq_gripper && git sparse-checkout set robotiq_description)
else
  echo "    already present, skipping"
fi

echo "==> Combined UR + Robotiq package"
if [ ! -d ur_robotiq ]; then
  git clone --depth 1 https://github.com/automaticaddison/ur_robotiq.git
else
  echo "    already present, skipping"
fi

echo "==> Applying our xacro overlay (see robot_model/README.md for what changed)"
# The upstream package targets a UR3e and was written against an older
# ur_description whose macro took different argument names. Our overlay
# retargets it to the UR5e and fixes the drifted paths/arguments.
cp "$HERE/robot_model/ur5e_urdf.xacro" \
   "$HERE/robot_model/ur5e_robotiq_2f_85_urdf.xacro" \
   "$HERE/robot_model/robotiq_2f_85_urdf.xacro" \
   ur_robotiq/ur_robotiq_description/urdf/
rm -f ur_robotiq/ur_robotiq_description/urdf/ur3e_urdf.xacro \
      ur_robotiq/ur_robotiq_description/urdf/ur3e_robotiq_2f_85_urdf.xacro

echo
echo "Done. Next:"
echo "  cd $HERE/ros2_ws && colcon build"
echo "  source install/setup.bash"
echo "  python -m puppet_pilot.tools.make_pybullet_urdf   # from ros2_ws/src/puppet_pilot"
