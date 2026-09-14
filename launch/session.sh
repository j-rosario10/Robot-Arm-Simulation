#!/bin/bash
# Auto-generated launcher - keeps the window open if the program crashes.
source /opt/ros/jazzy/setup.bash
source "/mnt/c/Robot Arm Project/ros2_ws/install/setup.bash"
source "$HOME/venvs/puppet_pilot/bin/activate"
cd "/mnt/c/Robot Arm Project/ros2_ws/src"
python -m puppet_pilot.sim.session --gui --record "$@"
status=$?
if [ $status -ne 0 ]; then
  echo; echo "*** exited with status $status - see messages above ***"
  read -r -p "press Enter to close this window"
fi
