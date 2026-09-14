#!/bin/bash
# Single parametrised WSL launcher used by the web control panel and by hand.
#   bash run.sh puppet_pilot.sim.session --gui --record
# Exports the WSLg display so the PyBullet window appears even when launched
# non-interactively (wsl.exe -- bash run.sh ... has no login profile).
export DISPLAY="${DISPLAY:-:0}"
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
export LIBGL_ALWAYS_SOFTWARE=1
source /opt/ros/jazzy/setup.bash
source "/mnt/c/Robot Arm Project/ros2_ws/install/setup.bash"
source "$HOME/venvs/puppet_pilot/bin/activate"
cd "/mnt/c/Robot Arm Project/ros2_ws/src"
exec python -u -m "$@"
