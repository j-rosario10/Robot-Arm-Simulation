"""Starts the whole project: both nodes, one command.

    pixi run launch

Closing either window (or pressing Q in the webcam preview) stops both.
"""

import sys
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (ExecuteProcess, RegisterEventHandler, SetEnvironmentVariable,
                            Shutdown)
from launch.event_handlers import OnProcessExit

PROJECT = str(Path(__file__).resolve().parents[1])


def node(module: str) -> ExecuteProcess:
    # Run a node with the same Python that ROS is using, from the project folder.
    return ExecuteProcess(cmd=[sys.executable, "-m", module], cwd=PROJECT, output="screen")


def generate_launch_description() -> LaunchDescription:
    hand_tracker = node("puppet_pilot.nodes.hand_tracker_node")
    arm_sim = node("puppet_pilot.nodes.arm_sim_node")
    return LaunchDescription([
        # Keep all ROS traffic on this computer, and send it over plain local
        # networking instead of shared memory. Shared memory leaves files
        # behind when a program is closed abruptly, and those leftovers slowed
        # messages from 30 a second down to 3.
        SetEnvironmentVariable("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST"),
        SetEnvironmentVariable("FASTRTPS_DEFAULT_PROFILES_FILE",
                               str(Path(__file__).with_name("fastdds_udp_only.xml"))),
        hand_tracker,
        arm_sim,
        # when one node exits, shut the other down too
        RegisterEventHandler(OnProcessExit(target_action=hand_tracker, on_exit=[Shutdown()])),
        RegisterEventHandler(OnProcessExit(target_action=arm_sim, on_exit=[Shutdown()])),
    ])
