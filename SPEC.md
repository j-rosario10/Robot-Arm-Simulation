# SPEC: Puppet to Pilot

## What it is

A simulated UR5e robot arm that I control with my hand through a webcam.
I move my hand to move the arm, pinch to grab an object off the table, carry
it over, and open my hand to drop it in a bin.

No machine learning, no training, no data collection.

## What you see

- A MuJoCo window with the arm, a table, a few coloured objects and a bin
- A small webcam preview with my hand landmarks drawn on it, plus the gripper
  state, the bin counter, FPS and latency

One command starts both: `pixi run launch`

## Controls

| Hand | Arm |
|---|---|
| Left / right in the image | Left / right |
| Up / down in the image | Height |
| Closer to / farther from the camera (palm looks bigger / smaller) | Reach forward / back |
| Pinch thumb and index together | Close the gripper |
| Open hand | Open the gripper |
| No hand in view | Arm holds still |

The gripper always points straight down. Only its position is controlled.

Keys (click the webcam preview first): `C` re-centre the hand mapping,
`R` reset the scene, `Q` quit.

## How it is built

Two ROS 2 Jazzy nodes and one launch file, all on one Windows machine.

1. `hand_tracker`: reads the webcam with OpenCV, runs MediaPipe Hands, and
   publishes a target position for the gripper and an open/close command.
2. `arm_sim`: receives those, solves inverse kinematics, runs the MuJoCo
   physics, and handles grasping and the bin.

The real logic lives in plain Python modules in `puppet_pilot/` that do not
import ROS. The nodes are thin wrappers, so the logic can be tested alone
with `pixi run test`.

## Rules the code follows

- **Simulator:** MuJoCo, with the UR5e and Robotiq 2F-85 models from MuJoCo
  Menagerie.
- **Scene:** a table, 4 objects (cubes and cylinders) in different colours,
  and a bin.
- **Grasping is "sticky":** when the gripper closes on an object it is welded
  to the gripper until the gripper opens. It does not rely on friction.
- **Smoothing:** a One Euro filter on the hand position, so the arm is steady
  when my hand is still and quick when it moves.
- **Safety box:** targets are clamped to a reachable region above the table.
- **Performance comes first:**
  - the camera is read in its own thread and only the newest frame is kept
  - webcam at 640x480, MediaPipe's lightest hand model, one hand
  - hand tracking, physics and drawing each run at their own rate, so a slow
    camera frame never freezes the simulation
