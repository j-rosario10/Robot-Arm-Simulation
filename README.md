# Puppet to Pilot

Control a simulated UR5e robot arm with your hand through a webcam. Move your
hand to move the arm, pinch to grab an object off the table, carry it over,
and open your hand to drop it in the bin.

It runs on one Windows machine with ROS 2 Jazzy and MuJoCo. No machine
learning and no training.

## Install

You need Windows 10 or 11 and a webcam.

1. Install [pixi](https://pixi.sh). It fetches everything else (ROS 2,
   MuJoCo, MediaPipe, OpenCV). In PowerShell:

   ```powershell
   winget install prefix-dev.pixi
   ```

2. Download this repository, open a terminal in its folder, and run:

   ```powershell
   pixi install
   ```

   The first time takes a few minutes.

## Run

Double-click `start.bat`, or run:

```powershell
pixi run launch
```

Two windows open: the simulator with the arm, and a small webcam preview that
stays on top. Hold your hand up in front of the camera. Wherever your hand is
when it is first seen becomes the centre position.

## Controls

| Your hand | The arm |
|---|---|
| Left / right | Left / right |
| Up / down | Up / down |
| Closer to / farther from the camera | Reach forward / back |
| Pinch thumb and index finger | Close the gripper |
| Open hand | Open the gripper |
| Out of view | Arm holds still |

An object lights up when it is close enough to grab. Pinch then, carry it
over the bin, and open your hand.

| Key | What it does | Where |
|---|---|---|
| `C` | Re-centre: your hand's current position becomes the middle | Webcam preview |
| `R` | Reset the scene and the bin counter | Either window |
| `Q` | Quit | Webcam preview |

Click a window first so it receives the key.

## How it works

Two ROS 2 nodes, started by one launch file.

```
  webcam
    |
    v
+----------------+   /hand/target           +----------------+
|  hand_tracker  | -----------------------> |    arm_sim     |
|                |   /hand/gripper_closed   |                |
|  OpenCV camera | -----------------------> |  IK solver     |
|  MediaPipe     |   /sim/reset             |  MuJoCo physics|
|  hand mapping  | -----------------------> |  sticky grasp  |
|                |   /sim/status            |  bin counter   |
|                | <----------------------- |                |
+----------------+                          +----------------+
    |                                           |
    v                                           v
  webcam preview window                     MuJoCo window
```

- `hand_tracker` finds your hand, turns it into a target position for the
  gripper plus an open/close command, and publishes both.
- `arm_sim` works out the joint angles that put the gripper there (inverse
  kinematics), runs the physics, and handles grabbing and the bin.

The nodes are thin. The real logic is in plain Python files in
`puppet_pilot/` that do not need ROS:

| File | Job |
|---|---|
| `camera.py` | Reads the webcam in its own thread, keeps only the newest frame |
| `hand_math.py` | Palm position, palm size and pinch from the hand landmarks |
| `filters.py` | One Euro filter that smooths out hand jitter |
| `mapping.py` | Hand position to arm target, clamped to a safe box above the table |
| `tracker.py` | Ties the four files above together and draws the preview |
| `scene.py` | Builds the MuJoCo scene: arm, gripper, table, objects, bin |
| `ik.py` | Inverse kinematics with the gripper kept pointing down |
| `grasp.py` | Sticky grasp and the bin counter |
| `sim.py` | Ties the scene, IK and grasping together |

## Performance

Measured on my laptop (Intel Core Ultra 7 155H, integrated graphics):

- Hand tracking: 30 fps, which is the webcam's limit
- Camera frame to arm command: 12 to 18 ms
- Physics: 500 steps per second, drawing at 60 fps

Camera, hand tracking, physics and drawing each run at their own pace, so a
slow camera frame never freezes the simulation. Both windows show these
numbers live.

## Other commands

```powershell
pixi run test    # run the tests
pixi run hand    # hand tracking only, without the robot
```

## Credits

The UR5e and Robotiq 2F-85 models come from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) and
keep their own licences (see `models/`). Hand tracking uses Google's
MediaPipe hand landmarker. My own code is under the MIT licence.
