# Phase 0 — Environment

## What was built

Two isolated runtime environments, one per OS, matching the project's fixed architecture:

| Side | Runtime | Owns |
|---|---|---|
| Windows 11 | Python 3.12 venv at `windows_side/.venv` | Webcam, MediaPipe pose/hand tracking, TCP publisher |
| WSL2 (Ubuntu 24.04) | ROS 2 Jazzy + Python venv at `~/venvs/puppet_pilot` | Everything else: simulation, kinematics, ROS 2 nodes, recording, learning |

Versions installed (verified at setup time):

- Windows Python: 3.12.10
- MediaPipe: 1.0.0 (OpenCV 5.0.0, NumPy 2.5.1 as its deps)
- WSL Ubuntu: 24.04.3 LTS, Python 3.12.3
- ROS 2: Jazzy Jalisco (`ros-jazzy-desktop` 0.11.0)
- WSL venv: PyBullet 3.2.7, PyTorch 2.13.0+cpu, NumPy 1.26.4 (1.x here vs 2.x on Windows — the two venvs never share an interpreter, so the mismatch is harmless)

## Why this split (interview material)

**The webcam never crosses into WSL2.** USB passthrough into WSL2 (usbipd) is unreliable — device reattachment on every reboot, kernel module fragility, and camera drivers that simply don't exist in the WSL kernel. Instead of fighting that, perception runs natively on Windows where the camera works out of the box, and publishes *normalised pose values* — not video — over a local TCP socket into WSL2.

This is the shape real robot stacks take anyway: perception and control are separate processes behind a message boundary. Consequences worth being able to articulate:

- **Swappability.** The control side consumes a stream of poses; it does not care that MediaPipe produced them. A depth camera or a different tracker could replace the Windows script without touching a line of the ROS side.
- **Bandwidth discipline.** Sending ~30 Hz of small pose packets over localhost is trivial; sending raw video frames across an OS boundary is not. The boundary forces the cheap representation across the wire.
- **Failure isolation.** If MediaPipe hiccups, the ROS side sees a stale value and can hold pose, rather than a driver crash taking down the control stack.

## Environment decisions and their reasons

- **WSL venv is created with `--system-site-packages`.** ROS 2's Python libraries (`rclpy` etc.) are installed by apt into the system `dist-packages`. A fully isolated venv would hide them, and later phases need `rclpy`, `pybullet`, and `torch` importable in the *same* interpreter. This flag makes the venv see system packages while pip-installed extras stay contained in the venv.
- **PyTorch installed from the CPU-only wheel index.** There is no GPU. The CPU wheel is ~180 MB instead of multi-GB CUDA builds, and avoids CUDA libraries that would never load. This also constrains later policy design: the network must train on CPU in minutes (small MLP first — see Phase 6).
- **MediaPipe only on Windows.** It is the camera consumer; nothing in WSL2 imports it. Keeping it out of the WSL env is a guard against accidentally coupling the two sides.
- **Repo lives on the Windows filesystem (`C:\Robot Arm Project`), accessed from WSL as `/mnt/c/Robot Arm Project`.** Both sides need to read SPEC.md and the code; the Windows filesystem is the only place both can reach naturally. The WSL venv deliberately lives in the WSL home directory instead — Python envs on `/mnt/c` are slow (9p filesystem) and can misbehave.

## Failure modes to know about

- **`/mnt/c` I/O is slow.** Fine for source files; wrong for venvs or heavy build churn. If `colcon build` (Phase 1) is painfully slow on `/mnt/c`, the workspace build can move into the WSL filesystem — source stays put.
- **WSLg is required for PyBullet's GUI window.** If `DISPLAY` is empty inside WSL, GUI apps can't open; `wsl --update` or checking `.wslconfig` for `guiApplications=false` are the fixes. Headless (`p.DIRECT`) still works regardless, so training/eval never depend on the GUI.
- **WSLg renders through Mesa llvmpipe (software rasterizer) on this machine — no GPU passthrough.** The GUI works, but every rendered pixel costs CPU. Consequence: keep the GUI viewport modest, prefer `p.DIRECT` for anything batch (training, evaluation), and hold the line on convex-hull collision meshes in Phase 3 — dense visual meshes would hurt twice here.
- **apt's ROS packages tie to the distro Python.** Jazzy on 24.04 means Python 3.12 everywhere. Wheels for pybullet/torch/mediapipe all exist for 3.12 — a newer Python would have broken that alignment.
- **MediaPipe installed as 1.0.0, where the legacy `mp.solutions.pose` / `mp.solutions.hands` API is deprecated/removed.** Phase 4 must use the Tasks API (`PoseLandmarker`, `HandLandmarker` with downloadable `.task` model bundles), not the old solutions examples that dominate tutorials.

## What I should be able to answer after this phase

- Why is the webcam handled on Windows, and why is that a strength rather than a workaround?
- What actually crosses the TCP socket, and why not video?
- Why `--system-site-packages` on the WSL venv, and what breaks without it?
- Why CPU-only PyTorch, and what does that constrain about the policy network later?
