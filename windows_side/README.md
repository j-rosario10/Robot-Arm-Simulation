# windows_side - the perception half

Owns the webcam. Publishes shoulder-relative wrist + calibrated pinch as
newline-delimited JSON over TCP (port 8877) at SPEC's 30 Hz. Never imports
WSL-side code; never sends video.

```powershell
.venv\Scripts\python download_models.py        # once: MediaPipe .task bundles
.venv\Scripts\python pose_publisher.py --self-test
.venv\Scripts\python pose_publisher.py         # calibrate pinch, then stream
```

Options: `--camera N`, `--arm left|right` (default right), `--no-window`.
If WSL cannot connect, allow Python through the Windows firewall when
prompted (the WSL bridge connects to this machine's NAT address).
