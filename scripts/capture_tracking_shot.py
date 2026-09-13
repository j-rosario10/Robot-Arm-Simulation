"""Grab one camera frame with the arm-tracking overlay, cropped below the face.

Waits until MediaPipe has a clean lock on shoulder / elbow / wrist plus the
hand, then saves a frame cropped to start just above the shoulder — so the
tracked arm and the pinch line are visible without the operator's face.

    windows_side\\.venv\\Scripts\\python scripts\\capture_tracking_shot.py

Writes media/tracking.jpg.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "windows_side"))

import pose_publisher as pp  # noqa: E402

# Keep a little headroom above the shoulder so the crop doesn't look severed,
# but stay well below chin height.
_HEADROOM = 0.06  # fraction of frame height


def main() -> None:
    perception = pp.Perception("right")
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        sys.exit("could not open the camera (is the control panel already using it?)")

    print("Get your shoulder, elbow, wrist and hand in frame.")
    print("Waiting for a clean lock...")

    best = None
    deadline = time.monotonic() + 90.0
    while time.monotonic() < deadline:
        ok, frame = capture.read()
        if not ok:
            continue
        result = pp._safe_process(perception, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not (result["pose_ok"] and result["hand_ok"] and result["arm_image"]):
            continue

        # Require the whole chain to actually be inside the frame, otherwise
        # the overlay is drawn against the edge and looks wrong.
        points = result["arm_image"] + (result["hand_image"] or [])
        if not all(0.04 < x < 0.96 and 0.04 < y < 0.96 for x, y in points):
            continue

        shoulder_y = result["arm_image"][0][1]
        wrist_y = result["arm_image"][2][1]
        # Prefer the frame where the arm is most extended down or across: it
        # reads most clearly as "this is the tracked limb".
        score = abs(wrist_y - shoulder_y)
        if best is None or score > best[0]:
            mirrored = cv2.flip(frame, 1)
            pp._draw_overlay(cv2, mirrored, result, 0.5, mirrored=True)
            best = (score, mirrored, shoulder_y)
            print(f"  candidate (arm span {score:.2f})")
        if best is not None and best[0] > 0.28:
            break

    capture.release()
    if best is None:
        sys.exit("never got a clean lock - try more light, or step back a little")

    _, image, shoulder_y = best
    height = image.shape[0]
    top = int(max(0, (shoulder_y - _HEADROOM) * height))
    cropped = image[top:, :]

    out = PROJECT / "media" / "tracking.jpg"
    cv2.imwrite(str(out), cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"saved {out}  (cropped {top}px off the top, kept {cropped.shape[1]}x{cropped.shape[0]})")


if __name__ == "__main__":
    main()
