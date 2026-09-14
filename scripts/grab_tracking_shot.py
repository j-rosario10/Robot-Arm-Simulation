"""Grab the arm-tracking shot from the control panel's live camera feed.

Unlike capture_tracking_shot.py this does NOT open the webcam, so it works
while the control panel is running and holding the camera. It watches the
frame the camera process is already writing, finds the drawn arm nodes by
their colour, and keeps the frame where the arm is most extended.

It then crops just above the shoulder node, so the tracked arm and the pinch
line are visible without the operator's face.

    windows_side\\.venv\\Scripts\\python scripts\\grab_tracking_shot.py

Writes media/tracking.jpg.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
LIVE_FRAME = PROJECT / "control_panel" / "live" / "camera.jpg"
OUT = PROJECT / "media" / "tracking.jpg"

# The overlay draws the arm chain in BGR (0, 220, 120). Match a band around
# it rather than the exact value, since JPEG compression shifts pixels.
NODE_LOW = np.array([0, 170, 70], dtype=np.uint8)
NODE_HIGH = np.array([90, 255, 190], dtype=np.uint8)

HEADROOM = 40      # px kept above the shoulder node, so the crop isn't severe
WATCH_SECONDS = 45


def arm_nodes(image: np.ndarray) -> list:
    """Centres of the drawn joint blobs, top to bottom."""
    mask = cv2.inRange(image, NODE_LOW, NODE_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centres = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 40:                      # text glyphs and line segments
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w > 60 or h > 60:               # the connecting lines, not a node
            continue
        centres.append((x + w // 2, y + h // 2))
    return sorted(centres, key=lambda c: c[1])


def main() -> None:
    if not LIVE_FRAME.exists():
        sys.exit(f"{LIVE_FRAME} not found. Start the camera from the control panel first.")

    print("Hold the pose: shoulder, elbow, wrist and hand all in frame.")
    print(f"Watching the live feed for {WATCH_SECONDS}s, keeping the best frame...")

    best_score = 0.0
    best = None
    last_mtime = 0.0
    deadline = time.monotonic() + WATCH_SECONDS

    while time.monotonic() < deadline:
        try:
            mtime = LIVE_FRAME.stat().st_mtime
            if mtime == last_mtime:
                time.sleep(0.1)
                continue
            last_mtime = mtime
            image = cv2.imread(str(LIVE_FRAME))
        except OSError:
            continue
        if image is None:
            continue

        nodes = arm_nodes(image)
        if len(nodes) < 3:
            continue

        # Most extended arm = biggest spread between the first and last node.
        spread = float(np.hypot(nodes[-1][0] - nodes[0][0], nodes[-1][1] - nodes[0][1]))
        if spread > best_score:
            best_score = spread
            best = (image.copy(), nodes[0][1])
            print(f"  got one (arm spread {spread:.0f}px)")

    if best is None:
        sys.exit("Never saw the full arm. Check 'pose' and 'hand' are both green, then retry.")

    image, shoulder_y = best
    top = max(0, shoulder_y - HEADROOM)
    cropped = image[top:, :]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"\nSaved {OUT}")
    print(f"  cropped {top}px off the top, kept {cropped.shape[1]}x{cropped.shape[0]}")


if __name__ == "__main__":
    main()
