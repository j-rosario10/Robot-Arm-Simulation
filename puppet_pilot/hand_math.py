"""Turn MediaPipe's 21 hand landmarks into the three numbers we need.

MediaPipe gives each landmark as (x, y) between 0 and 1 across the image.
From those we work out:
  - where the palm is            -> moves the arm left/right and up/down
  - how big the palm looks       -> closer to the camera = bigger = reach forward
  - how far apart thumb and index are -> open or close the gripper
"""

import math
from dataclasses import dataclass

# Landmark numbers, from MediaPipe's hand model.
WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, PINKY_MCP = 0, 4, 5, 8, 9, 17


@dataclass
class HandReading:
    x: float       # palm centre, 0 (left of image) .. 1 (right)
    y: float       # palm centre, 0 (top) .. 1 (bottom)
    size: float    # apparent palm size, as a fraction of image height
    pinch: float   # thumb-index gap divided by palm size (small = pinched)


def _dist(a, b, aspect: float) -> float:
    # x is stretched by the aspect ratio so distances are in "image heights"
    # both ways (a 640x480 image is wider than it is tall).
    return math.hypot((a[0] - b[0]) * aspect, a[1] - b[1])


def read_hand(landmarks, aspect: float = 640 / 480) -> HandReading:
    """landmarks: list of 21 (x, y) pairs. aspect: image width / height."""
    wrist, index_mcp, pinky_mcp = landmarks[WRIST], landmarks[INDEX_MCP], landmarks[PINKY_MCP]

    # Palm centre: average of three points that don't move when fingers curl.
    x = (wrist[0] + index_mcp[0] + pinky_mcp[0]) / 3.0
    y = (wrist[1] + index_mcp[1] + pinky_mcp[1]) / 3.0

    # Palm size: the longest of three palm bones. Taking the longest keeps the
    # size steady when the hand tilts and one bone looks shorter.
    size = max(
        _dist(wrist, index_mcp, aspect),
        _dist(wrist, pinky_mcp, aspect),
        _dist(wrist, landmarks[MIDDLE_MCP], aspect),
    )
    size = max(size, 1e-6)  # never divide by zero

    # Dividing by palm size makes the pinch value the same near or far.
    pinch = _dist(landmarks[THUMB_TIP], landmarks[INDEX_TIP], aspect) / size
    return HandReading(x=x, y=y, size=size, pinch=pinch)
