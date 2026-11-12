"""Hand-to-arm mapping: where should the gripper go for this hand position?

Rules (the gripper always points straight down, only its position moves):
  hand left/right in the image  -> arm left/right
  hand up/down in the image     -> arm height
  hand nearer/farther (palm looks bigger/smaller) -> arm reach forward/back
  thumb-index pinch             -> gripper closed/open

Everything is measured relative to a "centre" hand pose. Press the re-centre
key and wherever your hand is becomes the middle of the arm's workspace.
"""

import math
from dataclasses import dataclass, field

from puppet_pilot.filters import VectorFilter
from puppet_pilot.hand_math import HandReading


@dataclass
class MappingConfig:
    # Middle of the arm's workspace, in metres (x forward, y left, z up).
    home: tuple = (0.50, 0.0, 0.25)
    # The arm is never sent outside this box. z_min keeps the fingers off the table.
    x_range: tuple = (0.30, 0.75)
    y_range: tuple = (-0.40, 0.40)
    z_range: tuple = (0.03, 0.50)
    # How far the arm moves for a hand movement.
    gain_side: float = 1.1    # metres per image-width of hand travel
    gain_height: float = 0.9  # metres per image-height of hand travel
    gain_reach: float = 0.45  # metres per doubling of apparent palm size
    # Pinch hysteresis: two thresholds so the gripper doesn't flicker when the
    # pinch value hovers around a single cut-off.
    close_below: float = 0.30
    open_above: float = 0.50
    # One Euro filter settings (see filters.py).
    min_cutoff: float = 1.5
    beta: float = 4.0


def _clamp(v: float, lo_hi: tuple) -> float:
    return max(lo_hi[0], min(lo_hi[1], v))


@dataclass
class HandToArm:
    config: MappingConfig = field(default_factory=MappingConfig)

    def __post_init__(self):
        c = self.config
        self._filter = VectorFilter(3, min_cutoff=c.min_cutoff, beta=c.beta)
        self._centre = None       # (x, y, size) of the hand at re-centre time
        self._recentre = True     # first hand we see becomes the centre
        self.closed = False       # current gripper command
        self.target = c.home      # last target we produced

    def recentre(self) -> None:
        """Make the next hand position the middle of the workspace."""
        self._recentre = True

    def update(self, hand: HandReading | None, t: float):
        """Returns (target_xyz, gripper_closed). No hand -> hold everything."""
        if hand is None:
            return self.target, self.closed   # arm holds its position

        if self._recentre:
            self._centre = (hand.x, hand.y, hand.size)
            self._filter.reset()
            self._recentre = False

        cx, cy, csize = self._centre
        c = self.config

        # The webcam image is mirrored first (like looking in a mirror), and
        # the simulator camera faces the robot. So moving my hand to my right
        # moves the arm toward the right of the screen, which is +y.
        side = (hand.x - cx) * c.gain_side
        up = (cy - hand.y) * c.gain_height            # image y grows downward
        reach = math.log2(hand.size / csize) * c.gain_reach  # bigger = closer = forward

        raw = (c.home[0] + reach, c.home[1] + side, c.home[2] + up)
        x, y, z = self._filter(raw, t)                # smooth, then clamp
        self.target = (_clamp(x, c.x_range), _clamp(y, c.y_range), _clamp(z, c.z_range))

        # Hysteresis: only change state when clearly past the far threshold.
        if self.closed and hand.pinch > c.open_above:
            self.closed = False
        elif not self.closed and hand.pinch < c.close_below:
            self.closed = True

        return self.target, self.closed
