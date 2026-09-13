"""Retargeting: shoulder-relative wrist + pinch  ->  tool target + jaw opening.

Pure math, no ROS and no PyBullet, so every mapping decision is unit-testable.
The full design discussion (what the 7->6 DOF mapping keeps, what it discards,
and why) is in docs/phase4-teleop.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from puppet_pilot.spec import SortingSpec

# Margin added around the SPEC scene when deriving the robot task box [m].
_WORKSPACE_MARGIN = 0.06
# Height ceiling above the table for the teleop workspace: SPEC's home pose
# height is the natural "retracted" altitude, so the box tops out there.


@dataclass
class RetargetConfig:
    """Tunables for the operator -> robot mapping.

    ``axis_map`` rotates/flips MediaPipe's camera frame (x right, y down,
    z away from the camera) into the robot frame. The right values depend on
    where the camera physically sits relative to the operator - the brief
    budgets live iteration for exactly this. Defaults assume the operator
    faces the camera, "pushing toward the camera" reaches forward:

        robot x (forward) = -z_cam    (hand toward camera = reach out)
        robot y (left)    = -x_cam    (hand to operator's right = robot -y,
                                       putting the bin under the right hand)
        robot z (up)      = -y_cam    (camera y points down)
    """

    axis_map: np.ndarray = field(default_factory=lambda: np.array([
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]))
    # Operator reach box, in arm-lengths, AFTER the axis map (robot axes).
    # Maps affinely onto the robot task box below.
    human_box_min: np.ndarray = field(default_factory=lambda: np.array([0.15, -0.55, -0.55]))
    human_box_max: np.ndarray = field(default_factory=lambda: np.array([0.85, 0.55, 0.35]))
    stale_after: float = 0.3   # s without operator data before the arm holds pose


class Retargeter:
    """Maps operator samples to (pad-tip target [m], jaw opening [m]).

    Smoothing (SPEC's EMA alpha) is applied to the wrist vector *before* the
    workspace map and IK, so the joint commands inherit it - smoothing after
    IK would let jitter excite the solver and the singularity handling first.
    """

    def __init__(self, spec: SortingSpec, config: RetargetConfig | None = None):
        self.spec = spec
        self.config = config or RetargetConfig()
        self.alpha = spec.teleop.ema_alpha
        self._smoothed: np.ndarray | None = None
        self.workspace_min, self.workspace_max = _task_box(spec)

    def reset(self) -> None:
        self._smoothed = None

    def update(self, wrist: np.ndarray, pinch: float) -> tuple[np.ndarray, float]:
        """One operator sample in, one (target, opening) command out."""
        mapped = self.config.axis_map @ np.asarray(wrist, dtype=float)

        # EMA: new = alpha * sample + (1 - alpha) * previous.
        if self._smoothed is None:
            self._smoothed = mapped
        else:
            self._smoothed = self.alpha * mapped + (1.0 - self.alpha) * self._smoothed

        # Affine map from the operator's reach box onto the robot task box,
        # then clamp: the edges of the box are hard walls, by design.
        c = self.config
        fraction = (self._smoothed - c.human_box_min) / (c.human_box_max - c.human_box_min)
        target = self.workspace_min + fraction * (self.workspace_max - self.workspace_min)
        target = np.clip(target, self.workspace_min, self.workspace_max)

        return target, self.opening_from_pinch(pinch)

    def opening_from_pinch(self, pinch: float) -> float:
        """SPEC's two pinch thresholds define a proportional band with dead
        zones: fully closed below 0.25, fully open above 0.75, linear between.
        The dead zones give stable extremes without a hysteresis state
        machine."""
        teleop = self.spec.teleop
        fraction = (pinch - teleop.pinch_closed) / (teleop.pinch_open - teleop.pinch_closed)
        return float(np.clip(fraction, 0.0, 1.0)) * self.spec.gripper.stroke


def _task_box(spec: SortingSpec) -> tuple[np.ndarray, np.ndarray]:
    """Robot task box derived from SPEC's scene: the bounding box of the bin
    and both shelves plus a margin, floor to home height. If SPEC's layout
    changes, the teleop workspace follows automatically."""
    scene = spec.scene
    footprints = [
        (scene.bin_center, scene.bin_interior),
        (scene.good_shelf.center, scene.good_shelf.size),
        (scene.reject_shelf.center, scene.reject_shelf.size),
    ]
    xs, ys = [], []
    for (cx, cy, _), (sx, sy) in footprints:
        xs += [cx - sx / 2, cx + sx / 2]
        ys += [cy - sy / 2, cy + sy / 2]
    low = np.array([min(xs) - _WORKSPACE_MARGIN, min(ys) - _WORKSPACE_MARGIN,
                    scene.bin_thickness + 0.005])
    high = np.array([max(xs) + _WORKSPACE_MARGIN, max(ys) + _WORKSPACE_MARGIN,
                     spec.home.position[2]])
    return low, high
