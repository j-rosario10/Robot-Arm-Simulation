"""Tests for the plain-Python core. None of these need ROS, a webcam or MuJoCo."""
import math

from puppet_pilot.filters import OneEuroFilter
from puppet_pilot.hand_math import read_hand
from puppet_pilot.mapping import HandToArm, MappingConfig


def make_hand(cx=0.5, cy=0.5, scale=1.0, pinch_gap=0.15):
    """A fake hand: 21 landmarks, palm about 0.2 image-heights tall."""
    pts = [(cx, cy)] * 21
    pts[0] = (cx, cy + 0.10 * scale)                 # wrist
    pts[5] = (cx - 0.04 * scale, cy - 0.08 * scale)  # index knuckle
    pts[9] = (cx, cy - 0.10 * scale)                 # middle knuckle
    pts[17] = (cx + 0.05 * scale, cy - 0.06 * scale) # pinky knuckle
    pts[4] = (cx - 0.05 * scale, cy - 0.15 * scale)  # thumb tip
    pts[8] = (cx - 0.05 * scale, cy - 0.15 * scale - pinch_gap * scale)  # index tip
    return read_hand(pts)


def test_pinch_is_the_same_near_and_far():
    near, far = make_hand(scale=1.5), make_hand(scale=0.6)
    assert near.size > far.size
    assert math.isclose(near.pinch, far.pinch, rel_tol=1e-6)


def test_filter_removes_jitter_but_follows_movement():
    f = OneEuroFilter()
    out = [f(1.0 + (0.01 if i % 2 else -0.01), i / 30) for i in range(60)]
    assert max(out[30:]) - min(out[30:]) < 0.01      # 2 cm of shake mostly gone
    for i in range(60, 75):                          # then a real, fast move
        y = f(2.0, i / 30)
    assert y > 1.9                                   # caught up within half a second


def test_first_hand_becomes_the_centre():
    m = HandToArm()
    target, closed = m.update(make_hand(0.3, 0.7), 0.0)
    assert target == MappingConfig().home and closed is False


def test_directions():
    cfg = MappingConfig()
    def moved(**kw):
        m = HandToArm()
        m.update(make_hand(), 0.0)
        for i in range(1, 40):
            t, _ = m.update(make_hand(**kw), i / 30)
        return t
    assert moved(cx=0.6)[1] > cfg.home[1]       # hand right in image -> +y side
    assert moved(cy=0.4)[2] > cfg.home[2]       # hand up -> arm up
    assert moved(scale=1.3)[0] > cfg.home[0]    # hand closer -> reach forward


def test_target_never_leaves_the_safe_box():
    m, cfg = HandToArm(), MappingConfig()
    m.update(make_hand(), 0.0)
    for i in range(1, 60):
        (x, y, z), _ = m.update(make_hand(cx=5.0, cy=5.0, scale=4.0), i / 30)
    assert cfg.x_range[0] <= x <= cfg.x_range[1]
    assert cfg.y_range[0] <= y <= cfg.y_range[1]
    assert cfg.z_range[0] <= z <= cfg.z_range[1]


def test_pinch_hysteresis_does_not_flicker():
    m = HandToArm()
    m.update(make_hand(), 0.0)
    _, closed = m.update(make_hand(pinch_gap=0.03), 0.1)
    assert closed                                    # clearly pinched -> closed
    _, closed = m.update(make_hand(pinch_gap=0.08), 0.2)
    assert closed                                    # in between -> stays closed
    _, closed = m.update(make_hand(pinch_gap=0.15), 0.3)
    assert not closed                                # clearly open -> opens


def test_no_hand_holds_position():
    m = HandToArm()
    m.update(make_hand(), 0.0)
    before = m.update(make_hand(cx=0.6, pinch_gap=0.03), 0.1)
    assert m.update(None, 0.2) == before
