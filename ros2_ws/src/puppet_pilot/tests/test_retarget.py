"""Retargeting math: axis map, smoothing, workspace clamp, pinch band."""

import numpy as np
import pytest

from puppet_pilot.teleop.retarget import Retargeter, RetargetConfig


@pytest.fixture()
def retargeter(spec):
    return Retargeter(spec)


def test_workspace_derived_from_spec_scene(retargeter, spec):
    """The task box must cover bin and both shelves and stay near the table."""
    low, high = retargeter.workspace_min, retargeter.workspace_max
    for cx, cy, _ in (spec.scene.bin_center, spec.scene.good_shelf.center,
                      spec.scene.reject_shelf.center):
        assert low[0] < cx < high[0] and low[1] < cy < high[1]
    assert high[2] == pytest.approx(spec.home.position[2])
    assert low[2] > 0


def test_targets_always_inside_workspace(retargeter, rng):
    """Wild operator input must never escape the box (hard clamp)."""
    for _ in range(200):
        wrist = rng.uniform(-3, 3, 3)  # far beyond any human reach box
        target, _ = retargeter.update(wrist, 0.5)
        assert np.all(target >= retargeter.workspace_min - 1e-9)
        assert np.all(target <= retargeter.workspace_max + 1e-9)


def test_ema_smoothing_matches_spec_alpha(spec):
    """One EMA step: new = alpha*sample + (1-alpha)*old, with SPEC's alpha."""
    r = Retargeter(spec)
    a, b = np.array([0.0, -0.3, -0.5]), np.array([0.4, -0.3, -0.5])
    first, _ = r.update(a, 0.5)
    second, _ = r.update(b, 0.5)
    # Feed 'a' forever into a fresh retargeter to get the pure-a target:
    r2 = Retargeter(spec)
    for _ in range(300):
        steady_a, _ = r2.update(a, 0.5)
    r3 = Retargeter(spec)
    for _ in range(300):
        steady_b, _ = r3.update(b, 0.5)
    alpha = spec.teleop.ema_alpha
    expected = steady_a + alpha * (steady_b - steady_a)  # linear map preserves EMA
    np.testing.assert_allclose(second, expected, atol=1e-6)


def test_smoothing_attenuates_jitter(spec, rng):
    """The point of the EMA: high-frequency noise shrinks, a lot."""
    r = Retargeter(spec)
    base = np.array([0.0, -0.2, -0.5])
    raw_targets, smoothed = [], []
    unsmoothed = Retargeter(spec)
    unsmoothed.alpha = 1.0  # disable smoothing for the comparison
    for _ in range(300):
        noisy = base + rng.normal(0, 0.03, 3)
        smoothed.append(r.update(noisy, 0.5)[0])
        raw_targets.append(unsmoothed.update(noisy, 0.5)[0])
    assert np.std(np.diff(smoothed, axis=0)) < 0.5 * np.std(np.diff(raw_targets, axis=0))


def test_pinch_band(retargeter, spec):
    """SPEC's thresholds: closed below 0.25, open above 0.75, linear between."""
    stroke = spec.gripper.stroke
    assert retargeter.opening_from_pinch(0.0) == 0.0
    assert retargeter.opening_from_pinch(0.24) == 0.0
    assert retargeter.opening_from_pinch(0.76) == pytest.approx(stroke)
    assert retargeter.opening_from_pinch(1.0) == pytest.approx(stroke)
    assert retargeter.opening_from_pinch(0.5) == pytest.approx(0.5 * stroke)


def test_axis_map_is_pure_rotation_like(retargeter):
    """The default axis map must be orthonormal (a signed permutation):
    anything else would scale or shear the operator's motion."""
    m = retargeter.config.axis_map
    np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-12)
