"""Quintic trajectory tests: boundary conditions, continuity, limit respect."""

import numpy as np

from puppet_pilot.kinematics.trajectory import (
    QuinticSegment,
    plan_rest_to_rest,
)


def test_boundary_conditions_are_exact(rng):
    q0 = rng.uniform(-3.0, 3.0, 6)
    q1 = rng.uniform(-3.0, 3.0, 6)
    v0 = rng.uniform(-1.0, 1.0, 6)
    v1 = rng.uniform(-1.0, 1.0, 6)
    a0 = rng.uniform(-2.0, 2.0, 6)
    a1 = rng.uniform(-2.0, 2.0, 6)
    seg = QuinticSegment.from_boundary_conditions(q0, q1, 1.7, v0, v1, a0, a1)

    for t, q_exp, v_exp, a_exp in [(0.0, q0, v0, a0), (1.7, q1, v1, a1)]:
        q, v, a = seg.evaluate(t)
        np.testing.assert_allclose(q, q_exp, atol=1e-9)
        np.testing.assert_allclose(v, v_exp, atol=1e-9)
        np.testing.assert_allclose(a, a_exp, atol=1e-9)


def test_rest_to_rest_is_c2_across_waypoints(rng):
    """v and a must both vanish at interior waypoints (that is what makes the
    concatenation C2: both segments meet the waypoint at rest)."""
    waypoints = [rng.uniform(-2.0, 2.0, 6) for _ in range(3)]
    traj = plan_rest_to_rest(waypoints, durations=[1.0, 1.5])

    junction = 1.0
    for t in (junction - 1e-9, junction, junction + 1e-9):
        q, v, a = traj.evaluate(t)
        np.testing.assert_allclose(v, 0.0, atol=1e-6)
        np.testing.assert_allclose(a, 0.0, atol=1e-5)
    np.testing.assert_allclose(traj.evaluate(junction)[0], waypoints[1], atol=1e-9)


def test_velocity_limits_respected(limits, rng):
    """Auto-computed durations must keep every joint under the requested
    fraction of its datasheet velocity limit."""
    fraction = 0.5
    waypoints = [rng.uniform(-2.0, 2.0, 6) for _ in range(4)]
    traj = plan_rest_to_rest(waypoints, limits=limits, velocity_fraction=fraction)

    _, _, velocities, _ = traj.sample(dt=0.002)
    peak_ratio = np.max(np.abs(velocities) / limits.velocity)
    assert peak_ratio <= fraction * 1.001  # tiny numerical headroom


def test_duration_formula(limits):
    """T = (15/8) * max(|dq|/v_max) / fraction for the dominant joint."""
    q0 = np.zeros(6)
    q1 = np.zeros(6)
    q1[2] = 1.0  # elbow moves 1 rad, everything else stays
    traj = plan_rest_to_rest([q0, q1], limits=limits, velocity_fraction=0.5)

    expected = (15.0 / 8.0) * (1.0 / limits.velocity[2]) / 0.5
    np.testing.assert_allclose(traj.duration, expected, rtol=1e-9)


def test_zero_displacement_gets_minimum_duration(limits):
    traj = plan_rest_to_rest([np.zeros(6), np.zeros(6)], limits=limits, min_duration=0.1)
    assert traj.duration == 0.1
    q, v, a = traj.evaluate(0.05)
    np.testing.assert_allclose(q, 0.0, atol=1e-12)
    np.testing.assert_allclose(v, 0.0, atol=1e-12)
