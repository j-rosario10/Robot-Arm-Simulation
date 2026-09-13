"""Quintic (C2-continuous) joint-space trajectories.

A quintic polynomial is the lowest order that admits six boundary conditions -
position, velocity and acceleration at both ends of a segment - which is what
makes the result twice continuously differentiable across waypoints. Cubic
splines leave acceleration discontinuous at segment ends (felt as jerks);
trapezoidal profiles are only C1. With zero boundary velocity/acceleration a
quintic segment is exactly the classic minimum-jerk profile.

Design notes in docs/phase2-kinematics.md section 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import numpy as np

from puppet_pilot.kinematics.params import JointLimits

# A rest-to-rest quintic reaches its peak velocity at mid-segment:
#   q(t) = q0 + dq * (10 s^3 - 15 s^4 + 6 s^5),  s = t/T
#   qdot(T/2) = dq/T * (30/8 - 60/16 + 30/16) = (15/8) * dq / T
_PEAK_VELOCITY_FACTOR = 15.0 / 8.0
# Peak acceleration of the same profile: qddot_max = 10/sqrt(3) * dq / T^2
_PEAK_ACCEL_FACTOR = 10.0 / math.sqrt(3.0)


@dataclass(frozen=True)
class QuinticSegment:
    """One quintic segment for n joints: coeffs shape (6, n), duration [s]."""

    coeffs: np.ndarray
    duration: float

    @classmethod
    def from_boundary_conditions(
        cls,
        q0: np.ndarray,
        q1: np.ndarray,
        duration: float,
        v0: np.ndarray | None = None,
        v1: np.ndarray | None = None,
        a0: np.ndarray | None = None,
        a1: np.ndarray | None = None,
    ) -> "QuinticSegment":
        """Solve the six boundary conditions for the coefficients.

        c0..c2 fall straight out of the conditions at t=0. The remaining three
        come from a 3x3 linear system at t=T - solved numerically rather than
        via hand-derived closed forms, because a solve of a 3x3 system is
        cheap, and the boundary matrix is self-documenting where the closed
        forms are a transcription hazard.
        """
        q0 = np.atleast_1d(np.asarray(q0, dtype=float))
        q1 = np.atleast_1d(np.asarray(q1, dtype=float))
        n = q0.shape[0]
        v0 = np.zeros(n) if v0 is None else np.asarray(v0, dtype=float)
        v1 = np.zeros(n) if v1 is None else np.asarray(v1, dtype=float)
        a0 = np.zeros(n) if a0 is None else np.asarray(a0, dtype=float)
        a1 = np.zeros(n) if a1 is None else np.asarray(a1, dtype=float)
        if duration <= 0.0:
            raise ValueError("segment duration must be positive")

        t = duration
        coeffs = np.zeros((6, n))
        coeffs[0] = q0
        coeffs[1] = v0
        coeffs[2] = a0 / 2.0

        # Conditions at t=T on the remaining coefficients c3, c4, c5:
        boundary = np.array(
            [
                [t**3, t**4, t**5],          # position
                [3 * t**2, 4 * t**3, 5 * t**4],  # velocity
                [6 * t, 12 * t**2, 20 * t**3],   # acceleration
            ]
        )
        residual = np.vstack(
            [
                q1 - (q0 + v0 * t + 0.5 * a0 * t**2),
                v1 - (v0 + a0 * t),
                a1 - a0,
            ]
        )
        coeffs[3:] = np.linalg.solve(boundary, residual)
        return cls(coeffs=coeffs, duration=duration)

    def evaluate(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Position, velocity, acceleration at time t (clamped to [0, T])."""
        t = min(max(t, 0.0), self.duration)
        c = self.coeffs
        powers = np.array([1.0, t, t**2, t**3, t**4, t**5])
        d_powers = np.array([0.0, 1.0, 2 * t, 3 * t**2, 4 * t**3, 5 * t**4])
        dd_powers = np.array([0.0, 0.0, 2.0, 6 * t, 12 * t**2, 20 * t**3])
        return powers @ c, d_powers @ c, dd_powers @ c


@dataclass(frozen=True)
class JointTrajectory:
    """A sequence of quintic segments through waypoints."""

    segments: tuple[QuinticSegment, ...]

    @property
    def duration(self) -> float:
        return sum(s.duration for s in self.segments)

    def evaluate(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if t <= 0.0:
            return self.segments[0].evaluate(0.0)
        for segment in self.segments:
            if t <= segment.duration:
                return segment.evaluate(t)
            t -= segment.duration
        return self.segments[-1].evaluate(self.segments[-1].duration)

    def sample(self, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Uniformly sampled (times, positions, velocities, accelerations)."""
        times = np.arange(0.0, self.duration + dt / 2, dt)
        q, v, a = zip(*(self.evaluate(t) for t in times))
        return times, np.array(q), np.array(v), np.array(a)


def plan_rest_to_rest(
    waypoints: list[np.ndarray],
    durations: list[float] | None = None,
    limits: JointLimits | None = None,
    velocity_fraction: float = 0.5,
    accel_limit: float | None = None,
    min_duration: float = 0.1,
) -> JointTrajectory:
    """Plan a trajectory that comes to rest (v=a=0) at every waypoint.

    Segment durations are either given explicitly or derived from the joint
    velocity limits: a rest-to-rest quintic peaks at (15/8)*|dq|/T, so the
    shortest duration that respects velocity_fraction * v_max is
        T = (15/8) * max_i(|dq_i| / v_max,i) / velocity_fraction.

    velocity_fraction defaults to 0.5: half the datasheet joint speed leaves
    headroom for the simulator's position controller to actually track the
    reference instead of saturating. It is a tuning knob, not a physical
    constant.

    UR publishes no acceleration limits for the e-series (joint_limits.yaml
    says so explicitly), so acceleration is only constrained when the caller
    supplies accel_limit [rad/s^2]; the peak of a rest-to-rest quintic is
    (10/sqrt(3))*|dq|/T^2, giving T >= sqrt((10/sqrt(3))*|dq|/a_max).
    """
    if len(waypoints) < 2:
        raise ValueError("need at least two waypoints")
    if durations is not None and len(durations) != len(waypoints) - 1:
        raise ValueError("need one duration per segment")
    if durations is None and limits is None:
        raise ValueError("provide either durations or joint limits")

    segments = []
    for i in range(len(waypoints) - 1):
        q0 = np.asarray(waypoints[i], dtype=float)
        q1 = np.asarray(waypoints[i + 1], dtype=float)
        if durations is not None:
            duration = durations[i]
        else:
            delta = np.abs(q1 - q0)
            duration = _PEAK_VELOCITY_FACTOR * float(np.max(delta / limits.velocity)) / velocity_fraction
            if accel_limit is not None and np.max(delta) > 0.0:
                duration = max(
                    duration,
                    math.sqrt(_PEAK_ACCEL_FACTOR * float(np.max(delta)) / accel_limit),
                )
            duration = max(duration, min_duration)
        segments.append(QuinticSegment.from_boundary_conditions(q0, q1, duration))
    return JointTrajectory(segments=tuple(segments))
