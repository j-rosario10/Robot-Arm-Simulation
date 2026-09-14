"""IK tests: round-trip accuracy, honest failure on unreachable targets,
and the singularity behaviour that motivates damped least-squares."""

import math

import numpy as np

from puppet_pilot.kinematics.fk import fk_frames, geometric_jacobian, tool_pose
from puppet_pilot.kinematics.ik import solve_ik


def test_round_trip_pose_recovery(dh, limits, rng):
    """pose -> joints -> pose must close within tolerance.

    Note the contract: the *pose* must be recovered, not the joint vector -
    a 6R arm has multiple joint solutions for one pose, and warm-starting
    near the truth only makes recovering q_true likely, not guaranteed.
    """
    for _ in range(25):
        q_true = rng.uniform(-math.pi, math.pi, 6)
        target = tool_pose(dh, q_true)
        q_start = q_true + rng.normal(0.0, 0.3, 6)

        result = solve_ik(dh, q_start, target, limits=limits)

        assert result.converged, f"IK failed from perturbed start (residual {result.position_error:.2e} m)"
        achieved = tool_pose(dh, result.q)
        assert np.linalg.norm(achieved[:3, 3] - target[:3, 3]) < 1e-4
        assert limits.contains(result.q)


def test_unreachable_target_reports_failure(dh, limits):
    """A target beyond the 0.85 m reach must yield converged=False and an
    honest residual - not a silent wrong answer."""
    target = np.eye(4)
    target[:3, 3] = [2.0, 0.0, 0.5]  # ~2 m away: physically unreachable

    result = solve_ik(dh, np.zeros(6), target, limits=limits)

    assert not result.converged
    assert result.position_error > 0.5  # most of the gap remains, and is reported
    assert np.all(np.isfinite(result.q))
    assert limits.contains(result.q)


class TestWristSingularity:
    """q5 = 0 aligns the wrist_1 and wrist_3 axes: the classic UR wrist
    singularity. These tests demonstrate exactly what DLS buys there."""

    # A representative elbow-up pose with the wrist singular (q5 ~ 0).
    Q_SINGULAR = np.array([0.3, -1.2, 1.4, -1.7, 1e-6, 0.2])

    def test_jacobian_is_singular_here(self, dh):
        sigma = np.linalg.svd(geometric_jacobian(dh, self.Q_SINGULAR), compute_uv=False)
        assert sigma[-1] < 1e-4, "expected a near-zero singular value at the wrist singularity"

    def test_pseudoinverse_explodes_dls_stays_bounded(self, dh):
        """Same task error, two solvers: the pseudoinverse step is enormous,
        the DLS step respects its theoretical bound |dq| <= |e| / (2*lambda)."""
        jac = geometric_jacobian(dh, self.Q_SINGULAR)
        u, sigma, _ = np.linalg.svd(jac)
        # Ask for motion along the direction the mechanism is losing:
        error = u[:, -1] * 1e-2

        step_pinv = np.linalg.pinv(jac, rcond=1e-12) @ error
        damping = 0.05
        step_dls = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(6), error)

        assert np.linalg.norm(step_pinv) > 50 * np.linalg.norm(step_dls)
        assert np.linalg.norm(step_dls) <= np.linalg.norm(error) / (2 * damping) + 1e-12

    def test_ik_near_singularity_returns_finite_result(self, dh, limits):
        """Solving *to* a pose in the singular neighbourhood must stay finite
        and end near the target - sluggish is acceptable, exploding is not."""
        target = tool_pose(dh, self.Q_SINGULAR)
        q_start = self.Q_SINGULAR + 0.05

        result = solve_ik(dh, q_start, target, limits=limits, max_iterations=300)

        assert np.all(np.isfinite(result.q))
        assert result.position_error < 1e-3  # within a millimetre even if not fully converged
