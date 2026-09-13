"""Self-collision checker tests: distance primitive, clear vs colliding poses."""

import numpy as np

from puppet_pilot.kinematics.collision import segment_segment_distance


class TestSegmentDistance:
    def test_parallel_unit_apart(self):
        d = segment_segment_distance(
            np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]), np.array([1.0, 1.0, 0.0]),
        )
        assert abs(d - 1.0) < 1e-12

    def test_perpendicular_crossing_touches(self):
        d = segment_segment_distance(
            np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
            np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0]),
        )
        assert d < 1e-12

    def test_skew_lines_closest_between_endpoints(self):
        d = segment_segment_distance(
            np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
            np.array([0.0, -1.0, 0.5]), np.array([0.0, 1.0, 0.5]),
        )
        assert abs(d - 0.5) < 1e-12

    def test_endpoint_clamping(self):
        # Segments pointing away from each other: closest points are endpoints.
        d = segment_segment_distance(
            np.array([0.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0]),
        )
        assert abs(d - 2.0) < 1e-12

    def test_degenerate_point_segment(self):
        d = segment_segment_distance(
            np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]),  # a point
            np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
        )
        assert abs(d - 1.0) < 1e-12


class TestSelfCollision:
    def test_zero_pose_is_clear(self, checker):
        report = checker.check(np.zeros(6))
        assert not report.colliding, (
            "zero pose flagged: " + ", ".join(
                f"{p.name_a}-{p.name_b} ({p.distance:+.3f} m)"
                for p in report.pairs if p.colliding
            )
        )

    def test_typical_working_poses_are_clear(self, checker):
        """Poses of the kind teleop will command constantly must not trip the
        checker (false positives would make it useless as a veto)."""
        working_poses = [
            np.array([0.3, -1.2, 1.4, -1.7, -1.5, 0.0]),   # elbow-up reach
            np.array([-0.5, -0.9, 1.0, -1.6, -1.5, 0.5]),  # side reach
            np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0]),  # classic ready pose
        ]
        for q in working_poses:
            report = checker.check(q)
            assert not report.colliding, f"working pose {q} flagged {report.worst}"

    def test_folded_pose_collides(self, checker):
        """Fold the elbow fully so the wrist/gripper drives back into the
        upper arm - the checker must flag it."""
        q = np.array([0.0, -1.2, 3.1, 0.0, 0.0, 0.0])
        report = checker.check(q)
        assert report.colliding, f"folded pose not flagged; worst pair {report.worst}"

    def test_margin_shifts_distances(self, checker):
        q = np.zeros(6)
        margin = 0.02
        base = {(p.name_a, p.name_b): p.distance for p in checker.check(q).pairs}
        with_margin = checker.check(q, margin=margin)
        for pair in with_margin.pairs:
            expected = base[(pair.name_a, pair.name_b)] - margin
            assert abs(pair.distance - expected) < 1e-12
