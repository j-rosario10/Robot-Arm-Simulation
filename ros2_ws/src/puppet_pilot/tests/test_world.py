"""Simulation world tests - these are the Phase 3 acceptance criteria.

The grasp test is the load-bearing one: it drives the arm with the Phase 2
kinematics, closes the jaws on a real part, lifts, and checks the part came
with it. That exercises the whole stack at once - SPEC parsing, procedural
geometry, inertia, friction, the gripper's gear constraints, and IK.
"""

import numpy as np
import pytest

from puppet_pilot.kinematics.fk import tool_pose
from puppet_pilot.sim.world import WORLD_FROM_DH, SortingWorld


@pytest.fixture(scope="module")
def world():
    sim = SortingWorld.create(gui=False, seed=7)
    yield sim
    sim.disconnect()


class TestScene:
    def test_episode_composition_matches_spec(self, world, spec):
        assert len(world.parts) == spec.episode.parts_per_episode
        assert sum(part.is_reject for part in world.parts) == spec.episode.rejects_per_episode
        kinds = {part.kind for part in world.parts}
        assert kinds == {"hex_bolt", "bushing", "ring", "dowel"}, "every kind must appear"

    def test_all_parts_start_in_the_bin(self, world):
        for part in world.parts:
            assert world.part_location(part) == "bin", f"{part.kind} spawned outside the bin"

    def test_parts_rest_on_the_bin_floor(self, world, spec):
        """Nothing fell through the floor, nothing is floating."""
        floor = spec.scene.bin_thickness
        for part in world.parts:
            z = part.pose()[0][2]
            assert floor < z < floor + 0.05, f"{part.kind} at z={z:.4f}, floor at {floor:.4f}"

    def test_parts_settle_and_stay_settled(self, world, spec):
        """The jitter/creep check: after settling, a further second of physics
        must not move anything. Wrong inertia shows up here first."""
        world.settle()
        before = np.array([part.pose()[0] for part in world.parts])
        world.step(int(1.0 / world.timestep))
        after = np.array([part.pose()[0] for part in world.parts])

        drift = np.linalg.norm(after - before, axis=1)
        assert drift.max() < 1e-3, f"parts crept up to {drift.max() * 1000:.2f} mm while idle"

    def test_episode_reset_is_reproducible(self, world):
        """Same seed, same layout - required for reproducible demonstrations.

        Reset the *same* world twice rather than building two: PyBullet's
        module-level API targets one implicit client, so a second connection
        would land its bodies in the first world.
        """
        world.reset_episode(99)
        layout_a = [(part.kind, part.is_reject, part.pose()[0]) for part in world.parts]

        world.reset_episode(99)
        layout_b = [(part.kind, part.is_reject, part.pose()[0]) for part in world.parts]

        assert [k for k, _, _ in layout_a] == [k for k, _, _ in layout_b]
        assert [r for _, r, _ in layout_a] == [r for _, r, _ in layout_b]
        for (_, _, pos_a), (_, _, pos_b) in zip(layout_a, layout_b):
            assert np.allclose(pos_a, pos_b, atol=1e-6)

        world.reset_episode(7)  # restore the seed the other tests expect

    def test_second_world_is_refused(self, world):
        """The one-world-per-process guard must fire rather than silently
        corrupting the running world."""
        with pytest.raises(RuntimeError, match="already connected"):
            SortingWorld.create(gui=False)


class TestRobot:
    def test_arm_tracks_a_joint_command(self, world):
        target = np.array([0.3, -1.2, 1.4, -1.7, -1.5, 0.0])
        world.command_joints(target)
        world.step(int(3.0 / world.timestep))
        np.testing.assert_allclose(world.joint_positions(), target, atol=0.02)

    def test_simulated_fk_agrees_with_our_kinematics(self, world, dh):
        """The sim's tool pose must match the Phase 2 FK through the pinned
        world<-DH correction - proof the two stay in sync."""
        q = np.array([0.4, -1.0, 1.2, -1.8, -1.4, 0.6])
        world.reset_joints(q)
        world.step(5)

        simulated_position, _ = world.tool_pose()
        predicted = (WORLD_FROM_DH @ tool_pose(dh, q))[:3, 3]
        np.testing.assert_allclose(simulated_position, predicted, atol=1e-4)

    def test_gripper_opens_and_closes(self, world, spec):
        world.command_gripper(spec.gripper.stroke)      # fully open
        world.step(int(2.0 / world.timestep))
        assert world.gripper_opening() > 0.9 * spec.gripper.stroke

        world.command_gripper(0.0)                       # fully closed
        world.step(int(2.0 / world.timestep))
        assert world.gripper_opening() < 0.1 * spec.gripper.stroke

    def test_mimic_joints_follow_the_driven_joint(self, world, spec):
        """PyBullet ignores <mimic>; the explicit mirroring must restore it,
        or the fingers flop and nothing can be grasped."""
        import pybullet as p

        for opening in (0.0, spec.gripper.stroke):
            world.command_gripper(opening)
            world.step(int(2.0 / world.timestep))

            driven = p.getJointState(world.robot, world.gripper.driven_index)[0]
            for index, multiplier in world.gripper.mimic_indices.items():
                follower = p.getJointState(world.robot, index)[0]
                assert follower == pytest.approx(multiplier * driven, abs=0.05), (
                    f"joint {index} at {follower:.3f}, expected {multiplier * driven:.3f}"
                )


class TestHomePose:
    def test_home_matches_spec_position(self, world, spec):
        """SPEC's home pose must be solvable and actually reached."""
        world.reset_joints(world.home_configuration)
        world.step(int(1.0 / world.timestep))

        tool_position = np.array(world.tool_pose()[0])
        jaw_centre = tool_position + np.array([0.0, 0.0, -world.jaw_offset])
        np.testing.assert_allclose(jaw_centre, spec.home.position, atol=3e-3)

    def test_arm_holds_home_against_gravity(self, world):
        """The arm must not sag when parked - a sagging arm rakes through the
        bin and scatters the parts."""
        world.reset_joints(world.home_configuration)
        world.step(int(2.0 / world.timestep))
        np.testing.assert_allclose(
            world.joint_positions(), world.home_configuration, atol=0.02
        )


class TestGraspAssist:
    def test_lock_then_release(self, world, spec):
        """Closing on a nearby part welds it to the tool; opening lets go."""
        world.reset_episode(7)
        bushing = next(part for part in world.parts if part.kind == "bushing")
        start = np.array(bushing.pose()[0])
        grasp = np.array([start[0], start[1], world.grasp_height(spec.scene.bin_thickness)])

        world.command_gripper(spec.gripper.stroke)
        assert world.move_to(grasp + [0, 0, 0.10], seed=world.home_configuration).converged
        assert world.move_to(grasp, seconds=1.5).converged
        assert world.assist.held is None

        world.command_gripper(0.0)
        world.step(int(2.0 / world.timestep))
        assert world.assist.held is bushing, "assist should have locked the bushing"

        world.move_to(grasp + [0, 0, 0.12], seconds=1.5)
        assert bushing.pose()[0][2] - start[2] > 0.08, "held part must travel with the tool"

        world.command_gripper(spec.gripper.stroke)
        world.step(int(1.0 / world.timestep))
        assert world.assist.held is None, "opening the jaws must release the part"


class TestGrasp:
    """SPEC's own words: the bushing is 'the most forgiving part - use for
    early testing'. If the bushing cannot be picked up, nothing can."""

    def test_pick_up_a_bushing(self, world, spec):
        world.reset_episode()
        bushing = next(part for part in world.parts if part.kind == "bushing")
        start_position = np.array(bushing.pose()[0])

        # Positions are pad-tip positions. Bring the tips down to just above
        # the bin floor so the pads wrap the full height of the part.
        floor = spec.scene.bin_thickness
        grasp = np.array([start_position[0], start_position[1],
                          world.grasp_height(floor)])
        approach = grasp + np.array([0.0, 0.0, 0.10])

        world.command_gripper(spec.gripper.stroke)
        assert world.move_to(approach, seed=world.home_configuration).converged
        assert world.move_to(grasp, seconds=1.5).converged

        world.command_gripper(0.0)
        world.step(int(2.0 / world.timestep))

        assert world.move_to(approach, seconds=2.0).converged

        lifted = np.array(bushing.pose()[0])
        rise = lifted[2] - start_position[2]
        assert rise > 0.05, f"bushing rose only {rise * 1000:.1f} mm - grasp failed"
        assert 0.0 < world.gripper_opening() < spec.gripper.stroke, (
            "jaws fully closed: nothing is held between them"
        )
