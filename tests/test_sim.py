"""Simulation tests: the arm reaches targets, the sticky grasp holds, and a
full pick-and-place lands an object in the bin. No ROS or webcam needed."""
import numpy as np
import pytest

from puppet_pilot import scene
from puppet_pilot.sim import ArmSim


@pytest.fixture()
def sim():
    return ArmSim()


def run(sim, seconds):
    for _ in range(int(seconds / sim.timestep)):
        sim.step()


def go(sim, xyz, closed=False, seconds=1.5):
    sim.set_command(xyz, closed)
    run(sim, seconds)


def test_home_pose_points_down_over_the_table(sim):
    x, y, z = sim.tip_position
    assert 0.3 < x < 0.7 and abs(y) < 0.2 and 0.1 < z < 0.6


def test_arm_reaches_targets_and_stays_pointing_down(sim):
    for target in ([0.45, -0.25, 0.10], [0.65, 0.20, 0.30], [0.50, 0.30, 0.20]):
        go(sim, target)
        assert np.linalg.norm(sim.tip_position - target) < 0.01
        tool_z = sim.data.site_xmat[sim._tcp].reshape(3, 3)[:, 2]
        assert tool_z[2] < -0.99          # gripper axis still straight down


def test_objects_rest_on_the_table(sim):
    run(sim, 1.0)
    for obj in scene.OBJECTS:
        assert abs(sim.data.body(obj.name).xpos[2] - scene.OBJECT_HALF_HEIGHT) < 0.003


def test_closing_on_nothing_grabs_nothing(sim):
    go(sim, [0.50, 0.10, 0.30], closed=True, seconds=0.5)
    assert sim.grasper.held is None


def test_pick_and_place_into_the_bin(sim):
    obj = scene.OBJECTS[0]
    x, y = obj.start
    go(sim, [x, y, 0.15])
    go(sim, [x, y, 0.035])
    go(sim, [x, y, 0.035], closed=True, seconds=0.5)
    assert sim.grasper.held == obj.name

    go(sim, [x, y, 0.25], closed=True)
    assert sim.data.body(obj.name).xpos[2] > 0.18      # it came up with the gripper

    bx, by = scene.BIN_CENTRE
    go(sim, [bx, by, 0.25], closed=True)
    go(sim, [bx, by, 0.25], closed=False, seconds=1.0)  # open: it drops
    assert sim.grasper.held is None
    assert obj.name in sim.grasper.in_bin


def test_reset_puts_everything_back(sim):
    test_pick_and_place_into_the_bin(sim)
    sim.reset()
    assert not sim.grasper.in_bin and sim.grasper.held is None
    obj = scene.OBJECTS[0]
    assert np.allclose(sim.data.body(obj.name).xpos[:2], obj.start, atol=1e-6)
