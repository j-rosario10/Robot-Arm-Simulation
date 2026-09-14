"""SPEC.md parsing: every value the simulator depends on, restated here.

These assertions duplicate SPEC.md on purpose. If someone edits SPEC.md, the
tests fail and force a conscious decision - which is exactly the behaviour we
want from a single source of truth.
"""

import pytest

from puppet_pilot.spec import SpecError, load_spec


def test_part_dimensions_in_metres(spec):
    bolt = spec.parts["hex_bolt"]
    assert bolt.across_flats == pytest.approx(0.013)
    assert bolt.head_height == pytest.approx(0.0053)
    assert bolt.shank_diameter == pytest.approx(0.008)
    assert bolt.overall_length == pytest.approx(0.040)

    assert spec.parts["bushing"].outer_diameter == pytest.approx(0.025)
    assert spec.parts["bushing"].inner_diameter == pytest.approx(0.015)
    assert spec.parts["bushing"].height == pytest.approx(0.020)

    assert spec.parts["ring"].outer_diameter == pytest.approx(0.028)
    assert spec.parts["dowel"].diameter == pytest.approx(0.010)
    assert spec.parts["dowel"].length == pytest.approx(0.045)


def test_material_and_colours(spec):
    assert spec.material.density == pytest.approx(7850.0)
    assert spec.material.part_friction == pytest.approx(0.6)
    assert spec.material.pad_friction == pytest.approx(0.9)
    assert spec.material.table_friction == pytest.approx(0.5)
    assert spec.material.rolling_friction == pytest.approx(0.001)
    assert spec.material.spinning_friction == pytest.approx(0.001)
    assert spec.material.good_rgb == pytest.approx((0.60, 0.63, 0.65))
    assert spec.material.reject_rgb == pytest.approx((0.75, 0.23, 0.17))


def test_episode_and_success(spec):
    assert spec.episode.parts_per_episode == 8
    assert spec.episode.rejects_per_episode == 3
    assert spec.episode.timeout == pytest.approx(180.0)
    assert spec.success.settle_velocity == pytest.approx(0.005)
    assert spec.success.settle_time == pytest.approx(0.5)


def test_scene_layout(spec):
    scene = spec.scene
    assert scene.bin_center == pytest.approx((0.450, -0.250, 0.0))
    assert scene.bin_interior == pytest.approx((0.220, 0.220))
    assert scene.bin_wall_height == pytest.approx(0.060)
    assert scene.bin_thickness == pytest.approx(0.010)
    assert scene.good_shelf.center == pytest.approx((0.450, 0.250, 0.0))
    assert scene.reject_shelf.center == pytest.approx((0.250, 0.400, 0.0))
    assert scene.good_shelf.size == pytest.approx((0.200, 0.200))
    assert scene.good_shelf.lip_height == pytest.approx(0.010)
    assert scene.table_size == pytest.approx((1.400, 1.000))
    assert spec.home.position == pytest.approx((0.450, -0.250, 0.350))


def test_gripper_teleop_recording(spec):
    assert spec.gripper.stroke == pytest.approx(0.085)
    assert spec.gripper.max_force == pytest.approx(60.0)
    assert spec.gripper.close_speed == pytest.approx(0.060)

    assert spec.teleop.publish_rate == pytest.approx(30.0)
    assert spec.teleop.ema_alpha == pytest.approx(0.35)
    assert spec.teleop.pinch_closed == pytest.approx(0.25)
    assert spec.teleop.pinch_open == pytest.approx(0.75)

    assert spec.recording.log_rate == pytest.approx(30.0)
    assert spec.recording.target_demo_count == 50
    assert spec.recording.train_fraction == pytest.approx(0.8)


def test_grasp_assist(spec):
    assert spec.assist.enabled is True
    assert spec.assist.radius == pytest.approx(0.040)
    assert spec.assist.force == pytest.approx(1.5)
    assert spec.assist.lock_distance == pytest.approx(0.012)
    assert spec.assist.release_margin == pytest.approx(0.015)


def test_scene_is_within_ur5e_reach(spec):
    """Sanity check SPEC's own claim that every target is reachable.

    UR5e reach is 850 mm; require margin for approach and retreat poses.
    """
    import math

    scene = spec.scene
    for label, target in (
        ("bin", scene.bin_center),
        ("good shelf", scene.good_shelf.center),
        ("reject shelf", scene.reject_shelf.center),
    ):
        distance = math.hypot(target[0], target[1])
        assert distance < 0.75, f"{label} at {distance:.3f} m leaves too little reach margin"


def test_missing_section_raises_clearly(tmp_path):
    """A truncated SPEC must fail loudly, naming what it looked for."""
    broken = tmp_path / "SPEC.md"
    broken.write_text("# SPEC.md\n\n## Part inventory\n\nnothing useful here\n", encoding="utf-8")
    with pytest.raises(SpecError, match="no section starting with"):
        load_spec(broken)


def test_missing_value_in_present_section_raises_clearly(tmp_path, spec_path):
    """A section that exists but lost a value must also fail loudly, naming
    the pattern - the common case after a careless SPEC edit."""
    text = spec_path.read_text(encoding="utf-8")
    broken = tmp_path / "SPEC.md"
    broken.write_text(text.replace("| Max grip force | 60 N |", "| Max grip force | ? |"), encoding="utf-8")
    with pytest.raises(SpecError, match="could not find"):
        load_spec(broken)
