"""SPEC.md parser - the single source of truth for every task parameter.

Nothing in the codebase hardcodes task numbers: dimensions, colours, rates,
forces and thresholds are all read from SPEC.md at the repository root through
this module. The parser is deliberately strict - if an expected value cannot
be found (say, after an edit to SPEC.md), it raises with the section and
pattern it was looking for, rather than falling back to a silent default.

All lengths in SPEC.md are millimetres; everything this module returns is SI
(metres, kilograms, seconds, radians) unless a field name says otherwise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


def repo_root() -> Path:
    """The repository root (where SPEC.md lives), located relative to this file."""
    return Path(__file__).resolve().parents[3]


class SpecError(RuntimeError):
    """SPEC.md did not contain something the code needs."""


def _clean(text: str) -> str:
    """Normalise unicode that appears in SPEC.md (minus signs, multiply signs)."""
    return text.replace("−", "-").replace("×", "x").replace("ø", "O").replace("Ø", "O")


def _grab(pattern: str, text: str, context: str) -> str:
    match = re.search(pattern, text)
    if not match:
        raise SpecError(f"SPEC.md: could not find /{pattern}/ in {context}")
    return match.group(1)


def _num(pattern: str, text: str, context: str) -> float:
    return float(_grab(pattern, text, context))


def _mm(value: float) -> float:
    return value / 1000.0


def _sections(text: str) -> dict[str, str]:
    """Split the document into '## heading' -> body."""
    sections: dict[str, str] = {}
    current = "_preamble"
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            sections[current] = "\n".join(lines)
            current = line[3:].strip()
            lines = []
        else:
            lines.append(line)
    sections[current] = "\n".join(lines)
    return sections


def _section(sections: dict[str, str], name_prefix: str) -> str:
    for heading, body in sections.items():
        if heading.startswith(name_prefix):
            return body
    raise SpecError(f"SPEC.md: no section starting with {name_prefix!r}")


# --------------------------------------------------------------------------
# typed spec fragments
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PartDims:
    """Per-kind dimensions [m]. Only the fields for that kind are set."""

    outer_diameter: float = 0.0   # bushing, ring
    inner_diameter: float = 0.0   # bushing, ring (visual bore only)
    height: float = 0.0           # bushing, ring
    diameter: float = 0.0         # dowel
    length: float = 0.0           # dowel
    across_flats: float = 0.0     # hex bolt head
    head_height: float = 0.0      # hex bolt
    shank_diameter: float = 0.0   # hex bolt
    overall_length: float = 0.0   # hex bolt (head + shank)


@dataclass(frozen=True)
class MaterialSpec:
    density: float          # kg/m^3
    part_friction: float
    pad_friction: float
    table_friction: float
    rolling_friction: float
    spinning_friction: float
    good_rgb: tuple[float, float, float]
    reject_rgb: tuple[float, float, float]


@dataclass(frozen=True)
class EpisodeSpec:
    parts_per_episode: int
    rejects_per_episode: int
    timeout: float          # s


@dataclass(frozen=True)
class BoxPad:
    center: tuple[float, float, float]
    size: tuple[float, float]       # pad footprint (x, y)
    pad_thickness: float
    lip_height: float
    lip_thickness: float


@dataclass(frozen=True)
class SceneSpec:
    bin_center: tuple[float, float, float]
    bin_interior: tuple[float, float]   # usable interior footprint (x, y)
    bin_wall_height: float
    bin_thickness: float                # wall and floor thickness
    good_shelf: BoxPad
    reject_shelf: BoxPad
    table_size: tuple[float, float]


@dataclass(frozen=True)
class HomeSpec:
    position: tuple[float, float, float]  # tool0 position, tool pointing down


@dataclass(frozen=True)
class GripperSpec:
    stroke: float        # m, fully open to fully closed
    max_force: float     # N, fixed ceiling (not live-modulated; see SPEC)
    close_speed: float   # m/s at the jaws


@dataclass(frozen=True)
class GraspAssistSpec:
    enabled: bool
    radius: float           # m, attraction zone around the jaw centre
    force: float            # N, pull toward the jaw centre while closing
    lock_distance: float    # m, snap-and-hold threshold
    release_margin: float   # m, jaws must open this much wider than the part


@dataclass(frozen=True)
class SuccessSpec:
    settle_velocity: float   # m/s - "resting stably" threshold
    settle_time: float       # s the threshold must hold


@dataclass(frozen=True)
class TeleopSpec:
    publish_rate: float      # Hz
    ema_alpha: float
    pinch_closed: float
    pinch_open: float


@dataclass(frozen=True)
class RecordingSpec:
    log_rate: float          # Hz
    target_demo_count: int
    train_fraction: float


@dataclass(frozen=True)
class SortingSpec:
    parts: dict               # kind -> PartDims
    material: MaterialSpec
    episode: EpisodeSpec
    scene: SceneSpec
    home: HomeSpec
    gripper: GripperSpec
    assist: GraspAssistSpec
    success: SuccessSpec
    teleop: TeleopSpec
    recording: RecordingSpec


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

_NUMX = r"-?\d+(?:\.\d+)?"   # bare number, no capture group
_NUM = f"({_NUMX})"          # capturing variant - exactly one group


def _parse_parts(body: str) -> dict:
    parts = {}
    ctx = "Part inventory"

    bolt_row = _grab(r"\|\s*(Hex bolt[^|]*\|[^|]*)\|", body, ctx)
    parts["hex_bolt"] = PartDims(
        across_flats=_mm(_num(rf"Head {_NUM} across flats", bolt_row, ctx)),
        head_height=_mm(_num(rf"across flats x {_NUM} tall", bolt_row, ctx)),
        shank_diameter=_mm(_num(rf"shank O{_NUM}", bolt_row, ctx)),
        overall_length=_mm(_num(rf"overall length {_NUM}", bolt_row, ctx)),
    )

    for kind, label in (("bushing", "Bushing"), ("ring", "Ring")):
        row = _grab(rf"\|\s*({label}\s*\|[^|]*)\|", body, ctx)
        parts[kind] = PartDims(
            outer_diameter=_mm(_num(rf"O{_NUM} OD", row, ctx)),
            inner_diameter=_mm(_num(rf"O{_NUM} ID", row, ctx)),
            height=_mm(_num(rf"x {_NUM} tall", row, ctx)),
        )

    dowel_row = _grab(r"\|\s*(Dowel pin\s*\|[^|]*)\|", body, ctx)
    parts["dowel"] = PartDims(
        diameter=_mm(_num(rf"O{_NUM} x", dowel_row, ctx)),
        length=_mm(_num(rf"x {_NUM} long", dowel_row, ctx)),
    )
    return parts


def _parse_material(inventory: str, colours: str) -> MaterialSpec:
    def rgb(name: str) -> tuple[float, float, float]:
        match = re.search(rf"{name}[^`]*`\(({_NUMX}), ({_NUMX}), ({_NUMX})\)`", colours)
        if not match:
            raise SpecError(f"SPEC.md: no RGB triple for {name!r} in Good vs reject section")
        return (float(match.group(1)), float(match.group(2)), float(match.group(3)))

    return MaterialSpec(
        density=_num(rf"density {_NUM} kg/m", inventory, "Part mass"),
        part_friction=_num(rf"lateral friction coefficient {_NUM} for all parts", inventory, "Friction"),
        pad_friction=_num(rf"Gripper pad friction {_NUM}", inventory, "Friction"),
        table_friction=_num(rf"Table friction {_NUM}", inventory, "Friction"),
        rolling_friction=_num(rf"Rolling friction {_NUM}", inventory, "Friction"),
        spinning_friction=_num(rf"spinning friction {_NUM}", inventory, "Friction"),
        good_rgb=rgb("Good part"),
        reject_rgb=rgb("Reject part"),
    )


def _parse_episode(body: str, success_body: str) -> EpisodeSpec:
    return EpisodeSpec(
        parts_per_episode=int(_num(rf"Parts per episode:\*\*\s*{_NUM}", body, "Episode composition")),
        rejects_per_episode=int(_num(rf"Rejects per episode:\*\*\s*{_NUM}", body, "Episode composition")),
        timeout=_num(rf"Timeout: {_NUM} s", success_body, "Success criteria"),
    )


def _parse_xyz(row: str, ctx: str) -> tuple[float, float, float]:
    match = re.search(rf"\({_NUM}, {_NUM}, {_NUM}\)", row)
    if not match:
        raise SpecError(f"SPEC.md: no (x, y, z) triple in {ctx}: {row!r}")
    return tuple(_mm(float(match.group(i))) for i in (1, 2, 3))


def _parse_scene(body: str) -> SceneSpec:
    ctx = "Scene layout"
    bin_row = _grab(r"\|\s*(Bin\s*\|[^\n]*)", body, ctx)
    good_row = _grab(r"\|\s*(Good shelf\s*\|[^\n]*)", body, ctx)
    reject_row = _grab(r"\|\s*(Reject shelf\s*\|[^\n]*)", body, ctx)
    table_row = _grab(r"\|\s*(Table\s*\|[^\n]*)", body, ctx)

    def shelf(row: str) -> BoxPad:
        return BoxPad(
            center=_parse_xyz(row, ctx),
            size=(
                _mm(_num(rf"{_NUM} x \d+(?:\.\d+)? flat pad", row, ctx)),
                _mm(_num(rf"\d+(?:\.\d+)? x {_NUM} flat pad", row, ctx)),
            ),
            pad_thickness=_mm(_num(rf"{_NUM} pad thickness", row, ctx)),
            lip_height=_mm(_num(rf"{_NUM} lip", row, ctx)),
            lip_thickness=_mm(_num(rf"{_NUM} lip thickness", row, ctx)),
        )

    return SceneSpec(
        bin_center=_parse_xyz(bin_row, ctx),
        bin_interior=(
            _mm(_num(rf"{_NUM} x \d+(?:\.\d+)? interior", bin_row, ctx)),
            _mm(_num(rf"\d+(?:\.\d+)? x {_NUM} interior", bin_row, ctx)),
        ),
        bin_wall_height=_mm(_num(rf"{_NUM} walls", bin_row, ctx)),
        bin_thickness=_mm(_num(rf"{_NUM} wall/floor thickness", bin_row, ctx)),
        good_shelf=shelf(good_row),
        reject_shelf=shelf(reject_row),
        table_size=(
            _mm(_num(rf"{_NUM} x \d+(?:\.\d+)? surface", table_row, ctx)),
            _mm(_num(rf"\d+(?:\.\d+)? x {_NUM} surface", table_row, ctx)),
        ),
    )


def load_spec(path: Path | None = None) -> SortingSpec:
    if path is None:
        path = repo_root() / "SPEC.md"
    text = _clean(path.read_text(encoding="utf-8"))
    sections = _sections(text)

    inventory = _section(sections, "Part inventory")
    colours = _section(sections, "Good vs reject")
    episode_body = _section(sections, "Episode composition")
    scene_body = _section(sections, "Scene layout")
    gripper_body = _section(sections, "Gripper")
    assist_body = _section(sections, "Grasp assist")
    success_body = _section(sections, "Success criteria")
    teleop_body = _section(sections, "Teleop tuning")
    recording_body = _section(sections, "Recording and learning")

    home_row = _grab(r"(Home pose:\*\*[^\n]*)", scene_body, "Home pose")

    return SortingSpec(
        parts=_parse_parts(inventory),
        material=_parse_material(inventory, colours),
        episode=_parse_episode(episode_body, success_body),
        scene=_parse_scene(scene_body),
        home=HomeSpec(position=_parse_xyz(home_row, "Home pose")),
        gripper=GripperSpec(
            stroke=_mm(_num(rf"\| Jaw stroke \| {_NUM} mm", gripper_body, "Gripper")),
            max_force=_num(rf"\| Max grip force \| {_NUM} N", gripper_body, "Gripper"),
            close_speed=_mm(_num(rf"\| Close speed \| {_NUM} mm/s", gripper_body, "Gripper")),
        ),
        assist=GraspAssistSpec(
            enabled=_grab(r"\| Enabled \| (yes|no) \|", assist_body, "Grasp assist") == "yes",
            radius=_mm(_num(rf"\| Attraction radius \| {_NUM} mm", assist_body, "Grasp assist")),
            force=_num(rf"\| Attraction force \| {_NUM} N", assist_body, "Grasp assist"),
            lock_distance=_mm(_num(rf"\| Lock distance \| {_NUM} mm", assist_body, "Grasp assist")),
            release_margin=_mm(_num(rf"\| Release margin \| {_NUM} mm", assist_body, "Grasp assist")),
        ),
        success=SuccessSpec(
            settle_velocity=_mm(_num(rf"linear velocity under {_NUM} mm/s", success_body, "Success criteria")),
            settle_time=_num(rf"mm/s for {_NUM} s", success_body, "Success criteria"),
        ),
        teleop=TeleopSpec(
            publish_rate=_num(rf"\| Pose publish rate \| {_NUM} Hz", teleop_body, "Teleop tuning"),
            ema_alpha=_num(rf"alpha = {_NUM}", _clean(teleop_body).replace("α", "alpha"), "Teleop tuning"),
            pinch_closed=_num(rf"\| Pinch closed threshold \| {_NUM}", teleop_body, "Teleop tuning"),
            pinch_open=_num(rf"\| Pinch open threshold \| {_NUM}", teleop_body, "Teleop tuning"),
        ),
        recording=RecordingSpec(
            log_rate=_num(rf"\| Demonstration log rate \| {_NUM} Hz", recording_body, "Recording"),
            target_demo_count=int(_num(rf"\| Target demonstration count \| {_NUM}", recording_body, "Recording")),
            train_fraction=_num(rf"\| Train/validation split \| {_NUM}", recording_body, "Recording") / 100.0,
        ),
    )
