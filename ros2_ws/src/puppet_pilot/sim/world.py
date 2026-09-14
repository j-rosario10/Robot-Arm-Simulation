"""The sorting-task simulation world.

Builds, from SPEC.md alone: a table, a parts bin, two shelves, the UR5e with
its 2F-85, and a randomised assortment of parts. Owns the physics stepping
and exposes the state the rest of the project consumes (joint state, gripper
state, object poses, per-part sort status).

Collision-geometry policy: the URDF's ``<collision>`` meshes are the
decimated STLs UR and Robotiq ship, and PyBullet treats a mesh collision
shape on a *dynamic* body as its convex hull unless explicitly told
otherwise. So the arm gets convex hulls for free; the visual .dae meshes are
never used for contact. Parts are primitives (convex by definition) except
the bolt head, which is an explicit convex hull.

Self-collision is deliberately NOT enabled in the simulator: on the 2F-85's
faked closed loop it makes the gripper fight itself, and the project already
has a purpose-built veto - the Phase 2 capsule checker, which runs on the
*command* before it ever reaches the sim. Documented in
docs/phase3-simulation.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

from puppet_pilot.kinematics.ik import solve_ik
from puppet_pilot.kinematics.params import DHTable, JointLimits, JOINT_NAMES
from puppet_pilot.sim import gripper as gripper_module
from puppet_pilot.sim.parts import PART_KINDS, part_definition, spawn_part
from puppet_pilot.spec import SortingSpec, load_spec

ARM_JOINT_NAMES = [f"ur5e_{name}" for name in JOINT_NAMES]

# Phase 2 result: the only frame correction between our DH chain and the
# URDF/world frame is a pi rotation about z.
WORLD_FROM_DH = np.diag([-1.0, -1.0, 1.0, 1.0])

# Tool pointing straight down (tool z-axis along -world z): SPEC's home
# orientation, and the approach orientation for every top-down grasp.
TOOL_POINTING_DOWN = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])

# Elbow-up seed for IK. The UR5e has up to eight IK branches; seeding from a
# consistent elbow-up posture keeps the arm above the table instead of
# solving into a mirrored configuration that reaches under it.
_ELBOW_UP_SEED = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])

# Simulator settings (solver choices, not task parameters).
_SIM_TIMESTEP = 1.0 / 240.0     # PyBullet's default; small enough for stable contacts
_SOLVER_ITERATIONS = 150        # raised from the default 50: many small contacts in the bin
_TABLE_SLAB_THICKNESS = 0.05    # cosmetic only - static body, only its top face at z=0 matters
_SPAWN_SEPARATION_MARGIN = 1.2  # multiplier on part half-extents when rejecting overlaps
_SPAWN_MAX_ATTEMPTS = 500


@dataclass
class PartInstance:
    """One spawned part and its bookkeeping."""

    body: int
    kind: str
    is_reject: bool
    grasp_width: float = 0.0

    def pose(self) -> tuple:
        """(position, quaternion) in world coordinates."""
        return p.getBasePositionAndOrientation(self.body)

    def velocity(self) -> tuple:
        return p.getBaseVelocity(self.body)


@dataclass
class SortingWorld:
    """The assembled scene. Construct with ``SortingWorld.create``."""

    spec: SortingSpec
    client: int
    robot: int
    arm_indices: list
    tool_link: int
    gripper: gripper_module.GripperModel
    table: int
    bin_bodies: list
    good_shelf: int
    reject_shelf: int
    dh: DHTable
    limits: JointLimits
    home_configuration: np.ndarray
    jaw_offset: float           # tool0 -> pad tip along the tool approach axis, jaws open [m]
    pad_descent: float          # extra reach of the pad tip when the jaws close [m]
    pad_span: float             # height of the pad contact face [m]
    gripper_target: float = 0.0 # last commanded jaw opening [m]
    assist: object = None       # GraspAssist, attached after construction
    parts: list = field(default_factory=list)
    seed: int = 0

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        gui: bool = False,
        spec: SortingSpec | None = None,
        urdf_path: Path | None = None,
        seed: int = 0,
    ) -> "SortingWorld":
        spec = spec or load_spec()
        # One world per process. PyBullet's module-level API implicitly targets
        # client 0, so a second connection would silently pile its bodies into
        # the first world - which presents as an unexplained physics explosion
        # rather than an error. Fail loudly instead.
        if p.isConnected():
            raise RuntimeError(
                "a PyBullet client is already connected; SortingWorld is "
                "one-per-process - call disconnect() on the previous world first"
            )
        client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(_SIM_TIMESTEP)
        p.setPhysicsEngineParameter(numSolverIterations=_SOLVER_ITERATIONS)
        if gui:
            # The scene is small and the renderer is software (llvmpipe under
            # WSLg) - drop the debug panels and frame the workspace.
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(1.2, 50, -35, [0.4, 0.0, 0.1])

        robot, arm_indices, tool_link, gripper = _load_robot(spec, urdf_path)
        table = _build_table(spec)
        bin_bodies = _build_bin(spec)
        good_shelf = _build_shelf(spec.scene.good_shelf, spec)
        reject_shelf = _build_shelf(spec.scene.reject_shelf, spec)

        dh = DHTable.from_yaml()
        limits = JointLimits.from_yaml()
        jaw_offset, pad_descent, pad_span, home = _calibrate_jaw_offset(
            spec, dh, limits, robot, tool_link, gripper, arm_indices
        )

        world = cls(
            spec=spec, client=client, robot=robot, arm_indices=arm_indices,
            tool_link=tool_link, gripper=gripper, table=table, bin_bodies=bin_bodies,
            good_shelf=good_shelf, reject_shelf=reject_shelf,
            dh=dh, limits=limits, home_configuration=home,
            jaw_offset=jaw_offset, pad_descent=pad_descent, pad_span=pad_span,
            gripper_target=spec.gripper.stroke, seed=seed,
        )
        world.assist = GraspAssist(world)
        world.reset_episode(seed)
        return world

    def disconnect(self) -> None:
        p.disconnect(self.client)

    # ------------------------------------------------------------------
    # episodes
    # ------------------------------------------------------------------

    def reset_episode(self, seed: int | None = None) -> None:
        """Clear the bin and respawn a fresh randomised assortment.

        The seed is stored so every episode is reproducible and can be logged
        alongside the demonstration (SPEC: "Random seed: configurable, logged
        with every episode").
        """
        if seed is not None:
            self.seed = seed
        if self.assist is not None:
            self.assist.release()
        for part in self.parts:
            p.removeBody(part.body)

        # Park the arm at SPEC's home pose *before* anything else moves. The
        # URDF's zero configuration has the arm stretched out horizontally;
        # left there it sags under gravity into the table and the bin, which
        # sends parts flying. Holding home is both what SPEC asks for and
        # what keeps the scene physically sane.
        self.reset_joints(self.home_configuration)
        self.command_gripper(self.spec.gripper.stroke)

        self.parts = _spawn_episode_parts(self.spec, self.seed)
        self.settle()

    def settle(self, max_seconds: float = 2.0) -> float:
        """Step until every part is at rest, per SPEC's settle criterion.

        Returns the seconds simulated. A part that never settles is a physics
        red flag (usually wrong inertia), so the caller can assert on this.
        """
        threshold = self.spec.success.settle_velocity
        hold_steps = int(self.spec.success.settle_time / _SIM_TIMESTEP)
        quiet = 0
        for step in range(int(max_seconds / _SIM_TIMESTEP)):
            self.step()
            speeds = [np.linalg.norm(part.velocity()[0]) for part in self.parts]
            quiet = quiet + 1 if (not speeds or max(speeds) < threshold) else 0
            if quiet >= hold_steps:
                return (step + 1) * _SIM_TIMESTEP
        return max_seconds

    def step(self, count: int = 1) -> None:
        """Advance the physics, keeping the gripper's mimic joints in sync.

        The mimic sync has to run every step (PyBullet has no <mimic>), so it
        lives here rather than in any caller - there is exactly one place the
        simulation advances.
        """
        for _ in range(count):
            gripper_module.sync_mimic_joints(self.gripper)
            if self.assist is not None:
                self.assist.step()
            p.stepSimulation()

    @property
    def timestep(self) -> float:
        return _SIM_TIMESTEP

    # ------------------------------------------------------------------
    # robot state and commands
    # ------------------------------------------------------------------

    def joint_positions(self) -> np.ndarray:
        return np.array([p.getJointState(self.robot, i)[0] for i in self.arm_indices])

    def joint_velocities(self) -> np.ndarray:
        return np.array([p.getJointState(self.robot, i)[1] for i in self.arm_indices])

    def command_joints(self, positions: np.ndarray) -> None:
        """Position-control the six arm joints.

        Force ceilings are the UR5e's published max joint torques from
        joint_limits.yaml (150 Nm shoulder/elbow, 28 Nm wrists) - per joint,
        not one blanket number.
        """
        p.setJointMotorControlArray(
            self.robot, self.arm_indices, p.POSITION_CONTROL,
            targetPositions=list(np.asarray(positions, dtype=float)),
            forces=list(self.limits.effort),
        )

    def reset_joints(self, positions: np.ndarray) -> None:
        """Teleport the arm (episode setup only - not a control path)."""
        for index, angle in zip(self.arm_indices, np.asarray(positions, dtype=float)):
            p.resetJointState(self.robot, index, angle)
        self.command_joints(positions)

    def command_gripper(self, opening: float) -> None:
        self.gripper_target = float(opening)
        gripper_module.command_opening(self.gripper, opening)

    def jaw_centre(self) -> np.ndarray:
        """World position of the middle of the pad contact faces (jaws open)."""
        tool_position, tool_orientation = self.tool_pose()
        rotation = np.array(p.getMatrixFromQuaternion(tool_orientation)).reshape(3, 3)
        return np.array(tool_position) + rotation[:, 2] * (self.jaw_offset - self.pad_span / 2)

    def gripper_opening(self) -> float:
        return gripper_module.measured_opening(self.gripper)

    def tool_pose(self) -> tuple:
        """(position, quaternion) of the tool0 frame in world coordinates."""
        state = p.getLinkState(self.robot, self.tool_link, computeForwardKinematics=1)
        return state[4], state[5]

    def solve_pose(self, jaw_position, rotation=None, seed=None):
        """IK for a world-frame *pad tip* position, tool pointing down.

        The task layer thinks about where the jaws meet the work, not where
        the flange is; this converts, so no caller has to carry the gripper
        offset around. Returns the Phase 2 IKResult.
        """
        rotation = TOOL_POINTING_DOWN if rotation is None else rotation
        # Back off along the tool's own z-axis to get the flange position.
        flange = np.asarray(jaw_position, dtype=float) - rotation[:, 2] * self.jaw_offset
        world_target = np.eye(4)
        world_target[:3, :3] = rotation
        world_target[:3, 3] = flange
        dh_target = np.linalg.inv(WORLD_FROM_DH) @ world_target
        start = self.joint_positions() if seed is None else seed
        return solve_ik(self.dh, start, dh_target, limits=self.limits, max_iterations=300)

    def grasp_height(self, floor_z: float, clearance: float = 0.004) -> float:
        """Lowest pad-tip height that still clears ``floor_z`` once closed.

        The pads descend by ``pad_descent`` during the closing stroke, so a
        grasp staged at the floor jams the pads into it before they ever
        reach the part. Callers picking off a surface should stage here.
        """
        return floor_z + self.pad_descent + clearance

    def move_to(self, jaw_position, rotation=None, seconds=2.0, seed=None):
        """Solve IK for a pad-tip pose and drive there. Returns the IKResult."""
        result = self.solve_pose(jaw_position, rotation, seed=seed)
        if result.converged:
            self.command_joints(result.q)
            self.step(int(seconds / self.timestep))
        return result

    # ------------------------------------------------------------------
    # task evaluation
    # ------------------------------------------------------------------

    def part_location(self, part: PartInstance) -> str:
        """Where a part currently is: 'bin', 'good', 'reject' or 'loose'."""
        position, _ = part.pose()
        scene = self.spec.scene
        if _within(position, scene.bin_center, scene.bin_interior):
            return "bin"
        if _within(position, scene.good_shelf.center, scene.good_shelf.size):
            return "good"
        if _within(position, scene.reject_shelf.center, scene.reject_shelf.size):
            return "reject"
        return "loose"

    def parts_sorted_correctly(self) -> int:
        """SPEC's partial credit: how many parts are on the right shelf."""
        correct = 0
        for part in self.parts:
            location = self.part_location(part)
            if location in ("good", "reject"):
                wanted = "reject" if part.is_reject else "good"
                if location == wanted and self._is_resting(part):
                    correct += 1
        return correct

    def _is_resting(self, part: PartInstance) -> bool:
        linear, _ = part.velocity()
        return float(np.linalg.norm(linear)) < self.spec.success.settle_velocity


class GraspAssist:
    """SPEC's grasp assist: attraction toward the jaw centre while closing,
    then a rigid hold once the part is between closed pads.

    Why a bounded aid rather than "perfect physics": vision teleop gives the
    operator no force or tactile feedback, so pad-part alignment is done by
    eye at ~5 fps of software rendering. Real teleop rigs compensate with
    compliant fingers and haptics; this is the simulator's stand-in, and it
    is applied identically to human and autonomous runs.
    """

    # Light velocity damping in the attraction zone so a pulled part settles
    # instead of orbiting the jaw centre. Solver constant, not a task value.
    _DAMPING = 0.3  # N s / m

    def __init__(self, world: "SortingWorld"):
        self.world = world
        self.spec = world.spec.assist
        self.constraint: int | None = None
        self.held: PartInstance | None = None

    def release(self) -> None:
        if self.constraint is not None:
            p.removeConstraint(self.constraint)
        self.constraint = None
        self.held = None

    def step(self) -> None:
        if not self.spec.enabled:
            return
        world = self.world
        stroke = world.spec.gripper.stroke

        if self.held is not None:
            # Let go once the operator opens clearly wider than the part.
            if world.gripper_target > self.held.grasp_width + self.spec.release_margin:
                self.release()
            return

        # Only assist when the operator is actually closing on something.
        closing = world.gripper_target < 0.6 * stroke
        if not closing:
            return

        centre = world.jaw_centre()
        measured = world.gripper_opening()
        for part in world.parts:
            position, _ = part.pose()
            offset = centre - np.array(position)
            distance = float(np.linalg.norm(offset))
            if distance > self.spec.radius:
                continue
            if distance < self.spec.lock_distance and measured <= part.grasp_width + 0.006:
                self._lock(part)
                return
            velocity = np.array(part.velocity()[0])
            force = self.spec.force * offset / max(distance, 1e-6) - self._DAMPING * velocity
            p.applyExternalForce(part.body, -1, force.tolist(), position, p.WORLD_FRAME)

    def _lock(self, part: PartInstance) -> None:
        """Weld the part to the tool link at its current relative pose."""
        world = self.world
        tool_position, tool_orientation = world.tool_pose()
        part_position, part_orientation = part.pose()
        inv_position, inv_orientation = p.invertTransform(tool_position, tool_orientation)
        rel_position, rel_orientation = p.multiplyTransforms(
            inv_position, inv_orientation, part_position, part_orientation
        )
        self.constraint = p.createConstraint(
            world.robot, world.tool_link, part.body, -1, p.JOINT_FIXED,
            [0, 0, 0], rel_position, [0, 0, 0], rel_orientation, [0, 0, 0, 1],
        )
        self.held = part


def solve_home_configuration(
    spec: SortingSpec, dh: DHTable, limits: JointLimits, jaw_offset: float
) -> np.ndarray:
    """Joint configuration for SPEC's home pose: retracted above the bin,
    tool pointing straight down.

    SPEC's home position is read as the *end effector* - the jaw centre, not
    the flange - so it means the same thing as every other task-level
    position in the project. Seeded elbow-up so the arm arcs over the table
    rather than solving into a mirrored branch that reaches under it.
    """
    flange = np.asarray(spec.home.position, dtype=float) - TOOL_POINTING_DOWN[:, 2] * jaw_offset
    world_target = np.eye(4)
    world_target[:3, :3] = TOOL_POINTING_DOWN
    world_target[:3, 3] = flange
    dh_target = np.linalg.inv(WORLD_FROM_DH) @ world_target

    result = solve_ik(dh, _ELBOW_UP_SEED, dh_target, limits=limits, max_iterations=400)
    if not result.converged:
        raise RuntimeError(
            f"no IK solution for SPEC's home pose {spec.home.position}: "
            f"{result.position_error:.4f} m residual"
        )
    return result.q


def _calibrate_jaw_offset(spec, dh, limits, robot, tool_link, gripper, arm_indices):
    """Measure tool0 -> pad tip, and solve the home configuration with it.

    The reference point for every task-level position in this project is the
    *tip of the finger pads* - the part of the gripper that has to clear the
    bin floor. Using the finger-tip link origin instead would be wrong by
    54 mm: the pads hang from their link origin and extend 89..152 mm below
    tool0, so a "grasp height" referenced to the origin drives the pads
    straight through the bin floor.

    The measurement needs world-axis-aligned bounding boxes, which are only
    meaningful when the tool already points straight down - hence the two
    steps: solve a provisional home from the link origins, park there, then
    measure properly and re-solve. Everything is read off the loaded model,
    so swapping in the custom SolidWorks gripper later needs no code change.
    """
    tip_links = [
        i for i in range(p.getNumJoints(robot))
        if "finger_tip" in p.getJointInfo(robot, i)[12].decode()
    ]
    if not tip_links:
        raise RuntimeError("no finger-tip links found: cannot locate the jaws")

    def park(configuration, knuckle_angle):
        for index, angle in zip(arm_indices, configuration):
            p.resetJointState(robot, index, angle)
        p.resetJointState(robot, gripper.driven_index, knuckle_angle)
        for index, multiplier in gripper.mimic_indices.items():
            p.resetJointState(robot, index, multiplier * knuckle_angle)
        p.stepSimulation()

    def pad_tip_offset() -> float:
        tool_z = p.getLinkState(robot, tool_link, computeForwardKinematics=1)[4][2]
        return float(tool_z - min(p.getAABB(robot, i)[0][2] for i in tip_links))

    def pad_span() -> float:
        boxes = [p.getAABB(robot, i) for i in tip_links]
        return float(max(b[1][2] for b in boxes) - min(b[0][2] for b in boxes))

    provisional_offset = float(np.linalg.norm(
        np.mean([np.array(p.getLinkState(robot, i, computeForwardKinematics=1)[4])
                 for i in tip_links], axis=0)
        - np.array(p.getLinkState(robot, tool_link, computeForwardKinematics=1)[4])
    ))
    provisional_home = solve_home_configuration(spec, dh, limits, provisional_offset)

    # The pads swing outward *and downward* as the jaws close, so the tip
    # reaches further from the flange when closed than when open. That
    # difference is what stops a grasp from working near a floor: approach at
    # the open offset, and the closing motion drives the pads through it.
    park(provisional_home, gripper.open_angle)
    jaw_offset = pad_tip_offset()
    span = pad_span()
    park(provisional_home, gripper.closed_angle)
    pad_descent = pad_tip_offset() - jaw_offset

    home = solve_home_configuration(spec, dh, limits, jaw_offset)
    park(home, gripper.open_angle)
    return jaw_offset, pad_descent, span, home


# --------------------------------------------------------------------------
# scene construction helpers
# --------------------------------------------------------------------------

def _static_box(half_extents, position, rgba) -> int:
    """A fixed (mass 0) box - table, bin walls, shelf slabs."""
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba)
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )


def _build_table(spec: SortingSpec) -> int:
    width, depth = spec.scene.table_size
    half = _TABLE_SLAB_THICKNESS / 2
    table = _static_box(
        [width / 2, depth / 2, half],
        [0.0, 0.0, -half],          # top face exactly at z = 0
        [0.82, 0.80, 0.76, 1.0],
    )
    p.changeDynamics(table, -1, lateralFriction=spec.material.table_friction)
    return table


def _build_bin(spec: SortingSpec) -> list:
    """Open-topped box: floor slab plus four walls, interior per SPEC."""
    scene = spec.scene
    cx, cy, _ = scene.bin_center
    inner_x, inner_y = scene.bin_interior
    thickness = scene.bin_thickness
    wall_height = scene.bin_wall_height
    rgba = [0.45, 0.47, 0.50, 1.0]

    bodies = [
        _static_box([inner_x / 2 + thickness, inner_y / 2 + thickness, thickness / 2],
                    [cx, cy, thickness / 2], rgba)
    ]
    floor_top = thickness
    wall_centre_z = floor_top + wall_height / 2
    for dx, dy, hx, hy in (
        (inner_x / 2 + thickness / 2, 0.0, thickness / 2, inner_y / 2 + thickness),
        (-inner_x / 2 - thickness / 2, 0.0, thickness / 2, inner_y / 2 + thickness),
        (0.0, inner_y / 2 + thickness / 2, inner_x / 2 + thickness, thickness / 2),
        (0.0, -inner_y / 2 - thickness / 2, inner_x / 2 + thickness, thickness / 2),
    ):
        bodies.append(
            _static_box([hx, hy, wall_height / 2], [cx + dx, cy + dy, wall_centre_z], rgba)
        )

    for body in bodies:
        p.changeDynamics(body, -1, lateralFriction=spec.material.table_friction)
    return bodies


def _build_shelf(shelf, spec: SortingSpec) -> int:
    """Flat pad with a low lip. Returned as the pad body (the lip is welded on
    as extra static boxes sharing the same fate)."""
    cx, cy, _ = shelf.center
    size_x, size_y = shelf.size
    rgba = [0.55, 0.58, 0.62, 1.0]

    pad = _static_box([size_x / 2, size_y / 2, shelf.pad_thickness / 2],
                      [cx, cy, shelf.pad_thickness / 2], rgba)
    p.changeDynamics(pad, -1, lateralFriction=spec.material.table_friction)

    lip_z = shelf.pad_thickness + shelf.lip_height / 2
    for dx, dy, hx, hy in (
        (size_x / 2 - shelf.lip_thickness / 2, 0.0, shelf.lip_thickness / 2, size_y / 2),
        (-size_x / 2 + shelf.lip_thickness / 2, 0.0, shelf.lip_thickness / 2, size_y / 2),
        (0.0, size_y / 2 - shelf.lip_thickness / 2, size_x / 2, shelf.lip_thickness / 2),
        (0.0, -size_y / 2 + shelf.lip_thickness / 2, size_x / 2, shelf.lip_thickness / 2),
    ):
        lip = _static_box([hx, hy, shelf.lip_height / 2], [cx + dx, cy + dy, lip_z], rgba)
        p.changeDynamics(lip, -1, lateralFriction=spec.material.table_friction)
    return pad


def _within(position, center, size) -> bool:
    """Is a point inside a rectangular footprint (xy only)?"""
    return (
        abs(position[0] - center[0]) <= size[0] / 2
        and abs(position[1] - center[1]) <= size[1] / 2
    )


def _load_robot(spec: SortingSpec, urdf_path: Path | None):
    if urdf_path is None:
        urdf_path = Path(__file__).resolve().parents[1] / "generated" / "ur5e_robotiq_pybullet.urdf"
    if not Path(urdf_path).exists():
        raise FileNotFoundError(
            f"{urdf_path} not found - regenerate with tools/make_pybullet_urdf.py"
        )

    robot = p.loadURDF(str(urdf_path), basePosition=[0, 0, 0], useFixedBase=True)

    joint_index, joint_info, link_index = {}, {}, {}
    for i in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, i)
        joint_index[info[1].decode()] = i
        joint_info[info[1].decode()] = info
        link_index[info[12].decode()] = i

    arm_indices = [joint_index[name] for name in ARM_JOINT_NAMES]
    tool_link = link_index["ur5e_tool0"]
    gripper = gripper_module.build_gripper(robot, joint_index, joint_info, spec.gripper)

    # Gripper pad friction from SPEC: applied to the finger-tip links, which
    # are the only gripper geometry that touches a part.
    for name, index in link_index.items():
        if "finger_tip" in name or "finger_link" in name:
            p.changeDynamics(robot, index, lateralFriction=spec.material.pad_friction)

    return robot, arm_indices, tool_link, gripper


def _spawn_episode_parts(spec: SortingSpec, seed: int) -> list:
    """Randomised assortment per SPEC: n parts, k rejects, >=1 of each kind,
    random position and yaw inside the bin, no overlap at spawn."""
    rng = np.random.default_rng(seed)
    episode = spec.episode

    kinds = list(PART_KINDS)
    extra = rng.choice(PART_KINDS, size=episode.parts_per_episode - len(PART_KINDS))
    kinds.extend(extra.tolist())
    rng.shuffle(kinds)

    reject_flags = np.zeros(episode.parts_per_episode, dtype=bool)
    reject_flags[rng.choice(episode.parts_per_episode, episode.rejects_per_episode, replace=False)] = True

    scene = spec.scene
    floor_z = scene.bin_thickness
    inner_x, inner_y = scene.bin_interior

    placed: list = []
    instances = []
    for kind, is_reject in zip(kinds, reject_flags):
        dims = spec.parts[kind]
        definition = part_definition(kind, dims, spec.material)
        radius = definition.max_half_extent

        x, y = _sample_free_position(rng, scene.bin_center, inner_x, inner_y, radius, placed)
        placed.append((x, y, radius))
        body = spawn_part(
            kind, dims, spec.material, bool(is_reject),
            x, y, float(rng.uniform(-math.pi, math.pi)), floor_z,
        )
        instances.append(PartInstance(body=body, kind=kind, is_reject=bool(is_reject),
                                      grasp_width=definition.grasp_width))
    return instances


def _sample_free_position(rng, bin_center, inner_x, inner_y, radius, placed):
    """Rejection-sample a spawn point that keeps parts clear of each other and
    of the bin walls. Overlapping spawns explode on the first step."""
    cx, cy, _ = bin_center
    limit_x = max(inner_x / 2 - radius, 0.0)
    limit_y = max(inner_y / 2 - radius, 0.0)
    for _ in range(_SPAWN_MAX_ATTEMPTS):
        x = cx + rng.uniform(-limit_x, limit_x)
        y = cy + rng.uniform(-limit_y, limit_y)
        if all(
            math.hypot(x - px, y - py) > _SPAWN_SEPARATION_MARGIN * (radius + pr)
            for px, py, pr in placed
        ):
            return x, y
    raise RuntimeError(
        "could not place a part without overlap after "
        f"{_SPAWN_MAX_ATTEMPTS} attempts - bin too small for this assortment?"
    )
