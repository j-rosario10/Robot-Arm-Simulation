"""Builds the MuJoCo world: arm + gripper, table, coloured objects and a bin.

The UR5e and the Robotiq gripper come from MuJoCo Menagerie as two separate
models. Here they are joined together in code, and everything else in the
scene is added as simple shapes.

Directions used everywhere in this project (looking from behind the robot):
    x = forward, away from the robot      y = left      z = up
The table top is at z = 0.
"""

from dataclasses import dataclass
from pathlib import Path

import mujoco

MODELS = Path(__file__).resolve().parents[1] / "models"

# Joint angles (radians) that put the arm above the table, gripper facing down.
# The first one (3.1416 = half a turn) faces the arm forward along +x.
HOME_QPOS = [3.1416, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]

ARM_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
TCP_SITE = "grip_pinch"       # the point between the fingertips that we steer
ARM_STIFFNESS_SCALE = 2.0     # motor spring strength, relative to the stock model
ARM_SETTLE_TIME = 0.06        # seconds; smaller = arm follows faster (too small = wobble)
GRIPPER_BODY = "grip_base"    # objects get welded to this when grabbed
GRIPPER_ACTUATOR = "grip_fingers_actuator"   # 0 = open, 255 = closed


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    shape: str      # "box" or "cylinder"
    rgba: tuple
    start: tuple    # where it sits after a reset (x, y)


OBJECTS = [
    ObjectSpec("red_cube", "box", (0.85, 0.20, 0.20, 1), (0.45, -0.10)),
    ObjectSpec("blue_cube", "box", (0.20, 0.40, 0.85, 1), (0.60, -0.25)),
    ObjectSpec("green_cylinder", "cylinder", (0.20, 0.70, 0.30, 1), (0.60, 0.00)),
    ObjectSpec("yellow_cylinder", "cylinder", (0.95, 0.75, 0.15, 1), (0.45, -0.30)),
]
OBJECT_HALF_HEIGHT = 0.025    # all objects are 5 cm tall
OBJECT_RADIUS = 0.02          # cubes are 4 cm wide, cylinders 4 cm across

# The bin: an open box on the left of the table.
BIN_CENTRE = (0.50, 0.30)
BIN_HALF_SIZE = 0.09          # inside is 18 cm x 18 cm
BIN_HEIGHT = 0.07
BIN_WALL = 0.006


def build_model() -> mujoco.MjModel:
    """Assemble the whole scene and return a ready-to-simulate model."""
    spec = mujoco.MjSpec.from_file(str(MODELS / "universal_robots_ur5e" / "ur5e.xml"))
    gripper = mujoco.MjSpec.from_file(str(MODELS / "robotiq_2f85" / "2f85.xml"))

    # Bolt the gripper onto the arm's tool flange. Every gripper name gets a
    # "grip_" prefix so nothing clashes with the arm's names.
    flange = next(s for s in spec.sites if s.name == "attachment_site")
    spec.attach(gripper, prefix="grip_", site=flange)

    # Cancel gravity on the robot's own parts. Without this the motors let the
    # arm droop about 1 cm below where it was told to go.
    for body in spec.bodies:
        if body.name != "world":
            body.gravcomp = 1.0

    world = spec.worldbody
    box, cylinder = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER

    # Lighting and a bigger off-screen buffer for sharp screenshots.
    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 960
    world.add_light(pos=[0.4, 0.0, 1.8], dir=[0, 0, -1], diffuse=[0.5, 0.5, 0.5])
    # a soft blue-grey backdrop instead of plain black
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.55, 0.65, 0.78], rgb2=[0.16, 0.18, 0.22], width=256, height=1536)

    # Table: a slab whose top surface is exactly z = 0, and a floor below it.
    world.add_geom(name="table", type=box, size=[0.50, 0.60, 0.02],
                   pos=[0.40, 0.0, -0.02], rgba=[0.78, 0.74, 0.68, 1])
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1],
                   pos=[0, 0, -0.6], rgba=[0.25, 0.27, 0.30, 1])

    # Bin: a floor and four thin walls (static, so they never move).
    bx, by = BIN_CENTRE
    h, s, w = BIN_HEIGHT / 2, BIN_HALF_SIZE, BIN_WALL
    bin_rgba = [0.30, 0.33, 0.38, 1]
    world.add_geom(name="bin_floor", type=box, size=[s, s, 0.003], pos=[bx, by, 0.003], rgba=bin_rgba)
    for i, (dx, dy, sx, sy) in enumerate([(s, 0, w, s + w), (-s, 0, w, s + w),
                                          (0, s, s + w, w), (0, -s, s + w, w)]):
        world.add_geom(name=f"bin_wall{i}", type=box, size=[sx, sy, h],
                       pos=[bx + dx, by + dy, h], rgba=bin_rgba)

    # Objects: each is a free body (it can move any way) with one simple shape.
    for obj in OBJECTS:
        body = world.add_body(name=obj.name, pos=[obj.start[0], obj.start[1], OBJECT_HALF_HEIGHT])
        body.add_freejoint(name=f"{obj.name}_joint")
        if obj.shape == "box":
            size = [OBJECT_RADIUS, OBJECT_RADIUS, OBJECT_HALF_HEIGHT]
            body.add_geom(type=box, size=size, rgba=list(obj.rgba), mass=0.05)
        else:
            size = [OBJECT_RADIUS, OBJECT_HALF_HEIGHT, 0]
            body.add_geom(type=cylinder, size=size, rgba=list(obj.rgba), mass=0.05)

        # The "sticky grasp": a weld between the gripper and this object that
        # starts switched off. grasp.py switches it on while the object is held.
        spec.add_equality(name=f"hold_{obj.name}", type=mujoco.mjtEq.mjEQ_WELD,
                          objtype=mujoco.mjtObj.mjOBJ_BODY,
                          name1=GRIPPER_BODY, name2=obj.name, active=False)

    model = spec.compile()

    # Make the arm's motors snappier. Each motor is a spring pulling the joint
    # to its target, plus a brake. The stock brake is strong, so the arm takes
    # about half a second to catch up. A lighter brake cuts that to roughly
    # 0.15 s, still without overshooting.
    for i in range(len(ARM_JOINTS)):
        stiffness = model.actuator_gainprm[i, 0] * ARM_STIFFNESS_SCALE
        model.actuator_gainprm[i, 0] = stiffness
        model.actuator_biasprm[i, 1] = -stiffness
        model.actuator_biasprm[i, 2] = -stiffness * ARM_SETTLE_TIME
    return model


def reset(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Put the arm at home and every object back where it started."""
    mujoco.mj_resetData(model, data)
    for name, angle in zip(ARM_JOINTS, HOME_QPOS):
        data.joint(name).qpos[0] = angle
    data.ctrl[:6] = HOME_QPOS            # motors hold the home pose
    data.ctrl[model.actuator(GRIPPER_ACTUATOR).id] = 0   # gripper open
    for obj in OBJECTS:
        data.joint(f"{obj.name}_joint").qpos[:] = [obj.start[0], obj.start[1],
                                                   OBJECT_HALF_HEIGHT, 1, 0, 0, 0]
    data.eq_active[:] = model.eq_active0  # all sticky welds off
    mujoco.mj_forward(model, data)
