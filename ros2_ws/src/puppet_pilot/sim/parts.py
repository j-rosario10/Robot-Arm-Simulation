"""Procedural part generation: geometry, mass and inertia from SPEC.md.

Geometry strategy (per the project brief):
  - bushing / ring / dowel: PyBullet cylinder *primitives* for collision.
    Primitives are convex by definition, and nothing in this task ever passes
    through a bore, so modelling the hole in the collision shape would cost
    contact-solver time for zero physical benefit. The bore is shown in the
    visual mesh only (a cheap trimesh annulus).
  - hex bolt: the one part needing real geometry, because the jaws grip
    across the head flats. The head is a convex hull of 12 vertices (a hex
    prism, written out as a small OBJ at build time); the shank is a cylinder
    primitive; the two form a compound body. A single convex hull of the
    whole bolt would fill the neck between head and shank with phantom
    material and make the part rest on a slope.

Inertia strategy: computed analytically from the nominal geometry and the
steel density in SPEC.md, then written into PyBullet with changeDynamics.
PyBullet's own defaults (derived from the *collision* shape) would be wrong
twice over - the collision cylinder is solid where the real bushing is
hollow, and default density is not steel. Wrong inertia shows up as parts
that jitter, creep, or refuse to settle, which is miserable to trace back.
The closed forms used here are cross-checked against exact polyhedral mesh
integration (trimesh) in tests/test_parts.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pybullet as p

from puppet_pilot.spec import MaterialSpec, PartDims

PART_KINDS = ["hex_bolt", "bushing", "ring", "dowel"]

# Extra clearance above the floor at spawn so parts start unpenetrated and
# drop the last hair under gravity (algorithmic, not a task parameter).
_SPAWN_CLEARANCE = 0.001


# --------------------------------------------------------------------------
# analytic inertia (all about the centre of mass, axes aligned with symmetry)
# --------------------------------------------------------------------------

def tube_inertia(mass: float, r_outer: float, r_inner: float, height: float) -> tuple[float, float, float]:
    """Hollow cylinder (tube), axis along z.

    Izz = m/2 (ro^2 + ri^2);  Ixx = Iyy = m/12 (3(ro^2 + ri^2) + h^2).
    Setting ri = 0 gives the solid cylinder.
    """
    radial = r_outer**2 + r_inner**2
    izz = 0.5 * mass * radial
    ixx = mass * (3.0 * radial + height**2) / 12.0
    return (ixx, ixx, izz)


def hex_prism_inertia(mass: float, across_flats: float, height: float) -> tuple[float, float, float]:
    """Regular hexagonal prism, axis along z, across-flats width f.

    Derivation (recorded in docs/phase3-simulation.md): decompose the hexagon
    into six equilateral triangles of side s = f/sqrt(3); summing triangle
    polar moments with the parallel-axis theorem gives the area polar moment
    J = (5*sqrt(3)/8) s^4, and with area A = (3*sqrt(3)/2) s^2:

        Izz = m * J/A            = (5/12) m s^2 = (5/36) m f^2
        Ixx_lamina = Izz/2 (symmetry: in-plane moments are isotropic)
        Ixx = m ((5/72) f^2 + h^2/12)   (prism height term as for any prism)
    """
    f2 = across_flats**2
    izz = (5.0 / 36.0) * mass * f2
    ixx = mass * ((5.0 / 72.0) * f2 + height**2 / 12.0)
    return (ixx, ixx, izz)


def hexagon_vertices(across_flats: float, height: float) -> np.ndarray:
    """12 vertices of a hex prism centred at the origin, axis z.

    Vertices sit at the circumradius R = f/sqrt(3), angled so that the flat
    faces are normal to the x-axis (and at +-60 deg from it) - "across flats
    along x". apothem check: R cos(30 deg) = f/2.
    """
    circumradius = across_flats / math.sqrt(3.0)
    verts = []
    for k in range(6):
        angle = math.radians(30.0 + 60.0 * k)
        x, y = circumradius * math.cos(angle), circumradius * math.sin(angle)
        verts.append([x, y, -height / 2.0])
        verts.append([x, y, height / 2.0])
    return np.array(verts)


# --------------------------------------------------------------------------
# part definitions
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PartDefinition:
    """Everything needed to instantiate one part kind in the simulator."""

    kind: str
    mass: float                       # kg
    inertia_diag: tuple               # about COM, symmetry axes
    rest_height: float                # COM height when resting on a flat floor
    max_half_extent: float            # largest horizontal half-size (spawn spacing)
    grasp_width: float                # jaw opening at which the pads meet the part


def part_definition(kind: str, dims: PartDims, material: MaterialSpec) -> PartDefinition:
    rho = material.density
    if kind in ("bushing", "ring"):
        r_o, r_i, h = dims.outer_diameter / 2, dims.inner_diameter / 2, dims.height
        mass = rho * math.pi * (r_o**2 - r_i**2) * h
        return PartDefinition(kind, mass, tube_inertia(mass, r_o, r_i, h),
                              rest_height=h / 2, max_half_extent=r_o,
                              grasp_width=dims.outer_diameter)
    if kind == "dowel":
        r, length = dims.diameter / 2, dims.length
        mass = rho * math.pi * r**2 * length
        # Rests lying on its side (the only stable pose): COM at one radius.
        return PartDefinition(kind, mass, tube_inertia(mass, r, 0.0, length),
                              rest_height=r, max_half_extent=length / 2,
                              grasp_width=dims.diameter)
    if kind == "hex_bolt":
        return _bolt_definition(dims, rho)
    raise ValueError(f"unknown part kind {kind!r}")


def _bolt_definition(dims: PartDims, rho: float) -> PartDefinition:
    """Compound bolt: hex head + cylindrical shank, COM and inertia composed
    with the parallel-axis theorem along the common (z) axis.

    Build frame: bolt axis = z, shank tip at z=0, head top at z=overall.
    """
    f, head_h = dims.across_flats, dims.head_height
    shank_r = dims.shank_diameter / 2
    shank_len = dims.overall_length - head_h

    hex_area = (math.sqrt(3.0) / 2.0) * f**2
    m_head = rho * hex_area * head_h
    m_shank = rho * math.pi * shank_r**2 * shank_len
    mass = m_head + m_shank

    z_shank = shank_len / 2.0
    z_head = shank_len + head_h / 2.0
    z_com = (m_shank * z_shank + m_head * z_head) / mass

    ixx_h, _, izz_h = hex_prism_inertia(m_head, f, head_h)
    ixx_s, _, izz_s = tube_inertia(m_shank, shank_r, 0.0, shank_len)
    izz = izz_h + izz_s
    ixx = (ixx_h + m_head * (z_head - z_com)**2) + (ixx_s + m_shank * (z_shank - z_com)**2)

    return PartDefinition(
        "hex_bolt", mass, (ixx, ixx, izz),
        # Rests lying on a head flat, axis horizontal: COM sits at half the
        # across-flats height (the settle phase finds the exact equilibrium).
        rest_height=f / 2,
        max_half_extent=dims.overall_length / 2,
        grasp_width=f,  # SPEC: grasp the head across the flats
    )


# --------------------------------------------------------------------------
# instantiation in PyBullet
# --------------------------------------------------------------------------

def hex_head_mesh_path(dims: PartDims) -> Path:
    """Path to the bolt-head OBJ, generated on first use.

    PyBullet's compound-shape constructors (createCollisionShapeArray /
    createVisualShapeArray) take mesh *files*, not inline vertex arrays, so
    the head hull is written out as a small OBJ. The filename encodes the
    dimensions in tenths of a millimetre, so changing SPEC.md's across-flats
    or head height regenerates the mesh instead of silently reusing a stale
    one.
    """
    import trimesh

    generated = Path(__file__).resolve().parents[1] / "generated" / "parts"
    generated.mkdir(parents=True, exist_ok=True)
    name = f"hex_head_{dims.across_flats * 10000:.0f}_{dims.head_height * 10000:.0f}.obj"
    path = generated / name
    if not path.exists():
        hull = trimesh.convex.convex_hull(hexagon_vertices(dims.across_flats, dims.head_height))
        hull.export(path)
    return path


def _annulus_visual(dims: PartDims, rgba) -> int:
    """Visual-only mesh with the real bore (collision stays a solid cylinder)."""
    import trimesh

    mesh = trimesh.creation.annulus(
        r_min=dims.inner_diameter / 2, r_max=dims.outer_diameter / 2, height=dims.height
    )
    return p.createVisualShape(
        p.GEOM_MESH,
        vertices=mesh.vertices.tolist(),
        indices=mesh.faces.flatten().tolist(),
        rgbaColor=rgba,
    )


def spawn_part(
    kind: str,
    dims: PartDims,
    material: MaterialSpec,
    is_reject: bool,
    x: float,
    y: float,
    yaw: float,
    floor_z: float,
) -> int:
    """Create one part resting-pose-ready at (x, y) on a floor at floor_z.

    Returns the PyBullet body id. Mass/inertia are set explicitly; friction
    comes from SPEC. Cylindrical parts (dowel, bolt) spawn with their axis
    horizontal, rotated by yaw about world z.
    """
    definition = part_definition(kind, dims, material)
    rgba = list(material.reject_rgb if is_reject else material.good_rgb) + [1.0]

    if kind in ("bushing", "ring"):
        r_o = dims.outer_diameter / 2
        collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=r_o, height=dims.height)
        visual = _annulus_visual(dims, rgba)
        orientation = p.getQuaternionFromEuler([0.0, 0.0, yaw])
        body = p.createMultiBody(
            baseMass=definition.mass,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[x, y, floor_z + definition.rest_height + _SPAWN_CLEARANCE],
            baseOrientation=orientation,
        )
    elif kind == "dowel":
        r = dims.diameter / 2
        collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=dims.length)
        visual = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=dims.length, rgbaColor=rgba)
        # Cylinder axis is z; pitch it flat, then yaw.
        orientation = p.getQuaternionFromEuler([0.0, math.pi / 2, yaw])
        body = p.createMultiBody(
            baseMass=definition.mass,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[x, y, floor_z + definition.rest_height + _SPAWN_CLEARANCE],
            baseOrientation=orientation,
        )
    elif kind == "hex_bolt":
        body = _spawn_bolt(dims, definition, rgba, x, y, yaw, floor_z)
    else:
        raise ValueError(f"unknown part kind {kind!r}")

    p.changeDynamics(
        body,
        -1,
        mass=definition.mass,
        localInertiaDiagonal=definition.inertia_diag,
        lateralFriction=material.part_friction,
        # Rolling and spinning friction matter for the cylindrical parts: with
        # both at PyBullet's default of zero, a dowel pin that starts rolling
        # never stops, so no episode ever settles.
        rollingFriction=material.rolling_friction,
        spinningFriction=material.spinning_friction,
    )
    return body


def _spawn_bolt(dims, definition, rgba, x, y, yaw, floor_z) -> int:
    f, head_h = dims.across_flats, dims.head_height
    shank_r = dims.shank_diameter / 2
    shank_len = dims.overall_length - head_h

    # Shape frames relative to the COM (so the base frame IS the COM frame,
    # matching the inertia tensor we computed).
    hex_area = (math.sqrt(3.0) / 2.0) * f**2
    m_head = hex_area * head_h
    m_shank = math.pi * shank_r**2 * shank_len   # density cancels in the ratio
    z_com = (m_shank * (shank_len / 2) + m_head * (shank_len + head_h / 2)) / (m_head + m_shank)
    head_offset = [0.0, 0.0, (shank_len + head_h / 2) - z_com]
    shank_offset = [0.0, 0.0, (shank_len / 2) - z_com]

    head_mesh = str(hex_head_mesh_path(dims))
    unit_scale = [1.0, 1.0, 1.0]

    collision = p.createCollisionShapeArray(
        shapeTypes=[p.GEOM_MESH, p.GEOM_CYLINDER],
        fileNames=[head_mesh, ""],
        meshScales=[unit_scale, unit_scale],
        radii=[0.0, shank_r],
        lengths=[0.0, shank_len],
        collisionFramePositions=[head_offset, shank_offset],
    )
    visual = p.createVisualShapeArray(
        shapeTypes=[p.GEOM_MESH, p.GEOM_CYLINDER],
        fileNames=[head_mesh, ""],
        meshScales=[unit_scale, unit_scale],
        radii=[0.0, shank_r],
        lengths=[0.0, shank_len],
        rgbaColors=[rgba, rgba],
        visualFramePositions=[head_offset, shank_offset],
    )
    # Lying on its side: bolt axis (z in build frame) pitched horizontal.
    orientation = p.getQuaternionFromEuler([0.0, math.pi / 2, yaw])
    return p.createMultiBody(
        baseMass=definition.mass,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=[x, y, floor_z + definition.rest_height + _SPAWN_CLEARANCE],
        baseOrientation=orientation,
    )
