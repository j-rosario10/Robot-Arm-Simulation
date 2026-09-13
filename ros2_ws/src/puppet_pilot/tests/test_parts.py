"""Part mass and inertia: closed forms checked against mesh integration.

The analytic tensors in sim/parts.py are the ones actually loaded into
PyBullet, so an error there shows up as parts that jitter or creep - a
physics bug that is painful to trace. These tests validate them against
trimesh's exact polyhedral mass-properties integration, which is an entirely
independent computation path.
"""

import math

import numpy as np
import pytest
import trimesh

from puppet_pilot.sim.parts import (
    hex_prism_inertia,
    hexagon_vertices,
    part_definition,
    tube_inertia,
)


def test_hexagon_vertices_have_correct_across_flats():
    across_flats = 0.013
    verts = hexagon_vertices(across_flats, 0.0053)
    # Across flats is measured between opposite faces: the apothem is f/2.
    xy = verts[:, :2]
    circumradius = np.linalg.norm(xy, axis=1)
    assert np.allclose(circumradius, across_flats / math.sqrt(3.0))
    assert xy[:, 0].max() == pytest.approx(across_flats / 2.0)


def test_tube_inertia_matches_mesh_integration():
    """Hollow cylinder closed form vs trimesh on a fine annulus mesh."""
    r_outer, r_inner, height, density = 0.0125, 0.0075, 0.020, 7850.0
    mesh = trimesh.creation.annulus(r_min=r_inner, r_max=r_outer, height=height, sections=512)
    mesh.density = density

    mass = density * math.pi * (r_outer**2 - r_inner**2) * height
    assert mesh.mass == pytest.approx(mass, rel=1e-3)

    ixx, _, izz = tube_inertia(mass, r_outer, r_inner, height)
    measured = np.diag(mesh.moment_inertia)
    assert measured[0] == pytest.approx(ixx, rel=1e-3)
    assert measured[2] == pytest.approx(izz, rel=1e-3)


def test_hex_prism_inertia_matches_mesh_integration():
    """The hand-derived hex prism tensor is the one worth double-checking:
    its Izz = (5/36) m f^2 coefficient is easy to get wrong."""
    across_flats, height, density = 0.013, 0.0053, 7850.0
    mesh = trimesh.convex.convex_hull(hexagon_vertices(across_flats, height))
    mesh.density = density

    area = (math.sqrt(3.0) / 2.0) * across_flats**2
    mass = density * area * height
    assert mesh.mass == pytest.approx(mass, rel=1e-6)

    ixx, _, izz = hex_prism_inertia(mass, across_flats, height)
    measured = np.diag(mesh.moment_inertia)
    assert measured[2] == pytest.approx(izz, rel=1e-6)
    assert measured[0] == pytest.approx(ixx, rel=1e-6)


def test_bolt_compound_matches_mesh_union(spec):
    """The bolt's composed mass/COM/inertia vs a single integrated mesh."""
    dims = spec.parts["hex_bolt"]
    density = spec.material.density
    definition = part_definition("hex_bolt", dims, spec.material)

    shank_length = dims.overall_length - dims.head_height
    head = trimesh.convex.convex_hull(hexagon_vertices(dims.across_flats, dims.head_height))
    head.apply_translation([0, 0, shank_length + dims.head_height / 2])
    shank = trimesh.creation.cylinder(radius=dims.shank_diameter / 2, height=shank_length, sections=256)
    shank.apply_translation([0, 0, shank_length / 2])
    bolt = trimesh.util.concatenate([head, shank])
    bolt.density = density

    assert definition.mass == pytest.approx(bolt.mass, rel=1e-3)
    measured = np.diag(bolt.moment_inertia)  # trimesh reports about the COM
    assert definition.inertia_diag[2] == pytest.approx(measured[2], rel=1e-3)
    assert definition.inertia_diag[0] == pytest.approx(measured[0], rel=1e-3)


def test_part_masses_are_plausible(spec):
    """Sanity: steel parts of this size are grams, not kilograms - and well
    under the 2F-85's 5 kg payload."""
    for kind in ("hex_bolt", "bushing", "ring", "dowel"):
        definition = part_definition(kind, spec.parts[kind], spec.material)
        assert 0.001 < definition.mass < 0.100, f"{kind} mass {definition.mass:.4f} kg"


def test_all_parts_fit_the_jaw_stroke(spec):
    """SPEC claims every part sits comfortably inside the 85 mm stroke -
    verify against the actual grasp dimension of each part."""
    grasp_widths = {
        "hex_bolt": spec.parts["hex_bolt"].across_flats,
        "bushing": spec.parts["bushing"].outer_diameter,
        "ring": spec.parts["ring"].outer_diameter,
        "dowel": spec.parts["dowel"].diameter,
    }
    for kind, width in grasp_widths.items():
        assert width < spec.gripper.stroke, f"{kind} too wide for the jaws"
