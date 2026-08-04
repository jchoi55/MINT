"""Materials, volumes, and the geometry the rates are computed on.

Ray-tracing bugs are quiet: a chord that comes out too short just lowers a rate,
it does not raise. These tests pin the geometry against cases where the answer
is known analytically.
"""

import numpy as np
import pytest

import mint
from mint import detector_tools as dt
from mint import const


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

def test_material_number_densities_are_consistent():
    """N is the NUCLEON density: rho NA A / am. Confusing it with the nuclei
    density (a factor A smaller) is an easy and expensive mistake."""
    m = dt.standard_rock
    assert m.N == pytest.approx(m.density * const.NAvo * m.A / m.am, rel=1e-9)
    assert m.N > 0 and m.density > 0
    assert m.e == pytest.approx(m.N * m.Z / m.A, rel=1e-6)


@pytest.mark.parametrize("name", ["standard_rock", "GAr_5bar", "GAr_10bar", "W", "Air"])
def test_shipped_materials_are_physical(name):
    m = getattr(dt, name)
    assert m.density > 0
    assert m.Z > 0 and m.A > 0
    assert m.Z <= m.A, "more protons than nucleons"


def test_denser_gas_has_proportionally_more_nuclei():
    """5 bar and 10 bar argon differ only by density."""
    r = dt.GAr_10bar.N / dt.GAr_5bar.N
    assert r == pytest.approx(dt.GAr_10bar.density / dt.GAr_5bar.density, rel=1e-9)


def test_scaled_material_preserves_composition():
    base = dt.GAr_5bar
    twice = base.scaled(2.0)
    assert twice.density == pytest.approx(2 * base.density)
    assert twice.Z == base.Z and twice.A == base.A


def test_composite_dominant_nucleus_is_a_real_nucleus():
    """A composite's own Z/A is a weighted average and not a nucleus at all;
    dominant_nucleus must return an actual constituent. Getting this wrong
    inflated a nuclei density by 21x once."""
    rock = dt.standard_rock
    nuc = dt.dominant_nucleus(rock)
    assert nuc.A >= 1 and nuc.Z >= 1
    assert dt.nuclei_density(rock) == pytest.approx(rock.N / nuc.A, rel=1e-9)
    assert dt.nuclei_density(rock) < rock.N, "nuclei are fewer than nucleons"


# ---------------------------------------------------------------------------
# Volumes: chords against analytic answers
# ---------------------------------------------------------------------------

def _axial_ray(n):
    """n rays travelling along +z from far upstream, on the axis."""
    origin = np.zeros((3, n))
    origin[2] = -1e6
    direction = np.zeros((3, n))
    direction[2] = 1.0
    return origin, direction


def test_cylinder_axial_chord_is_its_length():
    L, R = 500.0, 100.0
    vol = dt.CylinderVolume(dt.GAr_5bar, radius=R, length=L, center=(0, 0, 1e4))
    o, d = _axial_ray(1)
    _, chord = vol.intersect(o, d)
    assert chord[0] == pytest.approx(L, rel=1e-9)


def test_ray_outside_the_radius_misses_entirely():
    vol = dt.CylinderVolume(dt.GAr_5bar, radius=10.0, length=500.0, center=(0, 0, 1e4))
    o, d = _axial_ray(1)
    o[0] = 50.0                                   # offset beyond the radius
    _, chord = vol.intersect(o, d)
    assert chord[0] == 0.0


def test_tube_axial_ray_passes_through_the_bore():
    """On-axis rays see no material: that is the point of a beam pipe."""
    tube = dt.TubeVolume(dt.W, r_in=2.0, r_out=4.0, length=100.0, center=(0, 0, 50.0))
    o, d = _axial_ray(1)
    _, chord = tube.intersect(o, d)
    assert chord[0] == 0.0


def test_tube_wall_ray_sees_the_wall():
    tube = dt.TubeVolume(dt.W, r_in=2.0, r_out=4.0, length=100.0, center=(0, 0, 50.0))
    o, d = _axial_ray(1)
    o[0] = 3.0                                    # between r_in and r_out
    _, chord = tube.intersect(o, d)
    assert chord[0] == pytest.approx(100.0, rel=1e-9)


def test_cone_axial_chord_spans_z0_to_z1():
    cone = dt.ConeVolume(dt.GAr_5bar, z0=1e4, z1=1e4 + 300.0, r0=50.0, r1=80.0)
    o, d = _axial_ray(1)
    _, chord = cone.intersect(o, d)
    assert chord[0] == pytest.approx(300.0, rel=1e-6)


def test_cone_widens_downstream():
    """A ray outside r0 but inside r1 must enter partway along, not at z0."""
    cone = dt.ConeVolume(dt.GAr_5bar, z0=0.0, z1=1000.0, r0=10.0, r1=110.0)
    o, d = _axial_ray(1)
    o[0] = 60.0
    _, chord = cone.intersect(o, d)
    assert 0.0 < chord[0] < 1000.0


def test_chords_scale_with_geometry_not_material():
    """Chord is pure geometry: swapping the material must not change it."""
    a = dt.CylinderVolume(dt.GAr_5bar, radius=100.0, length=400.0, center=(0, 0, 1e4))
    b = dt.CylinderVolume(dt.W, radius=100.0, length=400.0, center=(0, 0, 1e4))
    o, d = _axial_ray(1)
    assert a.intersect(o, d)[1][0] == pytest.approx(b.intersect(o, d)[1][0])


# ---------------------------------------------------------------------------
# The benchmark detector
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def det():
    return mint.detectors.benchmark


def test_modules_do_not_overlap(det):
    """Overlapping volumes would double-count material."""
    mods = sorted(det.modules, key=lambda m: m.z0)
    for a, b in zip(mods, mods[1:]):
        assert a.z1 <= b.z0 + 1e-6, f"{a.name} overlaps {b.name}"


def test_every_module_has_positive_thickness(det):
    for m in det.modules:
        assert m.thickness > 0, m.name


def test_signal_volume_is_a_subset_of_the_detector(det):
    sig = {v.name for v in det.signal_volumes()}
    allv = {v.name for v in det.volumes(kinds=None)}
    assert sig and sig <= allv


def test_column_table_totals_match_the_columns(det):
    """The printed budget and the computed column must agree."""
    tot = det.column(kinds=None)
    parts = sum(det.column(kinds=(k,)) for k in
                {m.kind for m in det.modules})
    assert parts == pytest.approx(tot, rel=1e-9)


def test_aperture_is_monotonic(det):
    z = np.linspace(0, det.length, 50)
    r = det.aperture(z)
    assert np.all(np.diff(r) >= 0)


def test_radiation_and_interaction_lengths_are_positive(det):
    assert det.radiation_lengths() > 0
    assert det.interaction_lengths() > 0


def test_rock_column_vanishes_when_detector_sits_at_the_rock_start():
    """No rock upstream if the detector is at (or inside) where rock begins."""
    near = mint.detectors.load("benchmark_5km", dist=mint.detectors.benchmark.rock_start)
    assert near.rock_column == pytest.approx(0.0, abs=1e-6)


def test_rock_column_grows_with_distance():
    a = mint.detectors.load("benchmark_5km", dist=1e5)
    b = mint.detectors.load("benchmark_5km", dist=5e5)
    assert b.rock_column > a.rock_column > 0


def test_volume_stack_is_built_from_the_modules(det):
    stack = det.detector()
    assert isinstance(stack, dt.VolumeStack)
    assert len(stack.volumes) == len(det.volumes(kinds=None))


# ---------------------------------------------------------------------------
# Beamline
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def beamline():
    ring = mint.lattices.load("mc_10tev_hybrid_v06")
    return mint.beamline.Beamline(mint.detectors.benchmark, lattice=ring)


def test_shield_segments_tile_the_straight(beamline):
    """The shield is a chain of tube segments with no gap or overlap."""
    edges = np.asarray(beamline.shield.edges)
    assert edges.size > 1
    assert np.all(np.diff(edges) > 0), "segment edges are not increasing"


def test_shield_volumes_match_the_segment_count(beamline):
    vols = beamline.shield.volumes(sign=+1)
    assert len(vols) > 1
    for v in vols:
        assert v.r_out > v.r_in > 0


def test_shield_bore_never_closes(beamline):
    """The bore is cut to the collimated beam and floored, so it stays open."""
    r_in = np.atleast_1d(beamline.shield.r_in)
    assert np.all(r_in >= beamline.shield.min_aperture - 1e-9)
    assert np.all(r_in > 0)


def test_density_map_is_non_negative_and_finite(beamline):
    z = np.linspace(0.0, 5e5, 60)
    r = np.linspace(0.0, 300.0, 20)
    rho = beamline.density_map(z, r)
    assert rho.shape == (len(z), len(r)) or rho.shape == (len(r), len(z))
    assert np.all(np.isfinite(rho)) and np.all(rho >= 0)


def test_rock_is_denser_than_the_tunnel_air(beamline):
    z = np.array([3e5])            # inside the rock
    r = np.array([500.0])          # off-axis, away from the bore
    rho_rock = float(np.max(beamline.density_map(z, r)))
    z_air = np.array([1e4])        # inside the straight section
    rho_air = float(np.min(beamline.density_map(z_air, r)))
    assert rho_rock > rho_air
