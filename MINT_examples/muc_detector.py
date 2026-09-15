"""The muon-collider detector, as MINT volumes.

A neutrino produced in the straight section of the collider ring passes through
the *detector itself* on its way out. This module describes that detector so
those interactions can be counted -- the "neutrino slice" through the MuC
experiment.

The geometry follows the MAIA/CLICdet-style layout used in the BIN_MC studies:
a tungsten nozzle shielding the interaction point, then ECAL, HCAL, the
solenoid, and the muon spectrometer, all coaxial about the beam with the
interaction point at ``z = 0``.

Notation
--------
Everything here is a **volume**, described by the radii and z-range it
occupies, and rays are traced by intersecting them. That is MINT's model, and
it differs from the surface-and-traversal-graph description this geometry
originally had: there is no component numbering, no ``next_ids``, and no need
to know which surface a ray reaches next. A ray is intersected against every
volume independently, and the chords add up.

Lengths are in cm throughout, matching the rest of MINT.

    from muc_detector import muc_detector, REGIONS

    det = muc_detector()                     # a dt.VolumeStack
    print(f"{len(det.volumes)} volumes")

Adding the beamline shielding
-----------------------------
:func:`with_beamline` returns the detector together with the tungsten shielding
that surrounds the beam pipe along the straight section, so a ray can be traced
from its decay point all the way through the experiment.
"""

import numpy as np

from mint import detector_tools as dt


# ---------------------------------------------------------------------------
# Geometry
#
# Each entry is (name, material, r_in, r_out, z_min, z_max) in cm. Regions are
# mirrored about z = 0 unless z_min is already negative.
# ---------------------------------------------------------------------------

#: Tube-shaped regions: (name, material, r_in, r_out, z_min, z_max) [cm].
REGIONS = [
    # --- beam pipe and the volume the tracker sits in ---------------------
    ("beam_pipe",       dt.Air,             0.0,   2.2,  -563.8, 563.8),
    ("tracker_volume",  dt.Air,             2.2, 150.0,  -221.0, 221.0),

    # --- electromagnetic calorimeter --------------------------------------
    ("ecal_barrel",     dt.ecal_CLICdet,  150.0, 170.2,  -221.0, 221.0),
    ("ecal_endcap_up",  dt.ecal_CLICdet,   33.9, 170.0,  -250.9, -230.7),
    ("ecal_endcap_dn",  dt.ecal_CLICdet,   33.9, 170.0,   230.7, 250.9),

    # --- hadronic calorimeter ---------------------------------------------
    ("hcal_barrel",     dt.hcal_CLICdet,  174.0, 333.0,  -221.0, 221.0),
    # Two slabs, not one: between |z| = 235.4 and 250.9 the ECAL endcap fills
    # r = 33.9-170, so the HCAL there starts at 170. Merging them would overlap
    # the ECAL and double-count the material forward rays cross.
    ("hcal_endcap1_up", dt.hcal_CLICdet,  170.0, 324.6,  -250.9, -235.4),
    ("hcal_endcap1_dn", dt.hcal_CLICdet,  170.0, 324.6,   235.4, 250.9),

    # --- solenoid: two iron flux returns around an aluminium coil ---------
    ("solenoid_inner",  dt.Fe,            348.3, 352.3,  -412.9, 412.9),
    ("solenoid_coil",   dt.Al,            364.9, 399.3,  -412.9, 412.9),
    ("solenoid_outer",  dt.Fe,            425.0, 429.0,  -412.9, 412.9),

    # --- muon spectrometer -------------------------------------------------
    ("muon_barrel",     dt.Fe,            446.1, 645.0,  -417.9, 417.9),
    ("muon_endcap_up",  dt.Fe,             57.5, 446.1,  -563.8, -417.9),
    ("muon_endcap_dn",  dt.Fe,             57.5, 446.1,   417.9, 563.8),
]

#: The tungsten nozzle, as a chain of cone segments (z_start, z_end, r_start,
#: r_end) on the downstream side; the upstream side is its mirror image. The
#: inner radius is the beam pipe, so each segment is a hollow cone.
NOZZLE_PROFILE = [
    (   6.5, 230.7,  2.2, 31.0),
    ( 230.7, 250.9, 31.0, 33.9),
    ( 250.9, 412.9, 33.9, 56.8),
    ( 412.9, 417.9, 56.8, 57.5),
    ( 417.9, 563.8, 57.5, 78.2),
]

BEAM_PIPE_RADIUS = 2.2        # cm
DETECTOR_HALF_LENGTH = 563.8  # cm
DETECTOR_RADIUS = 645.0       # cm


def _tube(name, material, r_in, r_out, z_min, z_max):
    """One annular region as a MINT TubeVolume."""
    return dt.TubeVolume(material, r_in=r_in, r_out=r_out,
                         length=z_max - z_min,
                         center=(0.0, 0.0, 0.5 * (z_min + z_max)),
                         name=name)


class HollowCone:
    """A cone with a cylindrical bore down its axis.

    MINT's :class:`~mint.detector_tools.ConeVolume` is solid, but the nozzle
    has the beam pipe running through it, so an on-axis neutrino must see no
    tungsten at all. This composes two MINT volumes and takes the difference of
    their chords, which is exact for coaxial convex bodies.
    """

    def __init__(self, material, z0, z1, r0, r1, r_bore, name="hollow_cone"):
        self.material = material
        self.name = name
        self.r_bore = float(r_bore)
        self.z0, self.z1 = float(z0), float(z1)
        self._outer = dt.ConeVolume(material, z0=z0, z1=z1, r0=r0, r1=r1,
                                    name=name + "_outer")
        self._bore = dt.CylinderVolume(material, radius=r_bore,
                                       length=z1 - z0,
                                       center=(0.0, 0.0, 0.5 * (z0 + z1)),
                                       name=name + "_bore")

    def intersect(self, origin, direction):
        t_out, c_out = self._outer.intersect(origin, direction)
        _, c_bore = self._bore.intersect(origin, direction)
        return t_out, np.maximum(c_out - c_bore, 0.0)

    def contains(self, x, y, z):
        return self._outer.contains(x, y, z) & (np.hypot(x, y) >= self.r_bore)


def nozzle_volumes(material=dt.W, r_bore=BEAM_PIPE_RADIUS):
    """The tungsten nozzle, both sides, as cone segments with the bore removed.

    Each segment carries material between the beam pipe and the nozzle's outer
    profile, so a neutrino travelling down the axis crosses none of it.
    """
    vols = []
    for z0, z1, r0, r1 in NOZZLE_PROFILE:
        vols.append(HollowCone(material, z0, z1, r0, r1, r_bore,
                               name=f"nozzle_dn_{z0:.0f}"))
        # mirror image upstream: z -> -z, so the cone opens toward -z
        vols.append(HollowCone(material, -z1, -z0, r1, r0, r_bore,
                               name=f"nozzle_up_{z0:.0f}"))
    return vols


#: HCAL endcap 2 (|z| 250.9-412.9) is bounded outside by r = 324.6 and inside
#: by the nozzle, whose radius runs 33.9 -> 56.8 across that span.
HCAL_ENDCAP2 = (250.9, 412.9, 33.9, 56.8, 324.6)


class ConicalBoreTube:
    """A cylinder with a conical bore: the difference of two MINT volumes.

    Used for the HCAL endcap, whose inner boundary follows the nozzle rather
    than being a constant radius.
    """

    def __init__(self, material, z0, z1, r_bore0, r_bore1, r_out, name="tube"):
        self.material = material
        self.name = name
        self._outer = dt.CylinderVolume(material, radius=r_out, length=z1 - z0,
                                        center=(0.0, 0.0, 0.5 * (z0 + z1)),
                                        name=name + "_outer")
        self._bore = dt.ConeVolume(material, z0=z0, z1=z1,
                                   r0=r_bore0, r1=r_bore1, name=name + "_bore")

    def intersect(self, origin, direction):
        t, c_out = self._outer.intersect(origin, direction)
        _, c_bore = self._bore.intersect(origin, direction)
        return t, np.maximum(c_out - c_bore, 0.0)

    def contains(self, x, y, z):
        return self._outer.contains(x, y, z) & ~self._bore.contains(x, y, z)


def hcal_endcap2_volumes(material=dt.hcal_CLICdet):
    """The outer HCAL endcaps, bored out along the nozzle profile."""
    z0, z1, rb0, rb1, r_out = HCAL_ENDCAP2
    return [ConicalBoreTube(material, z0, z1, rb0, rb1, r_out, name="hcal_endcap2_dn"),
            ConicalBoreTube(material, -z1, -z0, rb1, rb0, r_out, name="hcal_endcap2_up")]


def muc_detector(name="muc_detector", with_nozzle=True):
    """The muon-collider detector as a :class:`mint.detector_tools.VolumeStack`.

    Parameters
    ----------
    with_nozzle : bool
        Include the tungsten nozzles. They dominate the material a forward
        neutrino sees, so switching them off is a useful way to isolate what
        the calorimeters alone contribute.
    """
    vols = [_tube(*r) for r in REGIONS] + hcal_endcap2_volumes()
    if with_nozzle:
        vols += nozzle_volumes()
    return dt.VolumeStack(vols, name=name)


def with_beamline(det=None, lattice=None, straight_half_length=18000.0):
    """The detector plus the beam-pipe shielding of the straight section.

    Returns ``(volume_stack, beamline)``. The shielding is MINT's
    :class:`mint.beamline.StraightSectionShield`, so the material a neutrino
    crosses before reaching the experiment is described the same way as in the
    forward-detector studies.
    """
    import mint

    if det is None:
        det = muc_detector()
    bl = mint.beamline.Beamline(mint.detectors.benchmark, lattice=lattice,
                                straight_half_length=straight_half_length)
    return dt.VolumeStack(list(det.volumes) + list(bl.shield.volumes(sign=+1)),
                          name="muc_detector_with_beamline"), bl


# ---------------------------------------------------------------------------
# Convenience: what a ray actually crosses
# ---------------------------------------------------------------------------

def column_along(stack, origin, direction):
    """Nucleon column density [1/cm^2] each ray accumulates through ``stack``.

    ``origin`` and ``direction`` are (3, N) arrays, as elsewhere in MINT.
    """
    origin = np.asarray(origin, float)
    direction = np.asarray(direction, float)
    total = np.zeros(origin.shape[1])
    for v in stack.volumes:
        _, chord = v.intersect(origin, direction)
        total += chord * v.material.N
    return total


def chords_by_region(stack, origin, direction):
    """{region name: chord [cm]} for each volume the rays cross.

    Useful for asking which part of the detector a given neutrino population
    actually interacts in.
    """
    origin = np.asarray(origin, float)
    direction = np.asarray(direction, float)
    return {v.name: v.intersect(origin, direction)[1] for v in stack.volumes}


def density_map(stack, z_grid, r_grid):
    """Mass density [g/cm^3] on a (z, r) grid, for drawing the geometry.

    Points are tested against every volume; where regions overlap (the beam
    pipe inside a nozzle cone, say) the *last* match wins, which is why the
    beam pipe is listed first in :data:`REGIONS`.
    """
    Z, R = np.meshgrid(np.asarray(z_grid, float), np.asarray(r_grid, float),
                       indexing="ij")
    rho = np.zeros(Z.shape)
    for v in stack.volumes:
        inside = v.contains(R, np.zeros_like(R), Z)
        rho = np.where(inside, v.material.density, rho)
    return rho
