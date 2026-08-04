"""Material budget between the interaction point and a forward detector.

Everything a forward neutrino has to cross on its way out of the machine:

===========================  ======================================
``|z| < 180 m``              machine straight section: vacuum bore
                             wrapped in a tungsten shielding tube
``180 m < z < 250 m``        tunnel air
``250 m < z < D - gap``      standard rock
``D - gap < z < D``          air gap in front of the detector
``z > D``                    the detector itself (mint.detectors)
===========================  ======================================

The shielding geometry follows MuCol Milestone 15 (Calzolari & Vanwelde,
2024, DOI 10.5281/zenodo.14000854), Sec. 2.2 and Table 1: the vacuum aperture
is round with radius max(5 sigma), wrapped in 2.53 cm of tungsten for the
final-focus quadrupoles and 4.5 cm for the chicane dipoles.  The report states
that "in the drift sections, the beam pipe is surrounded by a tungsten
shielding" without giving a thickness, so the quadrupole value is used
throughout the drifts -- see ``SHIELD_T_QUAD``.  Everything else in the radial
build (coils, supports, cold bore, insulation) is deliberately ignored: it is
thin and light compared with the tungsten.

The 5 sigma aperture is the same envelope the beam itself is collimated at in
``mint.MuC.MuDecaySimulator`` (``aperture_nsigma``), so the bore of this tube
is exactly the surface the beam is cut on.
"""

import numpy as np

from mint import detector_tools as dt

#: Half-length of the machine straight section around the IP [cm]. Beyond this
#: the closed orbit bends away (verified against the packaged 10 TeV lattice:
#: the orbit excursion is < 2 cm out to 180 m and reaches 1.3 m by 240 m).
STRAIGHT_HALF_LENGTH = 180e2

#: Where the rock begins [cm]. Between the end of the straight and here the
#: line of sight runs through tunnel air.
ROCK_START = 250e2

#: Free drift for the IP and detector, L* [cm] (MuCol Milestone 15 Sec. 2.1).
#: No machine shielding inside this; the detector nozzle lives here and is NOT
#: modelled (it is upstream of everything the forward flux traverses).
L_STAR = 600.0

#: Tungsten shielding thickness [cm]: quadrupole/drift and chicane dipole
#: (MuCol Milestone 15 Table 1 and Sec. 2.2).
SHIELD_T_QUAD = 2.53
SHIELD_T_DIPOLE = 4.5

#: Engineering floor on the vacuum aperture radius [cm]. The 5 sigma envelope
#: collapses to microns at the beam waists in the long straight, which is not a
#: buildable pipe; ASSUMED, not from the report.
MIN_APERTURE = 2.0


class StraightSectionShield:
    """Tungsten tube around the beam pipe over ``|z| < half_length``.

    Modelled as a stack of coaxial :class:`mint.detector_tools.TubeVolume`
    segments, each of constant inner radius (the largest 5 sigma beam envelope
    inside that segment, floored at ``min_aperture``) and constant wall
    thickness. The geometry is symmetric about the IP, so the same stack serves
    the upstream detector under MINT's z -> -z mirroring.

    Parameters
    ----------
    lattice : a ``mint.lattices`` lattice, optional
        Used to evaluate the 5 sigma envelope. Without it the aperture is
        ``min_aperture`` everywhere.
    n_segments : int
        Number of z segments per side.
    n_sigma : float
        Aperture in units of the RMS beam size (5 in the MuCol design).
    """

    def __init__(self, lattice=None, half_length=STRAIGHT_HALF_LENGTH,
                 n_segments=36, n_sigma=5.0, min_aperture=MIN_APERTURE,
                 thickness=SHIELD_T_QUAD, chicane_thickness=SHIELD_T_DIPOLE,
                 material=None, l_star=L_STAR, n_probe=40001):
        self.half_length = float(half_length)
        self.n_sigma = float(n_sigma)
        self.min_aperture = float(min_aperture)
        self.thickness = float(thickness)
        self.chicane_thickness = float(chicane_thickness)
        self.material = material if material is not None else dt.W
        self.l_star = float(l_star)

        self.edges = np.linspace(self.l_star, self.half_length, n_segments + 1)
        self.r_in = np.full(n_segments, self.min_aperture)
        self.t_wall = np.full(n_segments, self.thickness)
        self.chicane_span = None

        if lattice is not None:
            self._profile_from_lattice(lattice, n_probe)

        self.r_out = self.r_in + self.t_wall

    # -- construction --------------------------------------------------------
    def _profile_from_lattice(self, lattice, n_probe):
        """Per-segment aperture = n_sigma * max(sigma_x, sigma_y), and the
        chicane located from the closed-orbit excursion."""
        u = np.linspace(0.0, 1.0, n_probe)
        s = np.asarray(lattice.s(u), float)
        x = np.asarray(lattice.x(u), float)
        y = np.asarray(lattice.y(u), float)
        z = np.asarray(lattice.z(u), float)
        i_ip = int(np.argmin(x**2 + y**2 + z**2))
        ds = np.abs(s - s[i_ip])          # both sides folded onto |z|
        sig = self.n_sigma * np.maximum(
            np.asarray(lattice.beamsize_x(u), float),
            np.asarray(lattice.beamsize_y(u), float))
        orbit = np.hypot(x, y)            # transverse excursion of the orbit

        for k in range(len(self.r_in)):
            m = (ds >= self.edges[k]) & (ds < self.edges[k + 1])
            if not m.any():
                continue
            self.r_in[k] = max(float(sig[m].max()), self.min_aperture)
            # the chicane dipoles bend the orbit off axis; give those segments
            # the thicker dipole shielding quoted in the report
            if orbit[m].max() > 0.5:      # cm
                self.t_wall[k] = self.chicane_thickness
        span = [self.edges[k:k + 2] for k in range(len(self.r_in))
                if self.t_wall[k] == self.chicane_thickness]
        if span:
            self.chicane_span = (float(span[0][0]), float(span[-1][1]))

    # -- geometry ------------------------------------------------------------
    def volumes(self, sign=+1, both_sides=True):
        """The tube segments as ``TubeVolume``s, in world coordinates.

        With ``both_sides`` the mirrored ``-z`` stack is included: neutrinos
        from the far straight cross the near one on their way out.
        """
        vols = []
        sides = (+1, -1) if both_sides else (+1,)
        for side in sides:
            for k in range(len(self.r_in)):
                z0, z1 = self.edges[k], self.edges[k + 1]
                vols.append(dt.TubeVolume(
                    self.material, r_in=self.r_in[k], r_out=self.r_out[k],
                    length=z1 - z0,
                    center=(0.0, 0.0, sign * side * 0.5 * (z0 + z1)),
                    name=f"shield{'+' if side > 0 else '-'}{k}"))
        return vols

    def segment_chords(self, origin, direction, sign=+1, both_sides=True):
        """(n_ray, n_seg) chord [cm] of each ray through each tube segment."""
        vols = self.volumes(sign=sign, both_sides=both_sides)
        return np.column_stack([v.intersect(origin, direction)[1] for v in vols])

    def chord(self, origin, direction, sign=+1, both_sides=True):
        """Total tungsten path length [cm] of each ray through the shielding."""
        return self.segment_chords(origin, direction, sign=sign,
                                   both_sides=both_sides).sum(axis=1)

    def segment_bounds(self, sign=+1, both_sides=True):
        """(z0, z1) of every segment returned by :meth:`volumes`, in the same
        order, as world z [cm] (already multiplied by ``sign``)."""
        z0, z1 = [], []
        sides = (+1, -1) if both_sides else (+1,)
        for side in sides:
            for k in range(len(self.r_in)):
                a, b = sign * side * self.edges[k], sign * side * self.edges[k + 1]
                z0.append(min(a, b))
                z1.append(max(a, b))
        return np.array(z0), np.array(z1)

    def __repr__(self):
        return (f"StraightSectionShield(|z| < {self.half_length/100:.0f} m, "
                f"{len(self.r_in)} segments, aperture "
                f"{self.r_in.min():.1f}-{self.r_in.max():.1f} cm, "
                f"wall {self.t_wall.min():.2f}-{self.t_wall.max():.2f} cm W)")


class Beamline:
    """The full IP -> detector material model.

    Bundles the straight-section shielding with the tunnel air, the rock and
    the air gap in front of a :class:`mint.detectors.ForwardDetector`, and
    turns a bundle of neutrino rays into a piecewise-constant material profile
    along the line of sight -- the input for secondary-muon and secondary-tau
    production anywhere upstream of the detector.
    """

    def __init__(self, det, lattice=None, shield=None, air=None,
                 straight_half_length=STRAIGHT_HALF_LENGTH):
        self.det = det
        self.shield = (shield if shield is not None
                       else StraightSectionShield(lattice=lattice,
                                                  half_length=straight_half_length))
        self.air = air if air is not None else dt.Air
        self.rock = det.rock
        self.rock_start = det.rock_start
        self.straight_half_length = float(straight_half_length)

    # -- 1D (on-axis) description -------------------------------------------
    def bulk_slabs(self, dist=None):
        """[(z0, z1, material, name)] of the transversely uniform media between
        the end of the straight section and the detector face: tunnel air,
        rock, and the air gap. The shielding is NOT here -- it is a thin tube
        and its contribution depends on the individual ray."""
        D = self.det.dist if dist is None else dist
        gap = self.det.gap
        z_rock0 = self.rock_start
        z_rock1 = max(D - gap, z_rock0)
        out = []
        if D > self.straight_half_length:
            out.append((self.straight_half_length, min(z_rock0, D),
                        self.air, "tunnel air"))
        if z_rock1 > z_rock0 and D > z_rock0:
            out.append((z_rock0, z_rock1, self.rock, "rock"))
        if D > z_rock1:
            out.append((z_rock1, D, self.air, "gap"))
        return [(a, b, m, n) for a, b, m, n in out if b > a]

    def columns(self, dist=None):
        """{name: (length_cm, nucleons/cm^2, g/cm^2)} on the axis, plus the
        shielding evaluated for a ray straight down the bore (which is zero --
        the bore is empty; use :meth:`ray_columns` for real rays)."""
        out = {}
        for z0, z1, mat, name in self.bulk_slabs(dist=dist):
            L = z1 - z0
            out[name] = (L, mat.N * L, mat.density * L)
        return out

    # -- per-ray description -------------------------------------------------
    def ray_profile(self, origin, direction, dist=None, sign=+1,
                    both_sides=True):
        """Piecewise-constant material along each ray, IP -> detector face.

        The shielding segments enter with an EFFECTIVE density: a ray crossing
        a 20 m segment with a chord ``l`` through the tungsten is given
        ``rho_W * l / (z1 - z0)`` over that segment. This reproduces the column
        density exactly and places the production vertex within the correct
        20 m segment, which is all the downstream transport is sensitive to.

        Returns a dict of arrays with a common slab axis:
          z0, z1   : (n_slab,)          slab boundaries along z [cm]
          n_nuc    : (n_ray, n_slab)    nucleons/cm^3
          n_e      : (n_ray, n_slab)    electrons/cm^3
          rho      : (n_ray, n_slab)    g/cm^3 (for muon transport)
          names    : (n_slab,)
        """
        D = self.det.dist if dist is None else dist
        n_ray = np.asarray(origin[0]).size

        sz0, sz1 = self.shield.segment_bounds(sign=sign, both_sides=both_sides)
        chords = self.shield.segment_chords(origin, direction, sign=sign,
                                            both_sides=both_sides)
        # a shield segment only counts if it is upstream of the face
        keep = sz1 <= max(D, 0.0)
        sz0, sz1, chords = sz0[keep], sz1[keep], chords[:, keep]
        frac = chords / np.maximum(sz1 - sz0, 1e-30)          # (n_ray, n_seg)
        mat = self.shield.material
        s_n = frac * mat.N
        s_e = frac * mat.e
        s_rho = frac * mat.density
        s_names = [f"shield{k}" for k in range(sz0.size)]

        b = self.bulk_slabs(dist=dist)
        bz0 = np.array([s[0] for s in b])
        bz1 = np.array([s[1] for s in b])
        b_n = np.tile([s[2].N for s in b], (n_ray, 1)) if b else np.zeros((n_ray, 0))
        b_e = np.tile([s[2].e for s in b], (n_ray, 1)) if b else np.zeros((n_ray, 0))
        b_rho = (np.tile([s[2].density for s in b], (n_ray, 1)) if b
                 else np.zeros((n_ray, 0)))
        b_names = [s[3] for s in b]

        z0 = np.concatenate([sz0, bz0]) if b else sz0
        z1 = np.concatenate([sz1, bz1]) if b else sz1
        order = np.argsort(z0)
        return dict(
            z0=z0[order], z1=z1[order],
            n_nuc=np.hstack([s_n, b_n])[:, order],
            n_e=np.hstack([s_e, b_e])[:, order],
            rho=np.hstack([s_rho, b_rho])[:, order],
            names=list(np.array(s_names + b_names)[order]),
        )

    def ray_columns(self, origin, direction, dist=None, sign=+1,
                    both_sides=True):
        """(nucleons/cm^2, electrons/cm^2, g/cm^2) integrated along each ray."""
        p = self.ray_profile(origin, direction, dist=dist, sign=sign,
                             both_sides=both_sides)
        L = p["z1"] - p["z0"]
        return (p["n_nuc"] @ L, p["n_e"] @ L, p["rho"] @ L)

    def flux_weighted_profile(self, sim, dist=None, sign=+1, exposure=1.0,
                              r_sel=None, E_min=0.0, both_sides=True):
        """Longitudinal target profile seen by the flux that reaches the face.

        Ray-traces the simulated neutrinos, keeps those crossing the detector
        face, and averages their material profiles with the flux weights. The
        tungsten shielding enters through the average chord of the accepted
        rays, so the total column is exact for any rate that is linear in it
        (all production rates are), while the ray-by-ray correlation between
        emission angle and tungsten path is averaged over.

        Returns ``(z0, z1, n_nuc, n_e, rho, names)`` -- 1-D arrays over slabs.
        """
        D = self.det.dist if dist is None else dist
        r_sel = self.det.radius if r_sel is None else r_sel
        (ox, oy, oz), (vx, vy, vz) = dt.sim_rays(sim)
        E = np.asarray(sim.pnu["E"], float)
        w = sim.weights.flatten() * exposure
        if sign < 0:
            oz, vz = -oz, -vz
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (D - oz) / vz
        rx, ry = ox + t * vx, oy + t * vy
        m = (t > 0) & (E > E_min) & (rx**2 + ry**2 < r_sel**2)
        origin = np.vstack([ox[m], oy[m], oz[m]])
        direction = np.vstack([vx[m], vy[m], vz[m]])
        p = self.ray_profile(origin, direction, dist=D, sign=+1,
                             both_sides=both_sides)
        ww = w[m] / max(w[m].sum(), 1e-300)
        return dict(z0=p["z0"], z1=p["z1"], names=p["names"],
                    n_nuc=ww @ p["n_nuc"], n_e=ww @ p["n_e"],
                    rho=ww @ p["rho"], n_rays=int(m.sum()), flux=float(w[m].sum()))

    # -- production sampling -------------------------------------------------
    @staticmethod
    def sample_vertices(profile, rng, on="n_nuc"):
        """One production vertex per ray, distributed along the line of sight
        in proportion to the target density ``profile[on]``.

        Returns ``(z, column, islab)``: the vertex position [cm], the TOTAL
        column of ``on`` along the ray (nucleons or electrons per cm^2 -- the
        normalisation, so that the per-ray yield is ``sigma * column``), and
        the index of the slab the vertex landed in.
        """
        n = np.asarray(profile[on], float)                 # (n_ray, n_slab)
        L = np.asarray(profile["z1"] - profile["z0"], float)
        w = n * L                                          # column per slab
        tot = w.sum(axis=1)
        cdf = np.cumsum(w, axis=1)
        u = rng.uniform(size=tot.size) * np.maximum(tot, 1e-300)
        islab = np.array([np.searchsorted(c, uu) for c, uu in zip(cdf, u)])
        islab = np.clip(islab, 0, L.size - 1)
        # uniform inside the chosen slab
        z0 = np.asarray(profile["z0"])[islab]
        z = z0 + rng.uniform(size=tot.size) * L[islab]
        return z, tot, islab

    @staticmethod
    def downstream_slabs(profile, z, dist):
        """Material a particle created at ``z`` still has to cross to reach the
        face at ``dist``.

        Returns ``(X, rho, length)``, each ``(n_ray, n_slab)``: grammage
        [g/cm^2], density [g/cm^3] and length [cm] of the portion of every slab
        downstream of ``z``. Slabs fully upstream contribute zero.
        """
        z = np.asarray(z, float)[:, None]
        z0 = np.asarray(profile["z0"], float)[None, :]
        z1 = np.minimum(np.asarray(profile["z1"], float)[None, :], dist)
        seg = np.clip(z1 - np.maximum(z0, z), 0.0, None)
        rho = np.asarray(profile["rho"], float)
        return rho * seg, rho, seg

    # -- density map (validation) -------------------------------------------
    def density_map(self, z_grid, r_grid, dist=None, sign=+1):
        """Mass density [g/cm^3] on a (z, r) grid -- the picture of the problem.

        Includes the shielding tube, tunnel air, rock, gap and the detector
        (gas + tungsten slabs). Returns an array of shape
        ``(len(r_grid), len(z_grid))`` suitable for ``pcolormesh``.
        """
        D = self.det.dist if dist is None else dist
        Z, R = np.meshgrid(np.asarray(z_grid, float), np.asarray(r_grid, float))
        rho = np.zeros_like(Z)

        for z0, z1, mat, _ in self.bulk_slabs(dist=dist):
            rho[(Z >= z0) & (Z < z1)] = mat.density

        # the straight section: vacuum bore, tungsten wall, tunnel air outside
        sh = self.shield
        for k in range(len(sh.r_in)):
            for side in (+1, -1):
                a, b = sorted((side * sh.edges[k], side * sh.edges[k + 1]))
                inseg = (Z >= a) & (Z < b)
                rho[inseg & (R < sh.r_in[k])] = 0.0
                rho[inseg & (R >= sh.r_in[k]) & (R < sh.r_out[k])] = \
                    sh.material.density
                rho[inseg & (R >= sh.r_out[k])] = self.air.density

        # the detector paints itself: a segmented, conical detector knows its
        # own module stack, and only the legacy cylinder needs the fallback
        det = self.det
        if hasattr(det, "fill_density"):
            return det.fill_density(Z, R, rho, sign=+1, dist=D)
        indet = (Z >= D) & (Z < D + det.length) & (R < det.radius)
        rho[indet] = det.gas.density
        for z_off, thick, mat in det.targets:
            slab = (Z >= D + z_off) & (Z < D + z_off + thick) & (R < det.radius)
            rho[slab] = mat.density
        return rho

    def __repr__(self):
        return (f"Beamline(straight |z|<{self.straight_half_length/100:.0f} m, "
                f"rock from {self.rock_start/100:.0f} m, det at "
                f"{self.det.dist/1e5:.2f} km)")
