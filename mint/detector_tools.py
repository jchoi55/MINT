import numpy as np
import pandas as pd

from mint import const


class Material:
    """Pure substances; periodic elements.

    ``X0`` and ``lambda_I`` are the PDG radiation and nuclear interaction
    lengths in g/cm^2. They are carried on the material rather than in a
    lookup table so that materials built at run time (a gas at a different
    pressure, say) keep them; nothing in the MINT transport depends on them,
    they exist to report a material budget.
    """

    def __init__(self, density, am, A, Z, X0=None, lambda_I=None):
        self.density = density
        self.am = am
        self.Z = Z
        self.A = A
        self.X0 = X0
        self.lambda_I = lambda_I
        nq = const.NAvo * self.density / self.am
        self.N = nq * self.A
        self.e = nq * self.Z

    def scaled(self, factor):
        """Same substance at ``factor`` times the density (e.g. a gas at a
        different pressure). X0 and lambda_I are per unit mass, so they are
        unchanged."""
        return Material(self.density * factor, self.am, self.A, self.Z,
                        X0=self.X0, lambda_I=self.lambda_I)


class CompositMaterial:
    def __init__(self, table):
        """Compositions of materials.
        fraction is the percentage of it that occupies the total material
        """
        self.components = [(m, f) for m, f in table]
        self.density = 0
        self.N = 0
        self.e = 0
        self.Z = 0
        self.A = 0
        for material, fraction in table:
            self.density += material.density * fraction
            self.N += material.N * fraction
            self.e += material.e * fraction
            self.Z += material.Z * fraction
            self.A += material.A * fraction
        # X0 and lambda_I add as 1/X = sum_i w_i / X_i over MASS fractions
        self.X0 = self.lambda_I = None
        if self.density > 0:
            for attr in ("X0", "lambda_I"):
                inv = 0.0
                for material, fraction in table:
                    val = getattr(material, attr, None)
                    if val is None or not np.isfinite(val) or val <= 0:
                        inv = None
                        break
                    w = material.density * fraction / self.density
                    inv += w / val
                if inv:
                    setattr(self, attr, 1.0 / inv)


# class unif(Material):
#     """Alloys/compositions of uniform densities."""

#     def __init__(self, density):
#         super().__init__()
#         self.density = density
#         self.N = self.density / (const.m_avg / const.g_to_GeV)
#         self.e = self.N / 2


# Pre-defined substances

# density in g/cm**3; atomic mass in g/mol
Si = Material(2.329, 28.0855, 28, 14, X0=21.82, lambda_I=108.4)
WSi2 = Material(9.3, 240.01, 240, 102)
Fe = Material(7.874, 55.845, 56, 26, X0=13.84, lambda_I=132.1)
Al = Material(2.7, 26.981539, 27, 13, X0=24.01, lambda_I=107.2)
W = Material(19.3, 183.84, 184, 74, X0=6.76, lambda_I=191.9)
Cu = Material(8.96, 63.546, 64, 29, X0=12.86, lambda_I=137.3)
PS = Material(1.05, 104.1, 104, 56, X0=43.79, lambda_I=81.7)
vacuum = Material(0, 1, 0, 0)

# "Standard rock" (conventional isoscalar rock: Z=11, A=22, rho=2.65 g/cm^3).
# The closest single DarkNews isotope for the upscattering cross section is Na23.
standard_rock = Material(2.65, 22.0, 22, 11, X0=26.54, lambda_I=120.0)

# High-pressure gaseous argon at 10 bar, ~293 K (DUNE ND-GAr-like TPC medium):
# ideal-gas density rho = P M / (R T) = 1e6 Pa * 39.95 g/mol / (8.314 * 293 K)
# = 0.01640 g/cm^3 (the previous 0.0166 corresponded to 10 atm, not 10 bar).
GAr_10bar = Material(0.01640, 39.95, 40, 18, X0=19.55, lambda_I=117.2)

# 5 bar argon: half the vessel load and half the radiation-length budget of the
# 10 bar option, in a demonstrated regime for micropattern gain. The benchmark
# forward detector uses this as its tracking medium.
GAr_5bar = Material(8.31e-3, 39.95, 40, 18, X0=19.55, lambda_I=117.2)

# Graphite / carbon-fibre composite: the benchmark vertex-target foils and the
# TPC pressure-vessel walls. X0 = 42.70 g/cm^2, lambda_I = 85.8 g/cm^2 (PDG).
graphite = Material(2.21, 12.011, 12, 6, X0=42.70, lambda_I=85.8)

def dominant_nucleus(material):
    """The single nucleus that best represents ``material``.

    A pure :class:`Material` represents itself. For a :class:`CompositMaterial`
    the ``Z`` and ``A`` attributes are volume-fraction sums and are NOT a
    nucleus -- a graphite/silicon mixture comes out at Z = 0.28 -- so anything
    that needs a real nucleus (tabulated upscattering cross sections, coherent
    form factors, non-isoscalar corrections) must resolve the mixture. This
    returns the component carrying the most nucleons, which is the right
    single-nucleus stand-in when one component dominates; loop over
    ``material.components`` when it does not.
    """
    comps = getattr(material, "components", None)
    if not comps:
        return material
    best, best_col = material, -1.0
    for m, f in comps:
        col = m.N * f
        if col > best_col:
            best, best_col = dominant_nucleus(m), col
    return best


def nuclei_density(material):
    """Nuclei per cm^3 of ``material``.

    For a pure substance this is simply ``N / A``. For a mixture ``A`` is a
    volume-fraction sum and is not a mass number at all (graphite + silicon
    gives A = 0.56), so dividing by it inflates the nuclei density by more than
    an order of magnitude. Here the nucleon density -- which is what drives
    deep-inelastic rates and is correct for a mixture -- is kept, and those
    nucleons are assigned to the dominant nucleus. That is exact per nucleon
    and approximate only for genuinely coherent processes.
    """
    return material.N / dominant_nucleus(material).A


def radiation_length(material):
    """X0 [g/cm^2] carried by a material (NaN if unknown)."""
    v = getattr(material, "X0", None)
    return np.nan if v is None else float(v)


def interaction_length(material):
    """lambda_I [g/cm^2] carried by a material (NaN if unknown)."""
    v = getattr(material, "lambda_I", None)
    return np.nan if v is None else float(v)

# from CLICdet paper
hcal_CLICdet = CompositMaterial(
    [[Fe, 20 / 26.5], [Al, 0.7 / 26.5], [Cu, 0.1 / 26.5], [PS, 3 / 26.5]]
)
ecal_CLICdet = CompositMaterial([[W, 1.9 / 5.05], [Cu, 2.3 / 5.05], [Si, 0.5 / 5.05]])

OinAir = Material(1.225e-3, 15.9994, 16, 8, X0=34.24, lambda_I=90.1)
NinAir = Material(1.225e-3, 14.0067, 14, 7, X0=37.99, lambda_I=89.7)
ArinAir = Material(1.225e-3, 39.95, 40, 18, X0=19.55, lambda_I=117.2)
Air = CompositMaterial(
    [
        [OinAir, 0.2095],
        [NinAir, 0.7812],
        [ArinAir, 0.0093],
    ]
)


####################################################################################################
# Detector geometry and neutrino interaction generation
####################################################################################################


def bin_flux(E, w, n_bins=40, E_min=20.0):
    """Compress an MC flux (E, w) into weighted energy bins.
    Returns (E_center [flux-weighted mean], W [nu/yr per bin]).

    NOTE: the output is a quadrature grid (delta weights at the centers) for
    folding with smooth cross sections or as an MC sampling measure. Do NOT
    re-histogram it onto a different binning for plotting -- incommensurate
    grids alias into a castle-wall pattern; histogram the RAW (E, w) instead."""
    E = np.asarray(E, float)
    w = np.asarray(w, float)
    edges = np.geomspace(max(E_min, E.min() * 0.999), E.max() * 1.001, n_bins + 1)
    W, _ = np.histogram(E, bins=edges, weights=w)
    Esum, _ = np.histogram(E, bins=edges, weights=w * E)
    Ec = np.where(W > 0, Esum / np.maximum(W, 1e-300), np.sqrt(edges[:-1] * edges[1:]))
    return Ec, W


def sim_rays(sim):
    """Extract the neutrino rays of a placed simulation.

    Returns (origin, direction): two (3, N)-like tuples of arrays -- decay
    positions [cm] and unit momentum directions -- ready for
    ``CylinderVolume.intersect(origin, direction)``.
    """
    u3 = sim.pnu.to_3D().unit()
    origin = (
        np.asarray(sim.pos["x"], dtype=float),
        np.asarray(sim.pos["y"], dtype=float),
        np.asarray(sim.pos["z"], dtype=float),
    )
    direction = (
        np.asarray(u3.x, dtype=float),
        np.asarray(u3.y, dtype=float),
        np.asarray(u3.z, dtype=float),
    )
    return origin, direction


class CylinderVolume:
    """A finite cylinder aligned with the z (beam) axis, filled with a material.

    Parameters
    ----------
    material : Material or CompositMaterial
        Target material; ``material.N`` (nucleons/cm^3) sets the interaction rate
        for per-nucleon cross sections.
    radius : float
        Cylinder radius [cm].
    length : float
        Cylinder length along z [cm].
    center : (x, y, z)
        Position of the cylinder center [cm] in world coordinates (the IP is at
        the origin, +z along the beam at the IP).
    name : str
        Label attached to the interactions generated in this volume.
    """

    def __init__(self, material, radius, length, center=(0.0, 0.0, 0.0), name="volume"):
        self.material = material
        self.radius = float(radius)
        self.length = float(length)
        self.center = np.asarray(center, dtype=float)
        self.name = name

    def intersect(self, origin, direction):
        """Chord of each ray through the volume.

        Parameters
        ----------
        origin : (3, N) array
            Ray starting points [cm] (e.g. muon decay positions).
        direction : (3, N) array
            Ray directions (need not be normalized; chords are returned in the
            same parametrization units as ``|direction|``, so pass unit vectors
            to get cm).

        Returns
        -------
        t_entry, chord : arrays [cm]
            Distance from the origin to the entry point (clamped to >= 0 for
            rays starting inside), and the path length inside the volume
            (0 where the ray misses the volume entirely).
        """
        ox, oy, oz = (np.asarray(origin[i], dtype=float) - self.center[i] for i in range(3))
        dx, dy, dz = (np.asarray(direction[i], dtype=float) for i in range(3))

        INF = np.inf

        # --- intersection with the infinite cylinder x^2 + y^2 = R^2
        a = dx**2 + dy**2
        b = 2.0 * (ox * dx + oy * dy)
        c = ox**2 + oy**2 - self.radius**2

        radial = a > 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            disc = b**2 - 4.0 * a * c
            sqrt_disc = np.sqrt(np.where(disc > 0.0, disc, 0.0))
            tr1 = np.where(radial & (disc >= 0.0), (-b - sqrt_disc) / (2.0 * a), -INF)
            tr2 = np.where(radial & (disc >= 0.0), (-b + sqrt_disc) / (2.0 * a), INF)
        # rays parallel to the axis: inside for all t if within the radius
        parallel_out = (~radial) & (c > 0.0)
        miss_radial = (radial & (disc < 0.0)) | parallel_out
        tr1 = np.where(miss_radial, INF, tr1)
        tr2 = np.where(miss_radial, -INF, tr2)

        # --- intersection with the slab |z| <= L/2
        half = self.length / 2.0
        axial = dz != 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            ta = np.where(axial, (-half - oz) / np.where(axial, dz, 1.0), -INF)
            tb = np.where(axial, (half - oz) / np.where(axial, dz, 1.0), INF)
        tz1 = np.minimum(ta, tb)
        tz2 = np.maximum(ta, tb)
        # rays perpendicular to the axis: inside the slab for all t or never
        slab_out = (~axial) & (np.abs(oz) > half)
        tz1 = np.where(slab_out, INF, tz1)
        tz2 = np.where(slab_out, -INF, tz2)

        # --- combined interval, forward-going only (t >= 0)
        t_entry = np.maximum(np.maximum(tr1, tz1), 0.0)
        t_exit = np.minimum(tr2, tz2)
        chord = np.clip(t_exit - t_entry, 0.0, None)
        t_entry = np.where(chord > 0.0, t_entry, 0.0)
        return t_entry, chord


class TubeVolume:
    """A hollow cylinder (annulus) aligned with the z axis: a pipe wall.

    Used for the tungsten shielding wrapped around the machine beam pipe,
    where the bore is vacuum and only the wall is material. The chord is
    obtained exactly as ``chord(outer) - chord(inner)``: the inner cylinder
    shares the z extent of the outer one, so the difference is the path
    length in the annulus, correctly summing the two crossings of a ray that
    passes through the bore.

    Parameters as :class:`CylinderVolume`, with ``r_in``/``r_out`` the inner
    and outer wall radii [cm].
    """

    def __init__(self, material, r_in, r_out, length, center=(0.0, 0.0, 0.0),
                 name="tube"):
        if not r_out > r_in >= 0.0:
            raise ValueError(f"need r_out > r_in >= 0, got {r_in}, {r_out}")
        self.material = material
        self.r_in = float(r_in)
        self.r_out = float(r_out)
        self.radius = float(r_out)
        self.length = float(length)
        self.center = np.asarray(center, dtype=float)
        self.name = name
        self._outer = CylinderVolume(material, r_out, length, center, name)
        self._inner = CylinderVolume(material, r_in, length, center, name)

    def contains(self, x, y, z):
        """Boolean mask: is the point inside the wall? (for density maps)"""
        r = np.hypot(np.asarray(x) - self.center[0], np.asarray(y) - self.center[1])
        dz = np.abs(np.asarray(z) - self.center[2])
        return (r >= self.r_in) & (r <= self.r_out) & (dz <= self.length / 2.0)

    def intersect(self, origin, direction):
        """(t_entry, chord): chord is the total path length inside the wall.

        ``t_entry`` is the entry into the OUTER cylinder, which for a ray
        arriving down the bore is the entry into the bore rather than into the
        material. It is returned only for interface compatibility with
        :class:`CylinderVolume`; use ``chord`` for column densities.
        """
        t_out, ch_out = self._outer.intersect(origin, direction)
        _, ch_in = self._inner.intersect(origin, direction)
        return t_out, np.clip(ch_out - ch_in, 0.0, None)


class ConeVolume:
    """A truncated cone (frustum) aligned with the z axis, filled with a material.

    The radius grows linearly from ``r0`` at ``z0`` to ``r1`` at ``z1``. With
    ``r0 == r1`` this is exactly a cylinder, so it can be used uniformly for
    every module of a detector whose aperture opens with a fixed acceptance
    angle.

    Parameters
    ----------
    material : Material or CompositMaterial
    z0, z1 : float
        Front and back planes [cm] in world coordinates (``z1 > z0``).
    r0, r1 : float
        Radii [cm] at ``z0`` and ``z1`` (both >= 0).
    center_xy : (x, y)
        Transverse position of the axis [cm].
    name : str
    """

    def __init__(self, material, z0, z1, r0, r1, center_xy=(0.0, 0.0),
                 name="cone"):
        if not z1 > z0:
            raise ValueError(f"need z1 > z0, got {z0}, {z1}")
        if min(r0, r1) < 0.0:
            raise ValueError(f"need r0, r1 >= 0, got {r0}, {r1}")
        self.material = material
        self.z0, self.z1 = float(z0), float(z1)
        self.r0, self.r1 = float(r0), float(r1)
        self.k = (self.r1 - self.r0) / (self.z1 - self.z0)   # dR/dz
        self.center = np.array([center_xy[0], center_xy[1],
                                0.5 * (self.z0 + self.z1)], dtype=float)
        self.name = name

    # -- geometry ------------------------------------------------------------
    @property
    def length(self):
        return self.z1 - self.z0

    @property
    def radius(self):
        """Largest radius (for halo selections and plot limits)."""
        return max(self.r0, self.r1)

    def radius_at(self, z):
        """Radius [cm] at ``z``; outside [z0, z1] the linear form is extrapolated."""
        return self.r0 + self.k * (np.asarray(z, float) - self.z0)

    def volume_cm3(self):
        """Exact frustum volume, pi/3 L (r0^2 + r0 r1 + r1^2)."""
        return (np.pi / 3.0 * self.length
                * (self.r0**2 + self.r0 * self.r1 + self.r1**2))

    def contains(self, x, y, z):
        r = np.hypot(np.asarray(x, float) - self.center[0],
                     np.asarray(y, float) - self.center[1])
        z = np.asarray(z, float)
        return (z >= self.z0) & (z <= self.z1) & (r <= self.radius_at(z))

    def intersect(self, origin, direction):
        """(t_entry, chord) for each ray, as :class:`CylinderVolume`.

        The frustum is convex, so a ray meets it in a single interval. Rather
        than case-splitting on the sign of the quadratic coefficient (which
        flips when the ray is steeper in z than the cone half-angle, and which
        also picks up the mirror nappe), the boundary times are collected --
        the two z planes, the (at most) two lateral-surface roots, and t = 0 --
        sorted, and each resulting sub-interval is classified by testing its
        midpoint. That is exact for any convex body and degrades gracefully
        when the quadratic is singular.
        """
        ox = np.asarray(origin[0], float) - self.center[0]
        oy = np.asarray(origin[1], float) - self.center[1]
        oz = np.asarray(origin[2], float)
        dx, dy, dz = (np.asarray(direction[i], float) for i in range(3))
        ox, oy, oz, dx, dy, dz = np.broadcast_arrays(ox, oy, oz, dx, dy, dz)

        # lateral surface: |r(t)|^2 = R(z(t))^2
        w0 = self.r0 + self.k * (oz - self.z0)          # cone radius at the origin
        a = dx**2 + dy**2 - (self.k * dz) ** 2
        b = 2.0 * (ox * dx + oy * dy - self.k * dz * w0)
        c = ox**2 + oy**2 - w0**2
        with np.errstate(divide="ignore", invalid="ignore"):
            disc = b**2 - 4.0 * a * c
            sq = np.sqrt(np.where(disc > 0.0, disc, 0.0))
            quad = np.abs(a) > 0.0
            tr1 = np.where(quad & (disc >= 0), (-b - sq) / (2.0 * a), 0.0)
            tr2 = np.where(quad & (disc >= 0), (-b + sq) / (2.0 * a), 0.0)
            # a == 0: single root of the linear equation b t + c = 0
            lin = (~quad) & (np.abs(b) > 0.0)
            tr1 = np.where(lin, -c / np.where(np.abs(b) > 0.0, b, 1.0), tr1)
            tr2 = np.where(lin, tr1, tr2)

        # z planes
        with np.errstate(divide="ignore", invalid="ignore"):
            axial = dz != 0.0
            dzs = np.where(axial, dz, 1.0)
            tz0 = np.where(axial, (self.z0 - oz) / dzs, 0.0)
            tz1 = np.where(axial, (self.z1 - oz) / dzs, 0.0)

        cand = np.stack([np.zeros_like(ox), tz0, tz1, tr1, tr2])
        cand = np.where(np.isfinite(cand), cand, 0.0)
        cand = np.sort(cand, axis=0)                     # (5, ...)

        lo, hi = cand[:-1], cand[1:]                     # (4, ...)
        mid = 0.5 * (lo + hi)
        inside = self.contains(ox + mid * dx + self.center[0],
                               oy + mid * dy + self.center[1],
                               oz + mid * dz)
        good = inside & (lo >= 0.0) & (hi > lo)
        chord = np.sum(np.where(good, hi - lo, 0.0), axis=0)
        t_entry = np.min(np.where(good, lo, np.inf), axis=0)
        t_entry = np.where(np.isfinite(t_entry), t_entry, 0.0)
        return t_entry, chord


class VolumeStack:
    """A set of non-overlapping material volumes that neutrinos can traverse.

    This is the ray-tracing substrate, not a detector model: it knows only
    where the material is. :class:`mint.detectors.Detector` is what describes
    an actual detector, and builds one of these on demand.

    Given a placed muon-decay simulation, ``generate_interactions`` ray-traces
    every simulated neutrino through the volumes and produces weighted
    interaction vertices, distributed along each chord according to the
    exponential attenuation law exp(-n sigma s) (uniform along the chord in the
    thin-target limit). Attenuation in upstream volumes is accounted for.
    """

    def __init__(self, volumes, name="detector"):
        self.volumes = list(volumes)
        self.name = name

    def generate_interactions(self, sim, exposure=1.0, xsec=None):
        """Generate neutrino interaction vertices inside the detector.

        Parameters
        ----------
        sim : mint.MuC.MuDecaySimulator
            A simulation after ``place_muons_on_lattice`` (needs ``pos``,
            ``pnu``, ``weights``, ``mutimes_to_bunchx``).
        exposure : float
            Multiplier applied to the simulation weights, e.g. the number of
            bunch crossings per year (weights are per bunch by default).
        xsec : callable, optional
            Per-nucleon total cross section sigma(E_nu) [cm^2]. Defaults to
            ``mint.xsecs.total_xsecs[sim.nuflavor]``.

        Returns
        -------
        pandas.DataFrame
            One row per simulated neutrino that crosses a volume, with columns:
            x, y, z [cm] interaction vertex; E [GeV]; ux, uy, uz neutrino
            direction; t_bunchx [s] vertex time relative to the IP bunch
            crossing; w [interactions per exposure]; volume label.
            ``df.attrs`` stores flavor, detector name, and the total rate.
        """
        from mint import xsecs as _xsecs

        if not hasattr(sim, "pnu") or not hasattr(sim, "pos"):
            raise RuntimeError(
                "The simulation has no placed events. "
                "Run decay_muons() and place_muons_on_lattice() first."
            )
        if xsec is None:
            try:
                xsec = _xsecs.total_xsecs[sim.nuflavor]
            except KeyError as exc:
                raise ValueError(
                    f"No default total cross section for flavor '{sim.nuflavor}'. "
                    "Pass xsec=callable(E_nu) explicitly."
                ) from exc

        E = np.asarray(sim.pnu["E"], dtype=float)
        origin, direction = sim_rays(sim)
        w_nu = sim.weights.flatten() * exposure
        sigma = np.clip(xsec(E), 0.0, None)

        # per-volume entry distance, chord, and optical depth tau = n sigma * chord
        entries, chords, taus = [], [], []
        for vol in self.volumes:
            t_entry, chord = vol.intersect(origin, direction)
            lam = vol.material.N * sigma  # inverse interaction length [1/cm]
            entries.append(t_entry)
            chords.append(chord)
            taus.append(lam * chord)

        frames = []
        for k, vol in enumerate(self.volumes):
            hit = (chords[k] > 0.0) & (E > 0.0) & (w_nu > 0.0)
            if not np.any(hit):
                continue

            tau = taus[k][hit]
            lam = vol.material.N * sigma[hit]

            # survival through material crossed upstream of this volume
            upstream_tau = np.zeros(hit.sum())
            for j in range(len(self.volumes)):
                if j == k:
                    continue
                before = entries[j][hit] < entries[k][hit]
                upstream_tau += np.where(before, taus[j][hit], 0.0)

            # interaction probability and vertex position along the chord,
            # sampled from the truncated exponential (uniform when tau -> 0)
            p_int = -np.expm1(-tau)
            u = np.random.uniform(size=hit.sum())
            with np.errstate(divide="ignore", invalid="ignore"):
                s = np.where(
                    tau > 0.0,
                    -np.log1p(u * np.expm1(-tau)) / np.where(lam > 0.0, lam, 1.0),
                    0.0,
                )

            t_vertex = entries[k][hit] + s
            w_int = w_nu[hit] * np.exp(-upstream_tau) * p_int

            frames.append(
                pd.DataFrame(
                    {
                        "x": origin[0][hit] + t_vertex * direction[0][hit],
                        "y": origin[1][hit] + t_vertex * direction[1][hit],
                        "z": origin[2][hit] + t_vertex * direction[2][hit],
                        "E": E[hit],
                        "ux": direction[0][hit],
                        "uy": direction[1][hit],
                        "uz": direction[2][hit],
                        "t_bunchx": np.asarray(sim.mutimes_to_bunchx)[hit]
                        + t_vertex / const.c_LIGHT,
                        "w": w_int,
                        "volume": vol.name,
                    }
                )
            )

        if frames:
            df = pd.concat(frames, ignore_index=True)
        else:
            df = pd.DataFrame(
                columns=["x", "y", "z", "E", "ux", "uy", "uz", "t_bunchx", "w", "volume"]
            )
        df.attrs = {
            "nuflavor": sim.nuflavor,
            "detector": self.name,
            "total_interactions": float(df["w"].sum()) if len(df) else 0.0,
        }
        return df


def uniform_hydrogen_cylinder(
    distance_cm=1e5, length_cm=10e2, diameter_cm=2e2, density_g_cm3=1.0
):
    """Benchmark detector: a uniform cylinder of hydrogen at 1 g/cm^3.

    Defaults: 10 m long, 2 m diameter, front face 1 km downstream of the IP,
    on the beam axis.
    """
    hydrogen = Material(density_g_cm3, 1.008, 1, 1)
    volume = CylinderVolume(
        hydrogen,
        radius=diameter_cm / 2.0,
        length=length_cm,
        center=(0.0, 0.0, distance_cm + length_cm / 2.0),
        name="hydrogen",
    )
    return VolumeStack([volume], name="uniform_hydrogen_cylinder")
