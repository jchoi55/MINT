"""Forward detectors for MINT.

A :class:`Detector` is a stack of coaxial modules on the beam axis, together
with the upstream rock column and air gap that shield it. It knows its own
geometry (apertures, chords, column densities) and can turn a
:class:`mint.MuC.MuDecaySimulator` sample into a flux or an interaction rate.

The benchmark geometry used throughout the MINT papers is available directly::

    import mint

    det = mint.detectors.benchmark          # the default, 5 km from the IP
    E, w = det.face_flux(sim, exposure=injections_per_year)
    rates = det.signal_interactions(sim, nuflavor="numubar", exposure=ipy)

Other distances are registered by name, and any constructor argument can be
overridden on the fly::

    det = mint.detectors.load("benchmark_1km")
    det = mint.detectors.load("benchmark_5km", n_sigma=3.0)

To describe your own detector, build a :class:`Detector` with your own module
list rather than subclassing -- see ``dev_examples/benchmark_detector.ipynb``.
"""


from dataclasses import dataclass

import numpy as np

from mint import detector_tools as dt

M_MU = 0.1056583755          # GeV

# Everything upstream of the calorimeter: the region in which an interaction
# can be reconstructed rather than merely absorbed.
TRACKING_KINDS = ("tracker", "vertex", "gas", "structure")

# The SIGNAL volume for neutrino scattering: the vertex tracker and the argon.
# A neutrino interaction counts only if it happens in one of these. The TPC
# vessel walls are deliberately excluded even though they carry more column
# than the gas -- a vertex inside a pressure wall has no upstream tracking and
# cannot be reconstructed -- as are the halo monitor, the calorimeter and the
# spectrometer air.
SIGNAL_KINDS = ("vertex", "gas")

# The fiducial volume for DECAYS of long-lived states is the gas alone: a decay
# vertex is found from where the ionisation starts, which needs the low-density
# tracking medium. This is NOT the same as SIGNAL_KINDS, and the difference
# matters -- see fiducial_volumes() vs signal_volumes().
DECAY_KINDS = ("gas",)


@dataclass
class Module:
    """One element of the detector stack.

    ``z0``/``z1`` are cm measured from the detector front face. ``aperture`` is
    ``"cone"`` (follows R(z)), ``"fixed"`` (constant ``r_ref``),
    ``"floor"`` (``max(R(z), r_ref)``) or ``"cap"`` (``min(R(z), r_ref)``).
    ``kind`` groups modules for reporting and for selecting which volumes take
    part in a rate calculation.
    """

    name: str
    z0: float
    z1: float
    material: object
    kind: str
    aperture: str = "cone"
    r_ref: float = 0.0
    color: str = "0.6"
    note: str = ""

    @property
    def thickness(self):
        """Longitudinal extent of the module [cm]."""
        return self.z1 - self.z0


class Detector:
    """Benchmark forward detector: conical aperture, segmented module stack.

    Parameters (lengths in cm unless noted)
    ---------------------------------------
    dist : distance from the IP to the front face.
    n_sigma : spot-size containment factor for the aperture (2.5 -> 96%).
    theta_acc : final-state acceptance half-angle [rad].
    sigma_div : RMS divergence of the parent muon beam [rad]. The default
        1.0e-4 reproduces the benchmark aperture table; it can be measured
        directly from a placed simulation with :meth:`fit_sigma_div`.
    E_beam : muon beam energy [GeV], only through the 1/gamma opening.
    gap, rock, rock_start : the beam-dump surroundings, as
        :class:`mint.detectors.ForwardDetector`.
    n_plates, plate_pitch, foil_thick, si_thick : vertex-tracker stack, smeared
        into one effective medium. The slab runs from ``Z_VTX0`` for
        ``n_plates * plate_pitch``.
    r_vertex_max : hard cap on the vertex-tracker half-width [cm]. The silicon
        area is ``n_plates * pi * r^2``, so this is the cost knob.
    tpc_pressure_bar : argon pressure; the gas density scales with it.
    """

    # ---- fixed layout constants [cm] --------------------------------------
    Z_HALO0, Z_HALO1 = 0.0, 200.0
    Z_VTX0 = 200.0               # vertex tracker starts here; its length is
    Z_DRIFT1 = 1200.0            # n_plates x plate_pitch. Air out to Z_DRIFT1.
    Z_TPC0, Z_TPC1 = 1200.0, 5200.0
    N_TPC_MODULES = 4
    TPC_MODULE_LEN = 950.0
    TPC_GAP = 50.0
    Z_CALO0, Z_CALO1 = 5250.0, 5550.0
    Z_MUON0, Z_MUON1 = 5550.0, 7550.0

    HALO_COLUMN = 0.4            # g/cm^2 of tracker material on the front face
    VESSEL_X0 = 0.20             # radiation lengths per internal TPC boundary
    ENDCAP_X0 = 0.30             # radiation lengths of the downstream end cap
    CALO_R_FLOOR = 300.0         # calorimeter / spectrometer minimum half-width
    HALO_RADIUS = 200.0

    # EM: 40 x (2.8 mm W + 0.5 mm Si + 3 mm gap); HAD: 84 x (2 cm Fe + 1 cm sc)
    EM_LAYERS, EM_W, EM_SI, EM_GAP = 40, 0.28, 0.05, 0.30
    HAD_LAYERS, HAD_FE, HAD_SC = 84, 2.0, 1.0

    def __init__(self, name="benchmark_5km", dist=5e5, n_sigma=2.5,
                 theta_acc=15e-3, sigma_div=1.0e-4, E_beam=5000.0,
                 gap=1000.0, rock=None, rock_start=250e2,
                 n_plates=24, plate_pitch=5.0, foil_mat=None, foil_thick=0.2,
                 si_thick=0.015, r_vertex_max=50.0,
                 n_sparse=4, sparse_thick=0.03,
                 tpc_pressure_bar=5.0, vessel_mat=None):
        self.name = name
        self.dist = float(dist)
        self.n_sigma = float(n_sigma)
        self.theta_acc = float(theta_acc)
        self.sigma_div = float(sigma_div)
        self.E_beam = float(E_beam)
        self.gap = float(gap)
        self.rock = rock if rock is not None else dt.standard_rock
        self.rock_start = float(rock_start)

        # -- vertex tracker: a homogenised slab, not n discrete cassettes.
        # Its job is charm: D0 and D+ have gamma c tau of order 1-15 cm at these
        # energies, so what matters is a few-cm sampling pitch and a small
        # impact-parameter resolution -- not target mass, which the gas supplies
        # far more cheaply. It is therefore kept thin and, crucially, NARROW:
        # silicon area is the cost driver, and covering the full acceptance cone
        # would need ~10^3 m^2. Everything else lives in the gas.
        self.n_plates = int(n_plates)
        self.plate_pitch = float(plate_pitch)
        self.foil_mat = foil_mat if foil_mat is not None else dt.graphite
        self.foil_thick = float(foil_thick)
        self.si_thick = float(si_thick)
        self.r_vertex_max = float(r_vertex_max)
        # Sparse tracker: a few large-area silicon layers spanning the drift
        # between the compact vertex tracker and the gas. They give the lever
        # arm that the 1.2 m vertex stack cannot, at negligible material
        # (4 x 300 um Si = 0.28 g/cm^2, 0.013 X0), and they are TRACKING rather
        # than target -- kind "tracker", so they stay out of SIGNAL_KINDS and
        # change no interaction rate.
        self.n_sparse = int(n_sparse)
        self.sparse_thick = float(sparse_thick)
        self.vessel_mat = vessel_mat if vessel_mat is not None else dt.graphite
        self.tpc_pressure_bar = float(tpc_pressure_bar)
        # ideal gas: rho scales with pressure off the tabulated 5 bar point
        # (X0 and lambda_I are per unit mass, so they carry over unchanged)
        self.gas = dt.GAr_5bar.scaled(self.tpc_pressure_bar / 5.0)

        # effective medium of the vertex slab: foil + sensor smeared over the
        # pitch, so one volume reproduces the column, X0 and lambda_I of the
        # stack exactly while placing vertices uniformly along z
        f_foil = self.foil_thick / self.plate_pitch
        f_si = self.si_thick / self.plate_pitch
        self.vertex_mat = dt.CompositMaterial([[self.foil_mat, f_foil],
                                               [dt.Si, f_si]])

        # homogenised calorimeter mixtures (volume fractions)
        em_pitch = self.EM_W + self.EM_SI + self.EM_GAP
        self.ecal_mat = dt.CompositMaterial([[dt.W, self.EM_W / em_pitch],
                                             [dt.Si, self.EM_SI / em_pitch]])
        had_pitch = self.HAD_FE + self.HAD_SC
        self.hcal_mat = dt.CompositMaterial([[dt.Fe, self.HAD_FE / had_pitch],
                                             [dt.PS, self.HAD_SC / had_pitch]])
        self.halo_mat = dt.PS
        self.modules = self._build_modules()

    # ---- vertex tracker -----------------------------------------------------
    @property
    def Z_VTX1(self):
        """Downstream face of the vertex tracker [cm]."""
        return self.Z_VTX0 + self.n_plates * self.plate_pitch

    @property
    def vertex_radius(self):
        """Half-width of the vertex tracker [cm] = min(R(z), r_vertex_max)."""
        return min(float(self.aperture(self.Z_VTX1)), self.r_vertex_max)

    @property
    def silicon_area(self):
        """Total silicon area of the vertex tracker [m^2] -- the cost driver."""
        return self.n_plates * np.pi * (self.vertex_radius / 100.0) ** 2

    # ---- aperture ----------------------------------------------------------
    @property
    def sigma_spot(self):
        """RMS neutrino spot size at the front face [cm]."""
        return self.dist * np.sqrt(self.sigma_div**2 + (M_MU / self.E_beam) ** 2)

    def aperture(self, z):
        """Detector half-width [cm] at ``z`` cm from the front face."""
        return (self.n_sigma * self.sigma_spot
                + np.asarray(z, float) * np.tan(self.theta_acc))

    @property
    def radius(self):
        """Front-face half-width -- the acceptance radius for face selections."""
        return float(self.aperture(0.0))

    @property
    def radius_back(self):
        """Aperture radius at the downstream end of the detector [cm]."""
        return float(self.aperture(self.Z_MUON1))

    @property
    def length(self):
        """Full instrumented length [cm]."""
        return self.Z_MUON1 - self.Z_HALO0

    @property
    def fiducial_length(self):
        """Active TPC gas length [cm] -- the decay volume."""
        return self.N_TPC_MODULES * self.TPC_MODULE_LEN

    # ---- layout ------------------------------------------------------------
    def _tpc_module_edges(self):
        """[(z0, z1), ...] of the active gas modules, on a 10 m pitch."""
        pitch = self.TPC_MODULE_LEN + self.TPC_GAP
        return [(self.Z_TPC0 + i * pitch, self.Z_TPC0 + i * pitch + self.TPC_MODULE_LEN)
                for i in range(self.N_TPC_MODULES)]

    def _build_modules(self):
        mods = []
        C = {"halo": "#8c8c8c", "target": "#3b6ea5", "si": "#6fa8dc",
             "gas": "#c7e0b4", "vessel": "#b07aa1", "ecal": "#d1495b",
             "hcal": "#8d6e4a", "air": "#eef2f6"}

        # -- 0. halo tagger: a thin tracking plane over the full face.
        # Not a target; sized to the rock-muon halo, hence the fixed 2 m radius
        # rather than the neutrino-cone aperture.
        t_halo = self.HALO_COLUMN / self.halo_mat.density
        mods.append(Module("halo tagger", self.Z_HALO0, self.Z_HALO0 + t_halo,
                           self.halo_mat, "tracker", "fixed", self.HALO_RADIUS,
                           C["halo"], "0.4 g/cm2 fibre/gas tracking, full face"))

        # -- 1. vertex tracker: ONE homogenised slab of graphite + silicon,
        # radius capped at r_vertex_max so the silicon area stays affordable.
        mods.append(Module("vertex tracker", self.Z_VTX0, self.Z_VTX1,
                           self.vertex_mat, "vertex", "cap", self.r_vertex_max,
                           C["target"],
                           f"{self.n_plates} plates x ({self.foil_thick*10:.0f} mm C"
                           f" + {self.si_thick*1e4:.0f} um Si) on a "
                           f"{self.plate_pitch:.0f} cm pitch"))

        # -- 1b. sparse tracker: n large-area Si layers across the drift
        if self.n_sparse > 0:
            span = self.Z_DRIFT1 - self.Z_VTX1
            for i in range(self.n_sparse):
                z = self.Z_VTX1 + (i + 0.5) * span / self.n_sparse
                mods.append(Module(f"sparse tracker {i}", z, z + self.sparse_thick,
                                   dt.Si, "tracker", "cone", 0.0, C["si"],
                                   f"{self.sparse_thick*1e4:.0f} um Si layer"))

        # -- 2. TPC core: active gas modules separated by pressure-vessel walls
        edges = self._tpc_module_edges()
        t_wall = self.VESSEL_X0 * dt.radiation_length(self.vessel_mat) \
            / self.vessel_mat.density
        for i, (za, zb) in enumerate(edges):
            mods.append(Module(f"TPC module {i}", za, zb, self.gas, "gas",
                               "cone", 0.0, C["gas"],
                               f"{self.tpc_pressure_bar:.0f} bar Ar, "
                               f"{(zb-za)/100:.1f} m"))
            if i < len(edges) - 1:                      # internal boundary
                zc = 0.5 * (zb + edges[i + 1][0])
                mods.append(Module(f"TPC wall {i}", zc - t_wall / 2,
                                   zc + t_wall / 2, self.vessel_mat, "structure",
                                   "cone", 0.0, C["vessel"],
                                   f"{self.VESSEL_X0:.2f} X0 vessel wall"))
        t_cap = self.ENDCAP_X0 * dt.radiation_length(self.vessel_mat) \
            / self.vessel_mat.density
        mods.append(Module("TPC end cap", self.Z_TPC1 - t_cap, self.Z_TPC1,
                           self.vessel_mat, "structure", "cone", 0.0,
                           C["vessel"], f"{self.ENDCAP_X0:.2f} X0 end cap"))

        # -- 3. calorimeter (homogenised: MINT never resolves shower layers)
        z_em1 = self.Z_CALO0 + self.EM_LAYERS * (self.EM_W + self.EM_SI + self.EM_GAP)
        mods.append(Module("ECAL", self.Z_CALO0, z_em1, self.ecal_mat, "absorber",
                           "floor", self.CALO_R_FLOOR, C["ecal"],
                           f"{self.EM_LAYERS} x (W/Si), 32 X0"))
        z_h0 = z_em1 + 5.0
        z_h1 = z_h0 + self.HAD_LAYERS * (self.HAD_FE + self.HAD_SC)
        mods.append(Module("HCAL", z_h0, z_h1, self.hcal_mat, "absorber",
                           "floor", self.CALO_R_FLOOR, C["hcal"],
                           f"{self.HAD_LAYERS} x (Fe/scint), 10 lambda_I"))

        # -- 4. muon spectrometer: air-core dipole, tracking stations only
        mods.append(Module("muon spectrometer", self.Z_MUON0, self.Z_MUON1,
                           dt.Air, "air", "floor", self.CALO_R_FLOOR, C["air"],
                           "air-core dipole, 0.5 T over 20 m"))
        return mods

    @property
    def targets(self):
        """[(z_from_face, thickness, material), ...] -- the dense production
        elements. The benchmark has exactly one: the vertex tracker. Kept in
        the same shape the legacy detector used so that code iterating over
        "the target slabs" keeps working."""
        return [(self.Z_VTX0, self.Z_VTX1 - self.Z_VTX0, self.vertex_mat)]

    # ---- volumes -----------------------------------------------------------
    def _segments(self, mod, dist=None):
        """[(z0, z1, r0, r1), ...] world-z segments realising the aperture mode.

        ``floor`` splits at the crossing R(z) = r_ref when it falls inside the
        module, so ``max(R, r_ref)`` is represented exactly rather than by a
        cone through the two end points.
        """
        D = self.dist if dist is None else dist
        scale = D / self.dist                      # spot size scales with L

        def r_at(z):
            return (self.n_sigma * self.sigma_spot * scale
                    + z * np.tan(self.theta_acc))

        z0, z1 = mod.z0, mod.z1
        if mod.aperture == "fixed":
            return [(z0, z1, mod.r_ref, mod.r_ref)]
        if mod.aperture == "cone":
            return [(z0, z1, float(r_at(z0)), float(r_at(z1)))]
        ra, rb = float(r_at(z0)), float(r_at(z1))
        F = mod.r_ref
        if mod.aperture == "cap":                  # min(R(z), r_ref)
            if ra >= F:
                return [(z0, z1, F, F)]
            if rb <= F:
                return [(z0, z1, ra, rb)]
            z_star = z0 + (F - ra) / np.tan(self.theta_acc)
            return [(z0, z_star, ra, F), (z_star, z1, F, F)]
        # "floor": max(R(z), r_ref)
        if ra >= F and rb >= F:
            return [(z0, z1, ra, rb)]
        if ra <= F and rb <= F:
            return [(z0, z1, F, F)]
        z_star = z0 + (F - ra) / np.tan(self.theta_acc)
        return [(z0, z_star, F, F), (z_star, z1, F, rb)]

    def volumes(self, sign=+1, dist=None, kinds=None):
        """Ray-traceable volumes in world coordinates.

        ``kinds`` restricts to a subset of module kinds (e.g. ``("gas",)`` for
        the decay volume, ``("target", "gas", "silicon")`` for the tracker).
        """
        D = self.dist if dist is None else dist
        out = []
        for m in self.modules:
            if kinds is not None and m.kind not in kinds:
                continue
            for (a, b, r0, r1) in self._segments(m, dist=D):
                if b <= a:
                    continue
                out.append(dt.ConeVolume(m.material, z0=sign * D + sign * a,
                                         z1=sign * D + sign * b, r0=r0, r1=r1,
                                         name=m.name)
                           if sign > 0 else
                           dt.ConeVolume(m.material, z0=-(D + b), z1=-(D + a),
                                         r0=r1, r1=r0, name=m.name))
        return out

    def detector(self, sign=+1, dist=None, kinds=None):
        """A :class:`mint.detector_tools.VolumeStack` over the module stack."""
        return dt.VolumeStack(self.volumes(sign=sign, dist=dist, kinds=kinds),
                           name=self.name)

    def fiducial_volumes(self, sign=+1, dist=None):
        """Active TPC gas cones -- the DECAY volume for long-lived states."""
        return self.volumes(sign=sign, dist=dist, kinds=DECAY_KINDS)

    def signal_volumes(self, sign=+1, dist=None):
        """Vertex tracker + TPC gas -- the volume in which a neutrino
        interaction counts as signal."""
        return self.volumes(sign=sign, dist=dist, kinds=SIGNAL_KINDS)

    def signal_column(self, kinds=SIGNAL_KINDS):
        """Nucleon column of the signal volume [1/cm^2]."""
        return sum(m.material.N * m.thickness for m in self.modules
                   if m.kind in kinds)

    def signal_interactions(self, sim, nuflavor=None, sign=+1, exposure=1.0,
                            dist=None, xsec=None, kinds=SIGNAL_KINDS):
        """Neutrino interactions per exposure in the signal volume.

        Sums per volume, since each carries its own nucleon density. ``xsec``
        defaults to the MINT total (CC+NC) for the simulation's flavour and is
        applied per nucleon; the non-isoscalar correction is NOT folded in here
        (use :meth:`cc_correction` if a CC-only rate needs it).
        """
        from mint import xsecs as _xs
        flavor = nuflavor or sim.nuflavor
        rays = dt.sim_rays(sim)
        E = np.asarray(sim.pnu["E"])
        w = sim.weights.flatten() * exposure
        sig = _xs.total_xsecs[flavor](E) if xsec is None else np.asarray(xsec(E))
        out = {}
        for v in self.volumes(sign=sign, dist=dist, kinds=kinds):
            _, ch = v.intersect(*rays)
            out[v.name] = float((w * sig * v.material.N * ch).sum())
        out["total"] = float(sum(out.values()))
        return out

    def volume(self, material=None, sign=+1, dist=None):
        """Single cone spanning the whole TPC core, for quick estimates.

        Filled with the gas unless ``material`` is given. Prefer
        :meth:`fiducial_volumes` when the 0.5 m inter-module gaps matter.
        """
        D = self.dist if dist is None else dist
        mat = material if material is not None else self.gas
        segs = self._segments(Module("tpc", self.Z_TPC0, self.Z_TPC1, mat, "gas"),
                              dist=D)
        a, b, r0, r1 = segs[0]
        if sign > 0:
            return dt.ConeVolume(mat, z0=D + a, z1=D + b, r0=r0, r1=r1, name="TPC")
        return dt.ConeVolume(mat, z0=-(D + b), z1=-(D + a), r0=r1, r1=r0, name="TPC")

    # ---- composition -------------------------------------------------------
    def column(self, kinds=None, materials=None):
        """On-axis column density [g/cm^2] of the selected modules."""
        tot = 0.0
        for m in self.modules:
            if kinds is not None and m.kind not in kinds:
                continue
            if materials is not None and m.material not in materials:
                continue
            tot += m.material.density * m.thickness
        return tot

    def nucleon_column(self, kinds=None):
        """On-axis nucleon column [1/cm^2]. Defaults to the SIGNAL volume
        (vertex tracker + gas) -- the material in which a neutrino interaction
        counts. Pass ``kinds=TRACKING_KINDS`` for everything upstream of the
        calorimeter, or an explicit tuple for anything else."""
        kinds = SIGNAL_KINDS if kinds is None else kinds
        return sum(m.material.N * m.thickness for m in self.modules
                   if m.kind in kinds)

    def electron_column(self, kinds=None):
        """On-axis electron column [1/cm^2] of the signal volume by default."""
        kinds = SIGNAL_KINDS if kinds is None else kinds
        return sum(m.material.e * m.thickness for m in self.modules
                   if m.kind in kinds)

    def radiation_lengths(self, kinds=None):
        """On-axis X0 budget of the selected modules."""
        tot = 0.0
        for m in self.modules:
            if kinds is not None and m.kind not in kinds:
                continue
            X0 = dt.radiation_length(m.material)
            if np.isfinite(X0):
                tot += m.material.density * m.thickness / X0
        return tot

    def interaction_lengths(self, kinds=None):
        """Hadronic interaction lengths along the axis, summed over modules."""
        tot = 0.0
        for m in self.modules:
            if kinds is not None and m.kind not in kinds:
                continue
            lam = dt.interaction_length(m.material)
            if np.isfinite(lam):
                tot += m.material.density * m.thickness / lam
        return tot

    def column_table(self):
        """Per-group summary: thickness, column, X0 and lambda_I."""
        groups = [("halo tagger", ("tracker",)), ("vertex tracker", ("vertex",)),
                  ("TPC gas", ("gas",)), ("TPC structure", ("structure",)),
                  ("calorimeter", ("absorber",)), ("muon spectrometer", ("air",))]
        rows = []
        for label, kinds in groups:
            ms = [m for m in self.modules if m.kind in kinds]
            if not ms:
                continue
            rows.append({
                "group": label,
                "z0 [m]": min(m.z0 for m in ms) / 100,
                "z1 [m]": max(m.z1 for m in ms) / 100,
                "material [cm]": sum(m.thickness for m in ms),
                "column [g/cm2]": self.column(kinds=kinds),
                "X0": self.radiation_lengths(kinds=kinds),
                "lambda_I": self.interaction_lengths(kinds=kinds),
            })
        return rows

    def cc_correction(self, Enu, nuflavor, kinds=None):
        """Column-weighted non-isoscalar CC correction of the SIGNAL volume.

        Multiplies an isoscalar per-nucleon CC cross section. Tiny compared with
        the old tungsten-slab detector: graphite (Z/A = 0.50) and argon
        (Z/A = 0.45) are both close to isoscalar, so this stays within a
        per-cent of unity instead of the +5%/-4% the tungsten slabs produced.
        """
        from mint import xsecs
        kinds = SIGNAL_KINDS if kinds is None else kinds
        num, den = 0.0, 0.0
        for m in self.modules:
            if m.kind not in kinds:
                continue
            col = m.material.N * m.thickness
            if col <= 0:
                continue
            nuc = dt.dominant_nucleus(m.material)
            num = num + col * xsecs.cc_nonisoscalar_correction(
                Enu, nuflavor, nuc.Z, nuc.A)
            den = den + col
        return num / den if den > 0 else np.ones_like(np.asarray(Enu, float))

    # ---- surroundings ------------------------------------------------------
    def rock_length(self, dist=None):
        """Thickness of rock upstream of the face [cm].

        Zero if the detector sits at or inside where the rock begins.
        """
        D = self.dist if dist is None else dist
        return max(D - self.gap - self.rock_start, 0.0)

    @property
    def rock_column(self):
        """Nucleon column density of the upstream rock [1/cm^2]."""
        return self.rock_length()

    # ---- flux helpers (same signatures as ForwardDetector) -----------------
    def face_rays(self, sim, sign=+1, r_sel=None, E_min=0.0, exposure=1.0,
                  dist=None):
        """Per-ray kinematics on the front-face plane, mirrored for the
        upstream (mu-) detector. ``r_sel`` defaults to the front-face aperture."""
        D = self.dist if dist is None else dist
        if r_sel is None:
            r_sel = self.radius * (D / self.dist)
        (ox, oy, oz), (vx, vy, vz) = dt.sim_rays(sim)
        E = np.asarray(sim.pnu["E"])
        w = sim.weights.flatten() * exposure
        if sign < 0:
            oz, vz = -oz, -vz
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (D - oz) / vz
        rx, ry = ox + t * vx, oy + t * vy
        m = (t > 0) & (E > E_min) & (rx**2 + ry**2 < r_sel**2)
        return E[m], w[m], rx[m], ry[m], (vx / vz)[m], (vy / vz)[m]

    def face_flux(self, sim, sign=+1, exposure=1.0, E_min=0.0, dist=None):
        """(E, weight) of the neutrinos crossing the front face.

        A convenience wrapper over :meth:`face_rays` that drops the positions
        and angles. Multiply ``exposure`` by injections per year to get a rate.
        """
        E, w, *_ = self.face_rays(sim, sign=sign, E_min=E_min,
                                  exposure=exposure, dist=dist)
        return E, w

    def flux_with_chords(self, sim, sign=+1, exposure=1.0, material=None,
                         dist=None, kinds=SIGNAL_KINDS):
        """(E, w, chord): rays crossing the selected modules, with the TOTAL
        chord summed over them.

        Defaults to the SIGNAL volume (vertex tracker + gas). Pass
        ``kinds=DECAY_KINDS`` for the gas alone, or ``kinds=None`` for every
        volume in the detector. Note that with a heterogeneous selection the
        chord alone is not enough to get a rate -- the modules have different
        nucleon densities -- so use :meth:`signal_interactions` unless you are
        deliberately treating the selection as one material."""
        rays = dt.sim_rays(sim)
        vols = self.volumes(sign=sign, dist=dist, kinds=kinds)
        if material is not None:
            vols = [dt.ConeVolume(material, v.z0, v.z1, v.r0, v.r1, name=v.name)
                    for v in vols]
        ch = np.zeros(np.asarray(sim.pnu["E"]).shape)
        for v in vols:
            ch = ch + v.intersect(*rays)[1]
        m = ch > 0
        E = np.asarray(sim.pnu["E"])[m]
        w = sim.weights.flatten()[m] * exposure
        return E, w, ch[m]

    # ---- density map -------------------------------------------------------
    def fill_density(self, Z, R, rho, sign=+1, dist=None):
        """Paint the detector materials onto an (Z, R) density grid [g/cm^3].

        ``Z``/``R`` are world-frame arrays; ``rho`` is modified in place and
        returned. Used by :mod:`mint.beamline` for the beamline density map.
        """
        D = self.dist if dist is None else dist
        Zl = sign * np.asarray(Z, float) - D          # local z from the face
        Rr = np.asarray(R, float)
        for m in self.modules:
            for (a, b, r0, r1) in self._segments(m, dist=D):
                if b <= a:
                    continue
                k = (r1 - r0) / (b - a)
                inside = (Zl >= a) & (Zl < b) & (Rr <= r0 + k * (Zl - a))
                rho = np.where(inside, m.material.density, rho)
        return rho

    def __repr__(self):
        return (f"Detector({self.name!r}: {self.length/100:.1f} m long, "
                f"R = {self.radius/100:.2f}-{self.radius_back/100:.2f} m, "
                f"{self.fiducial_length/100:.0f} m of "
                f"{self.tpc_pressure_bar:.0f} bar Ar, vertex tracker "
                f"{self.n_plates}x{self.plate_pitch:.0f} cm r<{self.vertex_radius:.0f} cm"
                f" at {self.dist/1e5:.2f} km)")


# ---------------------------------------------------------------------------
# Named geometries
# ---------------------------------------------------------------------------

_REGISTRY = {
    "benchmark_5km": {},
    "benchmark_1km": {"dist": 1e5},
    "benchmark_250m": {"dist": 250e2},
    "benchmark_20km": {"dist": 20e5},
}

DEFAULT = "benchmark_5km"


def available():
    """Names accepted by :func:`load`."""
    return sorted(_REGISTRY)


def load(name=DEFAULT, **overrides):
    """Build a named detector; keyword arguments override its defaults.

    >>> det = load()                          # benchmark_5km
    >>> det = load("benchmark_1km")
    >>> det = load("benchmark_5km", n_sigma=3.0)
    """
    if name not in _REGISTRY:
        raise KeyError(f"unknown detector {name!r}; available: {available()}")
    kw = dict(_REGISTRY[name])
    kw.update(overrides)
    return Detector(name=name, **kw)


#: The benchmark forward detector at 5 km, ready to use.
benchmark = load(DEFAULT)
