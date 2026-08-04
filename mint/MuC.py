import warnings

import vector
import numpy as np

from mint import const
from mint import mudecay_tools as mudec
from mint import lattice_tools as lt


class MuDecaySimulator:
    """Muon decays along an accelerator lattice, and the neutrinos they make.

    The generator samples the muon rest-frame decay phase space once with vegas,
    then places those decays on a lattice: each muon gets a position along the
    ring, a momentum drawn from the local beam envelope, and a boost to the lab
    frame. What comes out is a weighted sample of neutrinos, with the weights
    normalised so they sum to the number of muon decays per injection.

    Typical use::

        ring = mint.lattices.load("mc_10tev_hybrid_v06")
        sim = MuDecaySimulator(muon_polarization=0.0, lattice=ring,
                               nuflavor="numubar", n_evals=1e5)
        sim.decay_muons()
        sim.place_muons_on_lattice(lattice=ring, direction="clockwise")

    Because the rest-frame sample is flavor- and polarization-independent, any
    other flavor is an exact reweighting of the same events rather than a new
    vegas run::

        sim_nue = sim.reweighted_copy(nuflavor="nue")
        sim_nue.place_muons_on_lattice(lattice=ring, direction="clockwise")

    That is what makes :func:`mint.beams.both_beams` cheap: one generation,
    four flavors. Use :meth:`save_events` / :meth:`load_events` to reuse a
    sample across sessions.
    """

    def __init__(
        self,
        muon_polarization,
        lattice=None,
        nuflavor=None,
        cycles=1,
        direction="clockwise",
        n_evals=1e5,
        preloaded_events=None,
        remove_ring_fraction=0,
        NLO=True,
        mudecay_model="NLOmudecay_pol",
        beam_dynamics=True,
        aperture_nsigma=5.0,
    ):
        """
        Parameters
        ----------
        muon_polarization : float
            Longitudinal muon polarization, in [-1, 1]. 0 for an unpolarized beam.
        lattice : mint.lattice_tools.Lattice, optional
            The machine to place decays on. Required unless you only want
            rest-frame events, and required when ``cycles="full_injection"``.
        nuflavor : str, optional
            Which neutrino to keep: ``"numu"``, ``"numubar"``, ``"nue"`` or
            ``"nuebar"``. The mu+ beam makes numubar and nue; mu- the conjugates.
        cycles : float or str
            Turns each muon makes. ``"full_injection"`` computes the number of
            turns in one injection period from the lattice, which is what the
            standard setups use.
        direction : str
            ``"clockwise"`` or ``"counterclockwise"``. Sets which way the beam
            travels, and hence which detector sees it.
        n_evals : float
            vegas evaluations for the rest-frame sample. Raise for more
            statistics; everything else is a reweighting of it.
        preloaded_events : dict, optional
            A sample from :meth:`save_events`, to skip generation entirely.
        remove_ring_fraction : float or tuple
            Fraction of the ring to exclude from decay placement. A float
            removes a centred span; a tuple gives explicit start and end points.
            0 keeps the whole ring.
        NLO : bool
            Include the O(alpha) radiative corrections to muon decay.
        mudecay_model : str
            Matrix element to sample. The default is the polarized NLO one.
        beam_dynamics : bool
            If False, put every muon on the ideal closed orbit with no
            divergence, envelope or momentum spread -- useful for isolating what
            the optics contribute.
        aperture_nsigma : float
            Collimate the transverse beam at this many sigma, mirroring the
            machine's physical aperture. None keeps the untruncated Gaussian.
        """

        # Design contains all necessary inputs to specify the muon storage/accelerator

        self.lattice = lattice
        # Physical aperture: the machine collimates the beam at n sigma of its
        # transverse extent (the IMCC/MuCol shielding liners are cut to the
        # 5 sigma envelope, MuCol Milestone 15 Sec. 2.2), so muons are never
        # found in the far Gaussian tail. Set to None to keep untruncated
        # Gaussian envelopes. Note this is a tiny effect on the flux: a 2D
        # Gaussian truncated at 5 sigma retains 1 - exp(-25/2) = 1 - 3.7e-6
        # of the beam.
        self.aperture_nsigma = aperture_nsigma

        self.n_evals = n_evals
        self.preloaded_events = preloaded_events

        if cycles == "full_injection":
            # Simulate one full injection period: the number of machine turns the
            # muons make in 1/finj seconds. The machine circumference is
            # lattice.total_circumference (the lattice may cover only part of the
            # machine); for a full-ring lattice it defaults to the lattice length.
            # Muons are ultra-relativistic, so v = c to better than 1e-9.
            if self.lattice is None:
                raise ValueError(
                    "lattice must be provided when cycles='full_injection'."
                )
            C_machine = getattr(self.lattice, "total_circumference", None)
            if C_machine is None:
                C_machine = float(self.lattice.s(1))
            self.cycles = const.c_LIGHT / (self.lattice.finj * C_machine)
        else:
            self.cycles = cycles
        self.direction = direction

        if isinstance(remove_ring_fraction, float) or isinstance(
            remove_ring_fraction, int
        ):
            self.remove_ring_fraction = [
                (1 - remove_ring_fraction) / 2,
                (1 + remove_ring_fraction) / 2,
            ]
        elif isinstance(remove_ring_fraction, list) or isinstance(
            remove_ring_fraction, tuple
        ):
            self.remove_ring_fraction = [
                (1 - remove_ring_fraction[0]) / 2,
                (1 + remove_ring_fraction[1]) / 2,
            ]

        else:
            print("No remove_ring_fraction specified. Using all ring.")
            self.remove_ring_fraction = [0, 0]

        self.nuflavor = nuflavor
        if self.nuflavor in ("numu", "nuebar", "nutau"):
            # produced by mu- (mu- -> e- nu_mu nu_e-bar; exotic mu- -> e- nu_mu nu_tau)
            self.muon_charge = -1
        elif self.nuflavor in ("nue", "numubar", "nutaubar"):
            # produced by mu+ (mu+ -> e+ nu_e nu_mu-bar; exotic mu+ -> e+ nu_mu-bar nu_tau-bar)
            self.muon_charge = +1
        else:  # default antimuons
            self.muon_charge = +1

        self.muon_polarization = muon_polarization
        self.NLO = NLO
        self.mudecay_model = mudecay_model

        self.beam_dynamics = beam_dynamics

        if abs(self.muon_polarization) > 1:
            raise ValueError(
                f"muon_polarization must be between -1 and 1, found design['muon_polarization'] = {self.muon_polarization}"
            )

        if self.preloaded_events:
            for var in self.preloaded_events.keys():
                setattr(self, var, preloaded_events[var])

    def decay_muons(self):
        """
        Simulates the decay of muons and calculates their positions and velocities.

        If preloaded events are available, the function returns immediately.

        Otherwise, it simulates the decays with self.simulate_decays().

        Attributes:
            preloaded_events (bool): Flag indicating if events are preloaded.
            NLO (bool): Flag indicating if Next-to-Leading Order corrections are used.
            sample_size (int): Number of muon samples.
            pmu (np.ndarray): Array containing the momenta of the muons.
            pnu (np.ndarray): Array containing the momenta of the neutrinos.
            pos (np.ndarray): Array to store the positions of the muons.
            vmu (np.ndarray): Array to store the absolute velocities of the muons.
            momenta (np.ndarray): Array to store the momenta of the neutrinos.
            E (np.ndarray): Array to store the energies of the neutrinos.

        Returns:
            self: The instance of the class with updated attributes.
        """

        if self.preloaded_events:
            return self
        else:

            # Monte Carlo event generation
            Generator = mudec.GeneratorEngine(
                mudecay_model=self.mudecay_model,
                Mparent=const.m_mu,
                Mdaughter=const.m_e,
                NLO=self.NLO,
                muon_polarization=self.muon_polarization,
                muon_charge=self.muon_charge,
                nuflavor=self.nuflavor,
                NINT=20,
                NINT_warmup=10,
                NEVAL=self.n_evals,
                NEVAL_warmup=self.n_evals / 10,
            )

            event_dict = Generator.get_events()  # gets all events for +1 helicity

            # Muon 4-momenta -- at rest
            self.pmu_restframe = event_dict["P_decay_mu"]  # all muon decaying momenta

            # Neutrino 4-momenta
            self.pnu_restframe = event_dict["P_decay_nu"]

            # rest frame variables in integration for ease of reweighting
            self.x_CM = event_dict["x_CM"]
            self.costheta_CM = event_dict["costheta_CM"]

            # Normalized weights. Adds up to 1 for a single muon decay.
            self.weights = event_dict["w_flux"] / np.sum(event_dict["w_flux"])
            self.weights = self.weights[:, np.newaxis]

            # Pristine decay weights (before any lattice/exposure factors) --
            # used to reset the weights on (re-)placement and for reweighting.
            self.weights_decay = self.weights.copy()

            # number of events simulated
            self.sample_size = self.pmu_restframe.size

            # Muon decay position (initialized to 0)
            self.pos = vector.array(
                {
                    "x": np.zeros(self.sample_size),
                    "y": np.zeros(self.sample_size),
                    "z": np.zeros(self.sample_size),
                }
            )

    def reweight_with_new_polarization(self, new_polarization):

        numerator = mudec.mudecay_matrix_element_sqr(
            self.x_CM,
            self.costheta_CM,
            new_polarization,
            self.muon_charge,
            self.nuflavor,
            self.NLO,
        )
        denominator = mudec.mudecay_matrix_element_sqr(
            self.x_CM,
            self.costheta_CM,
            self.muon_polarization,
            self.muon_charge,
            self.nuflavor,
            self.NLO,
        )
        # NOTE: Changing polarization does not change the total integral, so this is a safe operation
        self.weights_new_pol = self.weights.flatten() * numerator / denominator

        return self.weights_new_pol

    @staticmethod
    def _muon_charge_for(nuflavor):
        """Muon charge implied by the requested neutrino flavor (same convention
        as __init__): numu/nuebar/nutau from mu-, nue/numubar/nutaubar from mu+."""
        if nuflavor in ("numu", "nuebar", "nutau"):
            return -1
        return +1

    def reweighted_copy(self, nuflavor=None, muon_polarization=None, NLO=None):
        """Return a new simulator that REUSES this simulator's rest-frame decay
        sample (no new vegas run), reweighted to a different neutrino flavor,
        muon polarization, and/or NLO setting.

        The muon decay phase space is fully described by the vegas variables
        (x = 2 E_nu / m_mu, costheta); flavor, polarization and NLO only change
        the matrix element, so the exact reweighting factor is the ratio of
        matrix elements evaluated on the stored sample. The reweighted weights
        are renormalized to sum to 1 (each muon decay produces exactly one
        neutrino of each flavor), matching the convention of decay_muons().

        Call place_muons_on_lattice() on the returned copy as usual. Lattice
        placement (positions, boosts, decay-in-flight weights) is redone per
        copy; only the vegas event generation is shared.
        """
        import copy as _copy

        if not hasattr(self, "weights_decay"):
            raise RuntimeError(
                "No decay sample available. Run decay_muons() (or load_events) first."
            )

        new = _copy.copy(self)
        new.nuflavor = self.nuflavor if nuflavor is None else nuflavor
        new.muon_polarization = (
            self.muon_polarization if muon_polarization is None else muon_polarization
        )
        new.NLO = self.NLO if NLO is None else NLO
        new.muon_charge = self._muon_charge_for(new.nuflavor)

        numerator = mudec.mudecay_matrix_element_sqr(
            self.x_CM,
            self.costheta_CM,
            new.muon_polarization,
            new.muon_charge,
            new.nuflavor,
            new.NLO,
        )
        denominator = mudec.mudecay_matrix_element_sqr(
            self.x_CM,
            self.costheta_CM,
            self.muon_polarization,
            self.muon_charge,
            self.nuflavor,
            self.NLO,
        )

        w = self.weights_decay.flatten() * numerator / denominator
        w = w / np.sum(w)  # exactly one neutrino of this flavor per muon decay
        new.weights_decay = w[:, np.newaxis]
        new.weights = new.weights_decay.copy()

        return new

    def save_events(self, filename):
        """Save the rest-frame decay sample to a compressed .npz file.

        Stores the vegas phase-space variables (x, costheta), the neutrino
        azimuth, the normalized decay weights, and the generation settings, so
        the (expensive) vegas run never needs to be repeated: reload with
        MuDecaySimulator.load_events() and reweight to any flavor/polarization.
        """
        if not hasattr(self, "weights_decay"):
            raise RuntimeError("No decay sample to save. Run decay_muons() first.")
        np.savez_compressed(
            filename,
            x_CM=self.x_CM,
            costheta_CM=self.costheta_CM,
            phi_nu=np.asarray(self.pnu_restframe.phi),
            weights_decay=self.weights_decay.flatten(),
            muon_polarization=self.muon_polarization,
            muon_charge=self.muon_charge,
            nuflavor=str(self.nuflavor),
            NLO=self.NLO,
            mudecay_model=str(self.mudecay_model),
        )

    @classmethod
    def load_events(cls, filename, **kwargs):
        """Create a simulator from a decay sample saved with save_events().

        Rebuilds the rest-frame four-momenta from (x, costheta, phi) and restores
        the stored weights -- no vegas run. Extra keyword arguments (lattice,
        cycles, beam_dynamics, remove_ring_fraction, ...) are forwarded to the
        constructor. Use reweighted_copy() afterwards for other flavors or
        polarizations.
        """
        data = np.load(filename, allow_pickle=False)

        sim = cls(
            muon_polarization=float(data["muon_polarization"]),
            nuflavor=str(data["nuflavor"]),
            NLO=bool(data["NLO"]),
            mudecay_model=str(data["mudecay_model"]),
            **kwargs,
        )

        x = np.asarray(data["x_CM"], dtype=float)
        costheta = np.asarray(data["costheta_CM"], dtype=float)
        phi = np.asarray(data["phi_nu"], dtype=float)

        sim.x_CM = x
        sim.costheta_CM = costheta
        sim.sample_size = x.size

        # Muon at rest; neutrino energy E = x m_mu / 2 with polar angle costheta
        # (same construction as mudecay_tools.three_body_decay_x_costheta).
        sim.pmu_restframe = vector.array(
            {
                "E": np.full(sim.sample_size, const.m_mu),
                "px": np.zeros(sim.sample_size),
                "py": np.zeros(sim.sample_size),
                "pz": np.zeros(sim.sample_size),
            }
        )
        Enu = const.m_mu / 2 * x
        sim.pnu_restframe = vector.array(
            {
                "E": Enu,
                "pt": Enu * np.sqrt(1 - costheta**2),
                "pz": Enu * costheta,
                "phi": phi,
            }
        )

        sim.weights_decay = np.asarray(data["weights_decay"], dtype=float)[
            :, np.newaxis
        ]
        sim.weights = sim.weights_decay.copy()
        sim.pos = vector.array(
            {
                "x": np.zeros(sim.sample_size),
                "y": np.zeros(sim.sample_size),
                "z": np.zeros(sim.sample_size),
            }
        )
        return sim

    def place_muons_on_lattice(self, lattice=None, direction="clockwise"):
        """Place muon decays on the storage-ring central orbit and rotate their
        momenta into the world (lab) frame.

        World-coordinate convention (identical in the lab frame and the
        beam-comoving frame):
            x -- toward the center of the ring (+x points to the center),
            y -- up, out of the ring plane (+y above the plane),
            z -- tangential, along the beam direction of travel.
        Momenta px, py, pz follow the same convention.

        The lattice returns world coordinates directly, so placement is a near
        identity: `pos = (lattice.x, lattice.y, lattice.z)(u)` plus transverse
        beam-envelope offsets, and momenta are rotated from the local beam frame
        (horizontal, vertical, tangential) onto the local world tangent.

        lattice: an lt.Lattice (or dict) describing the central orbit. Each field
            is a smooth function of a parameter `u` in [0, 1] (0 = IP, 1 = IP again
            after a full orbit):
                lattice.x/y/z(u): world position of the central orbit,
                lattice.s(u):     arc length along the orbit,
                lattice.inv_s(s): inverse of s(u),
                lattice.tangent(u): world unit tangent [tx, ty, tz] of the orbit,
                lattice.angle_of_central_p(u): tangent angle in the horizontal
                    z-x plane (for plotting only).

        direction: "clockwise" (reference beam, travels along +tangent) or
            "counter-clockwise" (counter-circulating beam, same orbit, momenta
            reversed to travel along -tangent).
        """

        if isinstance(lattice, dict):
            lattice = lt.Lattice(**lattice)
        elif lattice is None:
            if hasattr(self, "lattice"):
                lattice = self.lattice
            else:
                raise ValueError(
                    "No lattice was set. Please provide a dictionary that describes the lattice or pass a lt.Lattice object."
                )

        # Muon decay position (initialized to origin)
        self.pos = vector.array(
            {
                "x": np.zeros(self.sample_size),
                "y": np.zeros(self.sample_size),
                "z": np.zeros(self.sample_size),
            }
        )

        # Start from the pristine decay weights so that placement is idempotent:
        # calling place_muons_on_lattice again (or on a reweighted copy) does not
        # stack exposure/decay factors from a previous placement.
        if hasattr(self, "weights_decay"):
            self.weights = self.weights_decay.copy()

        # get total length of the central orbit covered by this lattice
        C = lattice.s(1)  # cm

        # Total circumference of the machine (cm). If the lattice describes only a
        # portion of the machine, set lattice.total_circumference to the full
        # machine length: decays are then placed only on the covered section, while
        # the muons propagate (age and decay) over the full circumference -- their
        # travel times self.mutimes account for the uncovered part of each turn.
        C_machine = getattr(lattice, "total_circumference", None)
        if C_machine is None:
            C_machine = C
        elif C_machine < C:
            warnings.warn(
                f"lattice.total_circumference ({C_machine:.3g} cm) is smaller than "
                f"the lattice arc length ({C:.3g} cm); ignoring it and treating the "
                "lattice as the full machine."
            )
            C_machine = C
        elif C_machine > C:
            warnings.warn(
                f"The lattice covers only {C / C_machine:.1%} of the machine "
                f"(arc length {C:.4g} cm of total_circumference {C_machine:.4g} cm). "
                "Muon decays are placed on the covered section only; muon travel "
                "times account for propagation through the full machine."
            )

        # Exposure (path length) within the covered section, over all cycles
        covered_s = self.cycles * C

        # Place muon decays uniformly along the covered path
        s_covered = np.random.uniform(0, covered_s, self.sample_size)
        # s_in_turn is the position of the muons within the covered section
        turn = np.floor(s_covered / C)
        self.s_in_turn = s_covered - turn * C
        # True path length traveled, including the uncovered part of each turn
        self.s_muon = turn * C_machine + self.s_in_turn

        # parameter u that goes from 0 to 1 along the lattice
        u_parameter = lattice.inv_s(self.s_in_turn)

        if self.beam_dynamics:
            # Get the beam momentum
            beam_p = np.random.normal(
                loc=lattice.beam_p0(u_parameter),
                scale=lattice.beamdiv_z(u_parameter) * lattice.beam_p0(u_parameter),
                size=self.sample_size,
            )
        else:
            beam_p = lattice.beam_p0(u_parameter) * np.ones(self.sample_size)

        # Build the muon 4-momentum in the LOCAL beam-comoving frame, with the beam
        # travelling along +z, the horizontal transverse direction along local x, and
        # the vertical direction along local y (same axis roles as the world frame).
        self.pmu = vector.array(
            {
                "E": np.sqrt(beam_p**2 + const.m_mu**2),
                "px": np.zeros(self.sample_size),
                "py": np.zeros(self.sample_size),
                "pz": beam_p,
            }
        )

        # Beam transverse divergence, sampled from the local envelopes:
        #   beamdiv_x -> horizontal (local x),  beamdiv_y -> vertical (local y).
        if self.beam_dynamics:
            theta_h = np.random.normal(
                loc=0.0,
                scale=lattice.beamdiv_x(u_parameter),
                size=self.sample_size,
            )
            theta_v = np.random.normal(
                loc=0.0,
                scale=lattice.beamdiv_y(u_parameter),
                size=self.sample_size,
            )
        else:
            theta_h = 0.0
            theta_v = 0.0

        # Apply the divergence in the local frame. rotateY bends +z into local x
        # (horizontal); rotateX bends +z into local y (vertical). This keeps the
        # horizontal/vertical divergences aligned with the x/y axes.
        self.pmu = self.pmu.rotateY(np.arctan(theta_h))
        self.pmu = self.pmu.rotateX(-np.arctan(theta_v))

        # Boost the neutrino 4-momenta into the same local beam-comoving frame.
        self.pnu = self.pnu_restframe.boost_p4(self.pmu)

        # Absolute velocity of muons
        self.vmu = const.c_LIGHT * np.sqrt((1 - (const.m_mu / self.pmu["E"]) ** 2))

        # spread muons in time according to number of beam lifetimes desired
        # (s_muon includes the full-machine path, so times are physical)
        self.mutimes = self.s_muon / self.vmu  # time in seconds
        # exposure time of the sampled (covered) path -- the MC measure
        max_time = covered_s / self.vmu

        self.muon_lifetime = const.tau0_mu * self.pmu["E"] / const.m_mu

        # Now, if we want to increase our efficiency in the simulation, we better force particles to be close to the detector in some way.
        # Let's enforce this by clipping the s_in_turn range:
        if (
            C_machine > C
            and self.remove_ring_fraction[0] != self.remove_ring_fraction[1]
        ):
            warnings.warn(
                "remove_ring_fraction is not supported for lattices covering only "
                "part of the machine (total_circumference > lattice length); "
                "ignoring it."
            )
            self.remove_ring_fraction = [0.5, 0.5]
        zacc_min = C * self.remove_ring_fraction[0]
        zacc_max = C * self.remove_ring_fraction[1]
        events_likely_within_acceptance = (self.s_in_turn <= zacc_min) | (
            self.s_in_turn > zacc_max
        )

        # Now we move muons along an extra distance to fall within acceptance, paying the price of a smaller survival probability
        shifted_events = ~events_likely_within_acceptance
        deltaz_min = zacc_max - self.s_in_turn
        deltaz_max = C - self.s_in_turn + zacc_min
        shift_z = np.zeros(self.sample_size)
        shift_z[shifted_events] = (
            np.random.uniform(0, 1, shifted_events.sum())
            * (deltaz_max[shifted_events] - deltaz_min[shifted_events])
            + deltaz_min[shifted_events]
        )

        # Apply shift to s_in_this_turn and to the travel time of the muons
        self.s_in_turn = self.s_in_turn + shift_z
        self.mutimes += shift_z / self.vmu
        self.s_muon = self.s_muon + shift_z
        self.s_in_turn = self.s_in_turn % C

        # Acceptance of the simulated region: after the shift ALL samples land
        # in the kept window (natives at per-turn density 1/C, shifted ones at
        # (C - W_acc)/C * 1/W_acc; combined exactly 1/W_acc), so each sample
        # over-represents the uniform-ring measure by C/W_acc. Scale by
        # W_acc/C to keep the decay estimator unbiased. No-op when nothing is
        # removed (W_acc = C).
        self.weights[:, 0] *= (zacc_min + (C - zacc_max)) / C

        # Apply exponential suppression on total length travelled by muons
        self.weights[:, 0] *= (
            # 1 - np.exp(-self.mutimes / self.muon_lifetime)
            lattice.Nmu_per_bunch
            * (max_time)
            * np.exp(-self.mutimes / self.muon_lifetime)
            / self.muon_lifetime
        )

        # Now place the decays in world coordinates along the central orbit.
        # The lattice returns world coordinates directly:
        #   x -> toward the ring center, y -> up, z -> tangential (travel direction).
        self.pos["x"] = lattice.x(u_parameter)
        self.pos["y"] = lattice.y(u_parameter)
        self.pos["z"] = lattice.z(u_parameter)

        # Sample the transverse beam envelopes (0 if beam dynamics are disabled).
        #   beamsize_x -> horizontal offset,  beamsize_y -> vertical offset.
        if self.beam_dynamics:
            sig_x = np.broadcast_to(
                np.asarray(lattice.beamsize_x(u_parameter), dtype=float),
                (self.sample_size,)).copy()
            sig_y = np.broadcast_to(
                np.asarray(lattice.beamsize_y(u_parameter), dtype=float),
                (self.sample_size,)).copy()
            # Collimate at the physical aperture: draw the transverse offsets
            # from a Gaussian TRUNCATED at the n-sigma envelope ellipse,
            # (x/sig_x)^2 + (y/sig_y)^2 <= n^2. This is the machine aperture the
            # MuCol shielding liners are cut to; mint.beamline.StraightSectionShield
            # uses the same envelope for the inner radius of the tungsten.
            # Sampling directly in normalized polar coordinates is exact and
            # needs no rejection loop: the squared normalized radius of a 2D
            # Gaussian is Exp(1/2), truncated to [0, n^2] by inverting its CDF.
            n_ap = self.aperture_nsigma
            g1 = np.random.normal(size=self.sample_size)
            g2 = np.random.normal(size=self.sample_size)
            if n_ap is not None and n_ap > 0:
                phi = np.random.uniform(0.0, 2.0 * np.pi, self.sample_size)
                u_r = np.random.uniform(0.0, 1.0, self.sample_size)
                r2 = -2.0 * np.log1p(-u_r * (-np.expm1(-0.5 * n_ap**2)))
                r_n = np.sqrt(r2)
                g1, g2 = r_n * np.cos(phi), r_n * np.sin(phi)
            x_horizontal = g1 * sig_x
            x_vertical = g2 * sig_y
        else:
            x_horizontal = np.zeros(self.sample_size)
            x_vertical = np.zeros(self.sample_size)

        # World unit tangent of the central orbit at each decay point, [3, N].
        t_stack = np.asarray(lattice.tangent(u_parameter))
        if t_stack.shape[0] != 3:
            t_stack = t_stack.reshape(3, -1)
        tx, ty, tz = t_stack[0, :], t_stack[1, :], t_stack[2, :]
        tnorm = np.sqrt(tx**2 + ty**2 + tz**2)
        tnorm[tnorm == 0] = 1.0
        tx, ty, tz = tx / tnorm, ty / tnorm, tz / tnorm

        # Counter-circulating beam: same central orbit, opposite travel direction.
        # Reverse the tangent so the momenta point the other way (positions unchanged).
        if direction == "counter-clockwise":
            tx, ty, tz = -tx, -ty, -tz

        # Build a transverse basis aligned with the world vertical and horizontal:
        #   n_v -> vertical (world +y made perpendicular to t),
        #   n_h = t x n_v -> horizontal, in the ring plane, perpendicular to t.
        evt = ty  # world +y . t
        nv_x = -evt * tx
        nv_y = 1.0 - evt * ty
        nv_z = -evt * tz
        nv_mag = np.sqrt(nv_x**2 + nv_y**2 + nv_z**2)
        nv_mag[nv_mag == 0] = 1.0
        nv_x, nv_y, nv_z = nv_x / nv_mag, nv_y / nv_mag, nv_z / nv_mag

        nh_x = ty * nv_z - tz * nv_y
        nh_y = tz * nv_x - tx * nv_z
        nh_z = tx * nv_y - ty * nv_x
        nh_mag = np.sqrt(nh_x**2 + nh_y**2 + nh_z**2)
        nh_mag[nh_mag == 0] = 1.0
        nh_x, nh_y, nh_z = nh_x / nh_mag, nh_y / nh_mag, nh_z / nh_mag

        # Transverse offsets: horizontal along n_h, vertical along n_v.
        self.pos["x"] = self.pos["x"] + x_horizontal * nh_x + x_vertical * nv_x
        self.pos["y"] = self.pos["y"] + x_horizontal * nh_y + x_vertical * nv_y
        self.pos["z"] = self.pos["z"] + x_horizontal * nh_z + x_vertical * nv_z

        # Map momenta from the local beam frame (n_h, n_v, t) into the world frame:
        #   local px -> n_h (horizontal), py -> n_v (vertical), pz -> t (tangential).
        for p in (self.pmu, self.pnu):
            # Copy the components first: p["px"] returns a view, so we must not
            # overwrite px before py/pz are read.
            px = np.array(p["px"])
            py = np.array(p["py"])
            pz = np.array(p["pz"])
            p["px"] = px * nh_x + py * nv_x + pz * tx
            p["py"] = px * nh_y + py * nv_y + pz * ty
            p["pz"] = px * nh_z + py * nv_z + pz * tz

        # Time relative to the bunch crossing at the interaction point (IP).
        # The IP is the point of the central orbit at the world origin (0, 0, 0)
        # -- the same origin the detectors are placed relative to. mutimes_to_bunchx
        # is 0 for a muon exactly at the IP, positive just after it (already crossed),
        # and negative just before it (about to cross).
        # The IP arc length only depends on the lattice, so cache it there.
        # NOTE: a single coarse scan is not enough. The grid spacing (~7 cm for
        # a 1.5 km lattice on 20001 points) leaves the "IP" point up to a few cm
        # from the true orbit crossing, and since the detector distance is
        # measured from the world origin, that mismatch enters the arrival times
        # directly as (offset)/c -- tens of ps, i.e. LARGER than the physical
        # bunch-length spread. Refine locally around the coarse minimum.
        if not hasattr(lattice, "_s_IP"):
            def _dist2(uu):
                return (lattice.x(uu) ** 2 + lattice.y(uu) ** 2
                        + lattice.z(uu) ** 2)

            u_grid = np.linspace(0.0, 1.0, 20001)
            i0 = int(np.argmin(_dist2(u_grid)))
            du = u_grid[1] - u_grid[0]
            lo, hi = max(u_grid[i0] - du, 0.0), min(u_grid[i0] + du, 1.0)
            for _ in range(4):  # ~1e-4 of the coarse spacing after 4 passes
                u_fine = np.linspace(lo, hi, 2001)
                j = int(np.argmin(_dist2(u_fine)))
                step = u_fine[1] - u_fine[0]
                lo, hi = max(u_fine[j] - step, 0.0), min(u_fine[j] + step, 1.0)
            lattice._s_IP = float(lattice.s(0.5 * (lo + hi)))
        s_IP = lattice._s_IP

        # Signed arc distance from the IP within one machine turn, wrapped to
        # [-C_machine/2, C_machine/2). The covered section occupies machine arc
        # [0, C), so s_in_turn is also the position along the machine.
        delta_s = (
            self.s_in_turn - s_IP + C_machine / 2.0
        ) % C_machine - C_machine / 2.0
        # A counter-clockwise bunch moves toward decreasing s: a muon at
        # s > s_IP has NOT yet crossed the IP, so the signed time flips.
        if direction == "counter-clockwise":
            delta_s = -delta_s
        self.mutimes_to_bunchx = delta_s / self.vmu

        # Finite bunch length: a muon at longitudinal offset dz within the bunch
        # (dz > 0 = ahead of the bunch centroid) crosses the IP dz/v earlier than
        # the centroid, so its decay time relative to the bunch crossing shifts
        # by -dz/v. The decay positions themselves are unaffected (a ~mm smear of
        # a distribution that is uniform along the ring).
        if self.beam_dynamics:
            dz_bunch = np.random.normal(
                loc=0.0,
                scale=lattice.beamsize_z(u_parameter) * np.ones(self.sample_size),
            )
            self.mutimes_to_bunchx -= dz_bunch / self.vmu

        return self

    def get_flux_at_generic_location(
        self,
        det_location=None,
        det_radius=1e2,
        ebins=100,
        E_range=None,
        per_cm2=True,
        new_polarization=None,
        normalization=1,
        mask=None,
    ):

        if mask is None:
            mask = np.ones(self.sample_size, dtype=bool)
        if det_location is None:
            det_location = [0.0, 0.0, 1e5]      # 1 km downstream, on axis
        det_location = np.asarray(det_location)
        if det_location.shape != (3,):
            raise ValueError("det_location must be a list or array of length 3.")
        det_vector = vector.array(
            {"x": det_location[0], "y": det_location[1], "z": det_location[2]}
        )
        # normal_to_detector_plane = det_vector.unit()
        distances = det_vector - self.pos[mask]
        neutrino_direction = self.pnu[mask].to_3D().unit()
        dotprod = distances.dot(neutrino_direction)

        # Project distance vector onto neutrino direction
        sintheta = np.sqrt(1 - distances.unit().dot(neutrino_direction) ** 2)

        # Position of closest approach on the neutrino path
        radial_distance = sintheta * distances.mag

        # Check if neutrino crosses the detector disk and has sufficient energy
        if E_range is None:
            E_range = (np.min(self.pnu["E"][mask]), np.max(self.pnu["E"][mask]))

        in_acceptance = (
            (dotprod > 0)
            & (radial_distance < det_radius)
            & (self.pnu["E"][mask] > E_range[0])
            & (self.pnu["E"][mask] < E_range[1])
        )

        # Detector area
        area = np.pi * det_radius**2

        # Select appropriate weights
        if new_polarization is not None:
            self.reweight_with_new_polarization(new_polarization)
            w = self.weights_new_pol[mask]
        else:
            w = self.weights[mask, 0]

        # Normalize accepted neutrino weights
        if np.sum(w) == 0:
            print("Warning: sum of weights is zero.")
            return 0, 0
        nu_eff_ND = np.sum(w[in_acceptance]) / np.sum(w)

        if nu_eff_ND > 0:
            Enu_ND, flux_nu_ND_p = get_flux(
                self.pnu["E"][mask][in_acceptance], w[in_acceptance], ebins
            )

            # Convert to flux per unit area per unit energy
            if per_cm2:
                flux_nu_ND = normalization * flux_nu_ND_p / area / np.diff(Enu_ND)
            else:
                flux_nu_ND = normalization * flux_nu_ND_p / np.diff(Enu_ND)

            return Enu_ND, flux_nu_ND
        else:
            print("No flux through detector.")
            return ebins, 0 * ebins[:-1]


def get_flux(x, w, nbins):
    hist1 = np.histogram(
        x, weights=w, bins=nbins, density=False, range=(np.min(x), np.max(x))
    )

    ans0 = hist1[1]
    ans1 = hist1[0]  # /(ans0[1]-ans0[0])
    return ans0, ans1
