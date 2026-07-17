"""
Heavy Neutral Lepton (HNL) sensitivity from the forward neutrino flux.

First-pass estimate of a forward-neutrino detector's reach to a heavy neutral
lepton N, for masses above the D meson (m_N ~ 2-6 GeV). Two portals are supported:

  * MixingPortal  -- N mixes with the active neutrinos, |U_aN|^2. Production is
        neutral-current DIS upscattering  nu N -> N X  (~ |U|^2 sigma_NC), decay is
        N -> nu l+ l- / l + hadrons (width ~ |U|^2 G_F^2 m_N^5).
  * DipolePortal  -- N couples via a transition magnetic dipole d [GeV^-1],
        L = d (nubar sigma^{mu nu} N) F_{mu nu}. Production is COHERENT Primakoff
        upscattering off the nucleus (~ d^2 Z^2 alpha), decay is the radiative
        N -> nu gamma (single photon, width = d^2 m_N^3 / 4pi).

The signal chain is the same for both:
    (1) upscattering in the detector material
    (2) the boosted N (gamma_N = E_N/m_N ~ 10^3) propagates through the block
    (3) it decays (visibly) inside the fiducial length.
The rate scales as coupling^2 (production) times a decay probability, giving the
familiar coupling^2 (short-lived) to coupling^4 (long-lived) reach.

Everything is deliberately simple and clearly approximate; the goal is a
feasibility-level estimate. Energies/masses in GeV, lengths in cm, xsecs in cm^2.
"""

from dataclasses import dataclass, field

import numpy as np

from mint import const, xsecs
from mint import detector_tools as dt


# ============================================================================
# Detector
# ============================================================================
@dataclass
class ForwardDetector:
    """A simple homogeneous forward detector block on the beam axis.

    Upscattering happens in the detector material and the produced HNL is required
    to decay (visibly) before exiting the block of length ``length_cm``.
    """

    distance_cm: float = 5e5          # 5 km from the IP
    radius_cm: float = 100.0          # 1 m radius acceptance
    length_cm: float = 1000.0         # 10 m long
    material: dt.Material = field(default_factory=lambda: dt.Fe)

    @property
    def nucleon_density(self):
        """Nucleon number density [1/cm^3]."""
        return self.material.N

    @property
    def nucleus_density(self):
        """Nucleus number density [1/cm^3]."""
        return self.material.N / self.material.A


# ============================================================================
# Mixing portal (|U_aN|^2): NC-DIS upscattering + weak decay
# ============================================================================
def mixing_total_width(m_N, U2, n_eff=10.0):
    """Total HNL decay width [GeV], leading inclusive estimate.

    Gamma = |U|^2 G_F^2 m_N^5 / (96 pi^3) * n_eff.  The m_N^5 scaling is exact at
    leading order; n_eff (~5-15 for GeV-scale N) is an effective count of open
    leptonic+hadronic channels. Replace with channel-resolved widths for precision.
    """
    return U2 * const.Gf**2 * np.asarray(m_N, float) ** 5 / (96 * np.pi**3) * n_eff


def nc_dis_xsec_sm(E_nu, nuflavor="numubar"):
    """SM per-nucleon neutral-current DIS cross section [cm^2] (isoscalar)."""
    E_nu = np.asarray(E_nu, float)
    return xsecs.sigma_NC_nubar(E_nu) if "bar" in nuflavor else xsecs.sigma_NC_nu(E_nu)


def mixing_upscattering_xsec(E_nu, m_N, U2, nuflavor="numubar"):
    """nu N -> N X upscattering cross section [cm^2/nucleon], = |U|^2 sigma_NC * threshold."""
    E_nu = np.asarray(E_nu, float)
    thr = np.clip(1.0 - m_N**2 / (2.0 * const.m_proton * E_nu), 0.0, 1.0)
    return U2 * nc_dis_xsec_sm(E_nu, nuflavor) * thr


# ============================================================================
# Dipole portal (d [GeV^-1]): coherent Primakoff upscattering + radiative decay
# ============================================================================
def dipole_radiative_width(m_N, d):
    """Radiative width Gamma(N -> nu gamma) [GeV] = d^2 m_N^3 / (4 pi)."""
    return d**2 * np.asarray(m_N, float) ** 3 / (4.0 * np.pi)


def _nuclear_coherence_scale_GeV(A):
    """Inverse nuclear radius 1/R_A [GeV], R_A = 1.2 A^{1/3} fm -- the coherence cutoff."""
    R_A_invGeV = 1.2 * A ** (1.0 / 3.0) * const.fm_to_GeV**-1  # fm -> GeV^-1
    return 1.0 / R_A_invGeV


def dipole_coherent_xsec(E_nu, m_N, d, Z, A, coeff=2.0):
    """Coherent Primakoff upscattering nu A -> N A cross section [cm^2/nucleus].

    Leading-log estimate:
        sigma = coeff * alpha * Z^2 * d^2 * ln(Q2_max / Q2_min) * (GeV^-2 -> cm^2),
    with Q2_min = (m_N^2 / 2E_nu)^2 (min. momentum transfer to make mass m_N) and
    Q2_max = (1/R_A)^2 (nuclear form-factor cutoff). This captures the d^2 Z^2 alpha
    scaling and the mild log growth with energy; the O(1) normalization and the
    incoherent/DIS-photon contributions at high Q^2 are approximate (see Magill,
    Plestid, Pospelov, Tsai 2018 for the full treatment).
    """
    E_nu = np.asarray(E_nu, float)
    Q2_min = (m_N**2 / (2.0 * E_nu)) ** 2
    Q2_max = _nuclear_coherence_scale_GeV(A) ** 2
    log_factor = np.clip(np.log(Q2_max / Q2_min), 0.0, None)
    sigma_GeV2 = coeff * const.alphaQED * Z**2 * d**2 * log_factor
    return sigma_GeV2 * const.invGeV2_to_cm2


# ============================================================================
# Shared: lab decay length
# ============================================================================
def decay_length(E_N, m_N, width):
    """Lab-frame decay length lambda = gamma*beta*c*tau [cm] for a given rest width [GeV]."""
    E_N = np.asarray(E_N, float)
    gamma = E_N / m_N
    beta = np.sqrt(np.clip(1.0 - 1.0 / gamma**2, 0.0, 1.0))
    ctau = const.invGeV_to_cm / width
    return gamma * beta * ctau


# ============================================================================
# Portals: bundle production + decay + target counting for the signal formula
# ============================================================================
class Portal:
    """Base portal. Subclasses provide production, decay, and target counting."""

    y_mean = 0.0        # E_N = (1 - y_mean) * E_nu
    br_visible = 1.0

    def production_xsec(self, E_nu, coupling, material):
        raise NotImplementedError

    def target_density(self, material):
        raise NotImplementedError

    def width(self, m_N, coupling):
        raise NotImplementedError


class MixingPortal(Portal):
    """Minimal (mass-mixing) HNL, coupling = |U_aN|^2."""

    def __init__(self, m_N, nuflavor="numubar", y_mean=0.3, br_vis=0.5, n_eff=10.0):
        self.m_N = m_N
        self.nuflavor = nuflavor
        self.y_mean = y_mean
        self.br_visible = br_vis
        self.n_eff = n_eff

    def production_xsec(self, E_nu, U2, material):
        # per-nucleon NC-DIS
        return mixing_upscattering_xsec(E_nu, self.m_N, U2, self.nuflavor)

    def target_density(self, material):
        return material.N  # nucleons / cm^3

    def width(self, m_N, U2):
        return mixing_total_width(m_N, U2, self.n_eff)


class DipolePortal(Portal):
    """Dipole-portal HNL, coupling = d [GeV^-1]. Production coherent, decay N->nu gamma."""

    def __init__(self, m_N, br_vis=1.0, coeff=2.0):
        self.m_N = m_N
        self.y_mean = 0.0        # coherent/elastic: N carries ~ the full neutrino energy
        self.br_visible = br_vis  # single photon -> fully visible
        self.coeff = coeff

    def production_xsec(self, E_nu, d, material):
        # per-nucleus coherent Primakoff
        return dipole_coherent_xsec(
            E_nu, self.m_N, d, material.Z, material.A, coeff=self.coeff
        )

    def target_density(self, material):
        return material.N / material.A  # nuclei / cm^3

    def width(self, m_N, d):
        return dipole_radiative_width(m_N, d)


# ============================================================================
# Signal and sensitivity
# ============================================================================
# ============================================================================
# DarkNews-backed upscattering (coherent + incoherent), for higher masses
# ============================================================================
# The analytic coherent estimate above loses accuracy as m_N grows (the required
# momentum transfer breaks coherence and the incoherent/nucleon-elastic channels
# take over). DarkNews computes the proper coherent + nucleon-elastic cross section
# with real nuclear form factors and the full m_N dependence, which is what sets the
# reach at high mass. We use it only for the *production* cross section (coherent +
# p-elastic + n-elastic); the decay widths stay analytic.

_DN_PROJECTILE = {"e": "nue", "mu": "numu", "tau": "nutau"}


def _get_darknews():
    """Import DarkNews lazily and quiet its logger. Raises a helpful error if absent."""
    try:
        import DarkNews as dn
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "DarkNews is required for the DarkNews-backed cross sections. "
            "Install it with `pip install DarkNews`."
        ) from exc
    dn.configure_loggers(loglevel="ERROR")
    return dn


def _loglog_interp(E_query, E_grid, sigma_grid):
    """Interpolate a (possibly threshold-truncated) cross section in log-log space.
    Returns 0 outside the support (below threshold / above the grid)."""
    E_query = np.asarray(E_query, float)
    good = sigma_grid > 0
    if good.sum() < 2:
        return np.zeros_like(E_query)
    out = np.exp(
        np.interp(
            np.log(E_query),
            np.log(E_grid[good]),
            np.log(sigma_grid[good]),
            left=-np.inf,
            right=np.log(sigma_grid[good][-1]),  # flat extrapolation at high E
        )
    )
    out[E_query < E_grid[good][0]] = 0.0
    return out


def darknews_upscattering_xsec(
    E_grid, m_N, coupling_kind="dipole", target="W184", nu_flavor="mu",
    regimes=("coherent", "p-el", "n-el"), ref_coupling=1e-3, NEVAL=1000,
):
    """Coherent + incoherent upscattering cross section from DarkNews [cm^2/nucleus].

    Computed at a single reference coupling (the cross section scales exactly as the
    coupling squared, so it is rescaled analytically elsewhere). ``coupling_kind`` is
    'dipole' (sets ``mu_tr_..4`` = ref_coupling) or 'mixing' (sets ``U..4`` = ref_coupling).
    Returns sigma_ref(E_grid), summed over ``regimes``, clipped to >= 0.
    """
    dn = _get_darknews()
    E_grid = np.atleast_1d(np.asarray(E_grid, float))
    proj = getattr(dn.pdg, _DN_PROJECTILE[nu_flavor])
    tgt = dn.NuclearTarget(target)

    mkw = dict(name="mint_hnl", m4=float(m_N), epsilon=0.0, gD=0.0, Umu4=0.0, mu_tr_mu4=0.0)
    if coupling_kind == "dipole":
        mkw[f"mu_tr_{nu_flavor}4"] = ref_coupling
        helicity = "flipping"
    elif coupling_kind == "mixing":
        mkw[f"U{nu_flavor}4"] = ref_coupling
        helicity = "conserving"
    else:
        raise ValueError("coupling_kind must be 'dipole' or 'mixing'")
    model = dn.model.ThreePortalModel(**mkw)

    sigma = np.zeros_like(E_grid)
    for regime in regimes:
        up = dn.UpscatteringProcess(
            nu_projectile=proj, nu_upscattered=dn.pdg.neutrino4,
            nuclear_target=tgt, scattering_regime=regime,
            TheoryModel=model, helicity=helicity,
        )
        sigma = sigma + np.asarray(up.total_xsec(E_grid, NEVAL=NEVAL), float)
    return np.clip(sigma, 0.0, None)


class DarkNewsPortal(Portal):
    """Portal whose *production* cross section (coherent + incoherent) comes from
    DarkNews, tabulated once per mass and rescaled by the coupling. Decay widths are
    the analytic radiative (dipole) or weak (mixing) expressions.

    coupling is d [GeV^-1] (dipole) or |U_aN|^2 (mixing), matching the analytic portals.
    """

    def __init__(
        self, m_N, coupling_kind="dipole", target="W184", nu_flavor="mu",
        regimes=("coherent", "p-el", "n-el"), br_vis=None, n_eff=10.0,
        E_grid=None, ref_coupling=1e-3, NEVAL=1000,
    ):
        self.m_N = m_N
        self.coupling_kind = coupling_kind
        self.nu_flavor = nu_flavor
        self.n_eff = n_eff
        self.y_mean = 0.0  # coherent/nucleon-elastic: N carries ~the full neutrino energy
        self.br_visible = (1.0 if coupling_kind == "dipole" else 0.5) if br_vis is None else br_vis

        if E_grid is None:
            E_grid = np.geomspace(max(2.0 * m_N, 20.0), 5e3, 40)
        self._E_grid = np.asarray(E_grid, float)
        self._ref = ref_coupling
        self._sigma_ref = darknews_upscattering_xsec(
            self._E_grid, m_N, coupling_kind=coupling_kind, target=target,
            nu_flavor=nu_flavor, regimes=regimes, ref_coupling=ref_coupling, NEVAL=NEVAL,
        )

    def production_xsec(self, E_nu, coupling, material):
        sigma = _loglog_interp(E_nu, self._E_grid, self._sigma_ref)
        if self.coupling_kind == "dipole":     # sigma ~ d^2, coupling = d
            return sigma * (coupling / self._ref) ** 2
        return sigma * (coupling / self._ref**2)  # sigma ~ |U|^2, coupling = |U|^2

    def target_density(self, material):
        return material.N / material.A  # DarkNews total_xsec is per nucleus

    def width(self, m_N, coupling):
        if self.coupling_kind == "dipole":
            return dipole_radiative_width(m_N, coupling)
        return mixing_total_width(m_N, coupling, self.n_eff)


def signal_events(E_nu, w_nu, portal, coupling, detector):
    """Expected number of visible HNL decays for a flux (E_nu, w_nu) and a portal.

    Args:
        E_nu, w_nu: neutrino energies [GeV] and weights (number of neutrinos crossing
            the detector face over the exposure of interest).
        portal: a MixingPortal or DipolePortal (carries m_N and model parameters).
        coupling: |U|^2 (mixing) or d [GeV^-1] (dipole).
        detector: ForwardDetector.
    """
    E_nu = np.atleast_1d(np.asarray(E_nu, float))
    w_nu = np.atleast_1d(np.asarray(w_nu, float))

    sigma = portal.production_xsec(E_nu, coupling, detector.material)
    n_target = portal.target_density(detector.material)
    E_N = (1.0 - portal.y_mean) * E_nu
    lam = decay_length(E_N, portal.m_N, portal.width(portal.m_N, coupling))

    L = detector.length_cm
    lam_safe = np.maximum(lam, 1e-300)
    # production uniform along L, HNL must decay before exiting the block
    decay_in_L = L - lam * (1.0 - np.exp(-L / lam_safe))
    p = n_target * sigma * decay_in_L
    return float(np.sum(w_nu * portal.br_visible * p))


def sensitivity_grid(
    E_nu, w_nu, mass_grid, coupling_grid, detector, portal_factory
):
    """Expected signal events over a (mass, coupling) grid.

    Args:
        portal_factory: callable m_N -> Portal (e.g. lambda m: MixingPortal(m, ...)).
    Returns array [n_mass, n_coupling].
    """
    mass_grid = np.asarray(mass_grid, float)
    coupling_grid = np.asarray(coupling_grid, float)
    out = np.zeros((mass_grid.size, coupling_grid.size))
    for i, m_N in enumerate(mass_grid):
        portal = portal_factory(m_N)
        for j, c in enumerate(coupling_grid):
            out[i, j] = signal_events(E_nu, w_nu, portal, c, detector)
    return out


def exclusion_band(mass_grid, coupling_grid, n_sig_grid, n_events=2.3):
    """Lower/upper coupling of the N_sig = n_events contour for each mass.

    For a homogeneous block (production + decay in the same volume) the reach has a
    lower edge only; ``upper`` is NaN where the signal does not turn back below
    ``n_events`` at large coupling.
    """
    coupling_grid = np.asarray(coupling_grid, float)
    lower = np.full(mass_grid.size, np.nan)
    upper = np.full(mass_grid.size, np.nan)
    for i in range(mass_grid.size):
        above = n_sig_grid[i] >= n_events
        if not np.any(above):
            continue
        idx = np.where(above)[0]
        lower[i] = coupling_grid[idx.min()]
        if idx.max() < coupling_grid.size - 1:
            upper[i] = coupling_grid[idx.max()]
    return lower, upper
