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

import os
from dataclasses import dataclass, field

import numpy as np

from mint import const, xsecs
from mint import detector_tools as dt

import hnl_widths


def _flavor_of(nuflavor):
    """Map a neutrino-flavor string ('numu', 'numubar', 'nue', ...) to the
    charged-lepton flavor label used by hnl_widths ('e', 'mu', 'tau')."""
    s = str(nuflavor).lower()
    if "tau" in s:
        return "tau"
    if "mu" in s:
        return "mu"
    return "e"


# ============================================================================
# Detector
# ============================================================================


# ============================================================================
# Mixing portal (|U_aN|^2): NC-DIS upscattering + weak decay
# ============================================================================
def mixing_total_width(m_N, U2, flavor="mu"):
    """Total HNL decay width [GeV], channel-resolved.

    Delegates to hnl_widths.hnl_total_width: sums the leptonic modes
    (3-nu, nu l+ l-, l l' nu), the exclusive meson modes below ~1 GeV, and the
    inclusive quark-level modes above, using the Coloma et al. (2007.03701)
    decay constants and meson couplings. ``flavor`` selects the mixing flavor
    ('e', 'mu', 'tau'). See hnl_widths for provenance.
    """
    return hnl_widths.hnl_total_width(m_N, U2, flavor=flavor)


def mixing_total_width_crude(m_N, U2, n_eff=10.0):
    """Legacy leading-order estimate, Gamma = |U|^2 G_F^2 m_N^5/(96 pi^3) * n_eff.
    Kept only for comparison with the channel-resolved mixing_total_width."""
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
    R_A_invGeV = 1.2 * A ** (1.0 / 3.0) * const.fm_to_GeV  # fm -> GeV^-1
    return 1.0 / R_A_invGeV


def dipole_pair_width(m_N, d, m_lepton):
    """EXACT width of the internal-conversion decay N -> nu l+ l- (off-shell
    photon) for the dipole portal, Eq. (B.5) of Jodlowski & Trojanowski,
    arXiv:2011.04751 (with mu_N = d):

        Gamma = e^2 d^2 [ (8 m_l^6 - 2 m_N^6) ln( 2 m_l / (sqrt(m_N^2-4m_l^2)+m_N) )
                          + m_N sqrt(m_N^2-4m_l^2) (-2m_l^4 + 5 m_l^2 m_N^2 - 3 m_N^4) ]
                / (96 pi^3 m_N^3)

    Vanishes at threshold m_N = 2 m_l; for m_N >> m_l it reduces to
    Gamma(N->nu gamma) * alpha/(6 pi) [ln(m_N^2/m_l^2) - 3].
    """
    m = np.asarray(m_N, float)
    ml = float(m_lepton)
    open_ch = m > 2.0 * ml
    msafe = np.where(open_ch, m, 2.0 * ml * (1.0 + 1e-12))
    root = np.sqrt(np.clip(msafe**2 - 4.0 * ml**2, 0.0, None))
    log_term = np.log(2.0 * ml / (root + msafe))
    num = (8.0 * ml**6 - 2.0 * msafe**6) * log_term + msafe * root * (
        -2.0 * ml**4 + 5.0 * ml**2 * msafe**2 - 3.0 * msafe**4
    )
    e2 = 4.0 * np.pi * const.alphaQED
    gamma = e2 * d**2 * num / (96.0 * np.pi**3 * msafe**3)
    return np.where(open_ch, np.clip(gamma, 0.0, None), 0.0)


def dipole_pair_br(m_N, m_lepton):
    """Branching ratio BR(N -> nu l+ l-) = Gamma_ll / Gamma(N -> nu gamma),
    using the EXACT internal-conversion width (dipole_pair_width, Eq. (B.5) of
    arXiv:2011.04751). The coupling cancels in the ratio. ~0.4% (e+e-) and
    ~0.1% (mu+mu-) at the GeV scale, within the B(N -> nu l l) ~ 1e-3 - 1e-2
    range quoted in that paper. The l+ l- pair gives a reconstructable displaced
    vertex even when the production point is invisible (coherent upscattering
    in rock/detector), at the price of this small BR.
    """
    return dipole_pair_width(m_N, 1.0, m_lepton) / dipole_radiative_width(m_N, 1.0)




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


def dipole_dis_xsec(E_nu, m_N, d, Z, A, Q2_min=1.0, coeff=2.0):
    """Deep-inelastic dipole upscattering nu q -> N q [cm^2/nucleon], leading log.

    Photon exchange off the valence quarks, with the per-nucleon charge sum
    sum e_q^2 = 1 (proton) / 2/3 (neutron), and the log running from the DIS
    scale Q2_min (~1 GeV^2; below it the scattering is elastic/coherent and is
    covered by dipole_coherent_xsec) up to the kinematic limit s = 2 m_p E_nu:

        sigma = coeff * alpha * d^2 * <sum e_q^2> * ln(s / Q2_min).

    Same leading-log spirit (and coeff) as the coherent estimate. No Z^2
    coherence, so this is a small fraction of the total production -- but the
    hadronic recoil makes the production vertex VISIBLE, enabling the
    displaced-vertex (min-displacement) selection in the detector.
    """
    E_nu = np.asarray(E_nu, float)
    s = 2.0 * const.m_proton * E_nu
    sum_eq2 = (Z * 1.0 + (A - Z) * (2.0 / 3.0)) / A
    with np.errstate(divide="ignore", invalid="ignore"):
        log_factor = np.clip(np.log(s / Q2_min), 0.0, None)
    log_factor = np.where(s > m_N**2 + Q2_min, log_factor, 0.0)
    return coeff * const.alphaQED * d**2 * sum_eq2 * log_factor * const.invGeV2_to_cm2


# ============================================================================
# Dipole upscattering on ELECTRONS: nu e- -> N e- (single-recoil signature)
# ============================================================================
def dipole_electron_dsigma_dt(t, E_nu, m_N, d):
    """Exact dsigma/dt [GeV^-4] for nu e- -> N e- via the dipole portal,
    Eq. (A.10) of Jodlowski & Trojanowski (arXiv:2011.04751), mu_N = d.
    Same formula validated (in the guise of a point-proton target) in
    validate_dipole_xsecs.py. t < 0; identical for nu and nubar projectiles.
    """
    me = const.m_e
    num = (2 * me**2 * (4 * E_nu**2 * t + m_N**4 - m_N**2 * t)
           + 4 * E_nu * me * t * (t - m_N**2) + m_N**2 * t * (m_N**2 - t))
    return -const.alphaQED * d**2 * num / (2 * me**2 * E_nu**2 * t**2)


def dipole_electron_Te_limits(E_nu, m_N):
    """Kinematic limits (Te_min, Te_max) [GeV] of the electron recoil kinetic
    energy in nu e- -> N e-, from the physical t range (Te = -t / 2 m_e).
    Below threshold (sqrt(s) < m_N + m_e, i.e. E_nu < ~ m_N^2/2m_e) returns
    Te_min = Te_max = 0.
    """
    me = const.m_e
    E_nu = np.asarray(E_nu, float)
    s = me**2 + 2.0 * me * E_nu
    sq = np.sqrt(s)
    open_ch = sq > (m_N + me)
    p1 = (s - me**2) / (2.0 * sq)
    E3 = (s + m_N**2 - me**2) / (2.0 * sq)
    p3 = np.sqrt(np.clip(E3**2 - m_N**2, 0.0, None))
    t0 = (m_N**2 / (2.0 * sq)) ** 2 - (p1 - p3) ** 2   # least negative
    t1 = (m_N**2 / (2.0 * sq)) ** 2 - (p1 + p3) ** 2   # most negative
    Te_min = np.where(open_ch, -t0 / (2.0 * me), 0.0)
    Te_max = np.where(open_ch, -t1 / (2.0 * me), 0.0)
    return Te_min, Te_max


def dipole_electron_dsigma_dTe(Te, E_nu, m_N, d):
    """dsigma/dTe [cm^2/GeV] of the electron recoil in nu e- -> N e-
    (t = -2 m_e Te, so dsigma/dTe = 2 m_e |dsigma/dt|), zero outside the
    kinematic Te range. Te and E_nu broadcast.
    """
    me = const.m_e
    Te = np.asarray(Te, float)
    E_nu = np.asarray(E_nu, float)
    Te_min, Te_max = dipole_electron_Te_limits(E_nu, m_N)
    t = -2.0 * me * Te
    with np.errstate(divide="ignore", invalid="ignore"):
        ds = 2.0 * me * dipole_electron_dsigma_dt(t, E_nu, m_N, d)
    inside = (Te > Te_min) & (Te < Te_max)
    return np.where(inside, np.clip(ds, 0.0, None), 0.0) * const.invGeV2_to_cm2


# ============================================================================
# Shared: lab decay length and selection-cut efficiencies
# ============================================================================
def decay_length(E_N, m_N, width):
    """Lab-frame decay length lambda = gamma*beta*c*tau [cm] for a given rest width [GeV]."""
    E_N = np.asarray(E_N, float)
    gamma = E_N / m_N
    beta = np.sqrt(np.clip(1.0 - 1.0 / gamma**2, 0.0, 1.0))
    ctau = const.invGeV_to_cm / width
    return gamma * beta * ctau


def decay_in_window(lam, L, min_displacement=0.0):
    """Per-neutrino expected decays for production uniform along a chord L, with the
    decay required at least ``min_displacement`` downstream of the production point
    and before the exit of the block:

        integral_0^{L-d} ds [ e^{-d/lam} - e^{-(L-s)/lam} ]
            = e^{-d/lam} (L - d) - lam (e^{-d/lam} - e^{-L/lam})

    Reduces to L - lam(1 - e^{-L/lam}) for d = 0 (the uncut case), to
    (L-d)^2 / (2 lam) in the long-lived limit, and to 0 for lam << d (a decay
    displaced by at least d is impossible for a promptly decaying N).
    Units: all lengths in cm; ``lam`` and ``L`` may be arrays.
    """
    lam = np.asarray(lam, float)
    L = np.asarray(L, float)
    d = float(min_displacement)
    lam_safe = np.maximum(lam, 1e-300)

    # window = e^{-d/lam} * [ (L-d) + lam * expm1(-(L-d)/lam) ], written to avoid
    # the catastrophic cancellation of L - lam(1 - e^{-L/lam}) when lam >> L
    # (series branch: bracket -> (L-d)^2/(2 lam) (1 - (L-d)/(3 lam) + ...)).
    Ld = np.clip(L - d, 0.0, None)
    x = Ld / lam_safe
    with np.errstate(over="ignore", invalid="ignore"):
        bracket = np.where(
            x < 1e-4,
            lam_safe * x * x / 2.0 * (1.0 - x / 3.0),
            Ld + lam * np.expm1(-x),
        )
    window = np.exp(-d / lam_safe) * bracket
    return np.where(L > d, np.clip(window, 0.0, None), 0.0)


def pt_cut_efficiency(m_N, pT_min):
    """Signal efficiency of requiring visible pT >= pT_min [GeV] on the HNL decay.

    Exact for the two-body decay N -> nu gamma (dipole portal): the photon carries
    pT = (m_N/2) sin(theta*) with isotropic cos(theta*), so
        eff = sqrt(1 - (2 pT_min / m_N)^2)   for m_N > 2 pT_min, else 0.
    For three-body decays (mixing portal, N -> nu l l ...) the same expression is a
    first-pass proxy (the visible pair carries pT up to ~m_N/2 with a softer
    spectrum, so this slightly overestimates the efficiency near threshold).
    """
    if pT_min <= 0.0:
        return np.ones_like(np.asarray(m_N, float)) if np.ndim(m_N) else 1.0
    m_N = np.asarray(m_N, float)
    x = 2.0 * pT_min / m_N
    return np.where(x < 1.0, np.sqrt(np.clip(1.0 - x**2, 0.0, 1.0)), 0.0)


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

    def __init__(self, m_N, nuflavor="numubar", y_mean=0.3, br_vis=None, n_eff=None):
        self.m_N = m_N
        self.nuflavor = nuflavor
        self.flavor = _flavor_of(nuflavor)
        self.y_mean = y_mean
        # Signal = decays into fully mass-reconstructable (no-neutrino) final
        # states: the channel-resolved visible branching ratio at this mass.
        # Pass br_vis explicitly to override (e.g. a flat efficiency).
        self.br_visible = (
            hnl_widths.visible_br(m_N, flavor=self.flavor) if br_vis is None else br_vis
        )
        self.n_eff = n_eff  # deprecated, unused (kept for call-signature compatibility)

    def production_xsec(self, E_nu, U2, material):
        # per-nucleon NC-DIS
        return mixing_upscattering_xsec(E_nu, self.m_N, U2, self.nuflavor)

    def target_density(self, material):
        return material.N  # nucleons / cm^3

    def width(self, m_N, U2):
        return mixing_total_width(m_N, U2, flavor=self.flavor)


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
        return dt.nuclei_density(material)  # nuclei / cm^3

    def width(self, m_N, d):
        return dipole_radiative_width(m_N, d)


class DipoleDISPortal(Portal):
    """Dipole-portal HNL with DEEP-INELASTIC (visible-vertex) production only.

    Same coupling d [GeV^-1] and radiative decay as DipolePortal, but the
    production cross section is the per-nucleon leading-log DIS estimate
    (dipole_dis_xsec): no Z^2 coherence, in exchange for a visible hadronic
    recoil at the upscattering vertex -- the sample that supports the
    displaced-vertex (min-displacement) selection. As in DIS, the HNL carries
    a fraction (1 - y_mean) of the neutrino energy.
    """

    def __init__(self, m_N, y_mean=0.3, br_vis=1.0, Q2_min=1.0, coeff=2.0):
        self.m_N = m_N
        self.y_mean = y_mean
        self.br_visible = br_vis   # decay photon (displaced from the vertex)
        self.Q2_min = Q2_min
        self.coeff = coeff

    def production_xsec(self, E_nu, d, material):
        return dipole_dis_xsec(
            E_nu, self.m_N, d, material.Z, material.A,
            Q2_min=self.Q2_min, coeff=self.coeff,
        )

    def target_density(self, material):
        return material.N  # nucleons / cm^3

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

    CONVENTION WARNING: DarkNews' internal dipole coupling is Tij = mu_tr/2, and it
    is Tij that plays the role of our d (L = d nubar sigma N F). So the returned
    sigma corresponds to d = ref_coupling/2. Validated numerically against the
    exact elastic formula of arXiv:2011.04751 (see validate_dipole_xsecs.py);
    DarkNewsPortal applies the factor internally.
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
        # CONVENTION (validated numerically to ~2%, see validate_dipole_xsecs.py):
        # DarkNews' dipole vertex coupling is Tij = mu_tr/2, and our d equals Tij.
        # To tabulate sigma_ref at d = ref_coupling we must hand DarkNews
        # mu_tr = 2 * ref_coupling. (Mixing couplings have no such factor.)
        dn_ref = 2.0 * ref_coupling if coupling_kind == "dipole" else ref_coupling
        self._sigma_ref = darknews_upscattering_xsec(
            self._E_grid, m_N, coupling_kind=coupling_kind, target=target,
            nu_flavor=nu_flavor, regimes=regimes, ref_coupling=dn_ref, NEVAL=NEVAL,
        )

    def production_xsec(self, E_nu, coupling, material):
        sigma = _loglog_interp(E_nu, self._E_grid, self._sigma_ref)
        if self.coupling_kind == "dipole":     # sigma ~ d^2, coupling = d
            return sigma * (coupling / self._ref) ** 2
        return sigma * (coupling / self._ref**2)  # sigma ~ |U|^2, coupling = |U|^2

    def target_density(self, material):
        return dt.nuclei_density(material)  # DarkNews total_xsec is per nucleus

    def width(self, m_N, coupling):
        if self.coupling_kind == "dipole":
            return dipole_radiative_width(m_N, coupling)
        return mixing_total_width(m_N, coupling, flavor=_flavor_of(self.nu_flavor))


# ============================================================================
# DarkNews-everything portals: per-material EXACT production cross sections
# ============================================================================
# Materials are mapped to DarkNews nuclear targets by (Z, A). Standard rock
# (Z=11, A=22, the effective-medium convention) is proxied by Na23 (Z=11, A=23);
# the nucleus number density still uses the mint material's own A.
_DN_TARGET_FOR_ZA = {
    (6, 12): "C12",          # graphite: the benchmark vertex-tracker plates
    (74, 184): "W184",
    (18, 40): "Ar40",
    (11, 22): "Na23",
    (11, 23): "Na23",
    (26, 56): "Fe56",
    (14, 28): "Si28",
}

_DN_XSEC_CACHE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".dn_dipole_xsec_cache.npz"
)
_DN_XSEC_CACHE = None
_DN_PDF_CACHE = {}


def _material_key(material):
    material = dt.dominant_nucleus(material)
    return (int(round(material.Z)), int(round(material.A)))


def _dn_cache():
    global _DN_XSEC_CACHE
    if _DN_XSEC_CACHE is None:
        if os.path.exists(_DN_XSEC_CACHE_FILE):
            with np.load(_DN_XSEC_CACHE_FILE) as f:
                _DN_XSEC_CACHE = {k: f[k] for k in f.files}
        else:
            _DN_XSEC_CACHE = {}
    return _DN_XSEC_CACHE


def _dn_cache_get(key):
    c = _dn_cache()
    if f"{key}__E" in c:
        return c[f"{key}__E"], c[f"{key}__sig"]
    return None


def _dn_cache_put(key, E, sig):
    c = _dn_cache()
    c[f"{key}__E"] = np.asarray(E, float)
    c[f"{key}__sig"] = np.asarray(sig, float)
    np.savez_compressed(_DN_XSEC_CACHE_FILE, **c)


def darknews_dis_xsec(
    E_grid, m_N, target="W184", ref_d=1e-3, Q2_min=2.0, nx=48, ny=48,
    pdf_name="CT18NNLO",
):
    """Exact PDF-convolved dipole DIS cross section [cm^2/nucleus] at d = ref_d
    (DarkNews mu_tr = 2 ref_d), integrated over the allowed (x, y) with
    Q^2 > Q2_min. Grid-integration of DarkNews' dis_diff_xsec_dxdy."""
    from DarkNews import phase_space as dn_ps
    from DarkNews import pdf as dnpdf
    from DarkNews import amplitudes as dn_amps
    from DarkNews import pdg as dn_pdg
    from DarkNews.nuclear_tools import NuclearTarget
    from DarkNews.model import ThreePortalModel
    from DarkNews.processes import UpscatteringProcess

    if pdf_name not in _DN_PDF_CACHE:
        _DN_PDF_CACHE[pdf_name] = dnpdf.mkPDF(pdf_name)
    tgt = NuclearTarget(target)
    mdl = ThreePortalModel(m4=float(m_N), mu_tr_mu4=2.0 * ref_d, mzprime=1.25)
    proc = UpscatteringProcess(
        nu_projectile=dn_pdg.numu, nu_upscattered=dn_pdg.neutrino4,
        nuclear_target=tgt, scattering_regime="DIS",
        TheoryModel=mdl, helicity="conserving")
    proc.target.pdf = _DN_PDF_CACHE[pdf_name]
    M = proc.target.mass

    out = np.zeros(len(E_grid))
    for iE, Enu in enumerate(np.asarray(E_grid, float)):
        xlo = dn_ps.dis_xmin(Enu, m_N, M)
        if xlo >= 1.0:
            continue
        xs = np.geomspace(max(xlo, 1e-6), 1.0, nx)
        dsdx = np.zeros_like(xs)
        for i, xv in enumerate(xs):
            ymin, ymax = dn_ps.dis_ylimits(Enu, xv, m_N, M)
            ylo = max(max(float(ymin), 0.0), Q2_min / (2 * M * Enu * xv))
            if ylo >= ymax:
                continue
            ys = np.geomspace(ylo, float(ymax), ny)
            d2 = dn_amps.dis_diff_xsec_dxdy(Enu, np.full_like(ys, xv), ys, proc)
            dsdx[i] = np.trapezoid(d2, ys)
        out[iE] = np.trapezoid(dsdx, xs)
    return out


class DarkNewsMultiPortal(Portal):
    """Dipole portal with the ELASTIC production cross sections tabulated from
    DarkNews per material (real nuclear form factors; default coherent + p-el),
    resolved by the mint material passed at call time. Tabulations are cached on
    disk (.dn_dipole_xsec_cache.npz) -- the vegas integrals are expensive, so
    only the first run per (mass, target) pays. Decay stays analytic.
    """

    E_MAX = 5e3

    def __init__(self, m_N, materials, regimes=("coherent", "p-el"),
                 n_E=12, ref_d=1e-3, NEVAL=1000, br_vis=1.0):
        self.m_N = m_N
        self.y_mean = 0.0
        self.br_visible = br_vis
        self._ref = ref_d
        self._tables = {}
        E_lo = max(1.2 * m_N, 10.0)
        for mat in materials:
            key = _material_key(mat)
            if key not in _DN_TARGET_FOR_ZA:
                raise KeyError(f"No DarkNews target mapped for material Z,A={key}")
            tgt = _DN_TARGET_FOR_ZA[key]
            if E_lo >= 0.99 * self.E_MAX:      # mass out of kinematic reach
                self._tables[key] = None
                continue
            E_grid = np.geomspace(E_lo, self.E_MAX, n_E)
            ck = f"el_{tgt}_{'+'.join(regimes)}_{m_N:.6g}_{n_E}"
            cached = _dn_cache_get(ck)
            if cached is None:
                sig = darknews_upscattering_xsec(
                    E_grid, m_N, coupling_kind="dipole", target=tgt,
                    nu_flavor="mu", regimes=regimes,
                    ref_coupling=2.0 * ref_d, NEVAL=NEVAL)
                _dn_cache_put(ck, E_grid, sig)
                cached = (E_grid, sig)
            self._tables[key] = cached

    def production_xsec(self, E_nu, d, material):
        tab = self._tables[_material_key(material)]
        if tab is None:
            return np.zeros_like(np.asarray(E_nu, float))
        E, sig = tab
        return _loglog_interp(E_nu, E, sig) * (d / self._ref) ** 2

    def target_density(self, material):
        return dt.nuclei_density(material)  # nuclei / cm^3

    def width(self, m_N, d):
        return dipole_radiative_width(m_N, d)


class DarkNewsDISPortal(Portal):
    """Dipole portal with the DIS production cross section tabulated from
    DarkNews (PDF-convolved, per nucleus) per material -- the visible-vertex
    sample. Cached on disk like DarkNewsMultiPortal. As in DIS, the HNL carries
    a fraction (1 - y_mean) of the neutrino energy.
    """

    E_MAX = 5e3
    M_NUCLEON = const.m_proton

    def __init__(self, m_N, materials, y_mean=0.3, Q2_min=2.0,
                 n_E=10, ref_d=1e-3, br_vis=1.0):
        self.m_N = m_N
        self.y_mean = y_mean
        self.br_visible = br_vis
        self._ref = ref_d
        self._tables = {}
        E_thr = 1.05 * (m_N**2 + Q2_min) / (2.0 * self.M_NUCLEON)
        E_lo = max(E_thr, 10.0)
        for mat in materials:
            key = _material_key(mat)
            tgt = _DN_TARGET_FOR_ZA[key]
            if E_lo >= 0.99 * self.E_MAX:      # mass out of DIS reach
                self._tables[key] = None
                continue
            E_grid = np.geomspace(E_lo, self.E_MAX, n_E)
            ck = f"dis_{tgt}_{m_N:.6g}_{Q2_min:g}_{n_E}"
            cached = _dn_cache_get(ck)
            if cached is None:
                sig = darknews_dis_xsec(E_grid, m_N, target=tgt,
                                        ref_d=ref_d, Q2_min=Q2_min)
                _dn_cache_put(ck, E_grid, sig)
                cached = (E_grid, sig)
            self._tables[key] = cached

    def production_xsec(self, E_nu, d, material):
        tab = self._tables[_material_key(material)]
        if tab is None:
            return np.zeros_like(np.asarray(E_nu, float))
        E, sig = tab
        return _loglog_interp(E_nu, E, sig) * (d / self._ref) ** 2

    def target_density(self, material):
        return dt.nuclei_density(material)  # nuclei / cm^3

    def width(self, m_N, d):
        return dipole_radiative_width(m_N, d)


def darknews_nc_dis_xsec(
    E_grid, m_N, target="W184", ref_U2=1e-4, Q2_min=2.0, nx=48, ny=48,
    pdf_name="CT18NNLO",
):
    """Exact Z-mediated (mixing-portal) DIS upscattering cross section
    [cm^2/nucleus] at |U_mu4|^2 = ref_U2, PDF-convolved with the proper massive-N
    kinematics and the finite-Z propagator (DarkNews dis_diff_xsec_dxdy_NC),
    integrated over the allowed (x, y) with Q^2 > Q2_min."""
    from DarkNews import phase_space as dn_ps
    from DarkNews import pdf as dnpdf
    from DarkNews import amplitudes as dn_amps
    from DarkNews import pdg as dn_pdg
    from DarkNews.nuclear_tools import NuclearTarget
    from DarkNews.model import ThreePortalModel
    from DarkNews.processes import UpscatteringProcess

    if pdf_name not in _DN_PDF_CACHE:
        _DN_PDF_CACHE[pdf_name] = dnpdf.mkPDF(pdf_name)
    tgt = NuclearTarget(target)
    mdl = ThreePortalModel(m4=float(m_N), Umu4=float(np.sqrt(ref_U2)), mzprime=1.25)
    proc = UpscatteringProcess(
        nu_projectile=dn_pdg.numu, nu_upscattered=dn_pdg.neutrino4,
        nuclear_target=tgt, scattering_regime="DIS",
        TheoryModel=mdl, helicity="conserving")
    proc.target.pdf = _DN_PDF_CACHE[pdf_name]
    M = proc.target.mass

    out = np.zeros(len(E_grid))
    for iE, Enu in enumerate(np.asarray(E_grid, float)):
        xlo = dn_ps.dis_xmin(Enu, m_N, M)
        if xlo >= 1.0:
            continue
        xs = np.geomspace(max(xlo, 1e-6), 1.0, nx)
        dsdx = np.zeros_like(xs)
        for i, xv in enumerate(xs):
            ymin, ymax = dn_ps.dis_ylimits(Enu, xv, m_N, M)
            ylo = max(max(float(ymin), 0.0), Q2_min / (2 * M * Enu * xv))
            if ylo >= ymax:
                continue
            ys = np.geomspace(ylo, float(ymax), ny)
            d2 = dn_amps.dis_diff_xsec_dxdy_NC(Enu, np.full_like(ys, xv), ys, proc)
            dsdx[i] = np.trapezoid(d2, ys)
        out[iE] = np.trapezoid(dsdx, xs)
    return out


class DarkNewsMixingPortal(Portal):
    """Mixing-portal HNL (coupling = |U_aN|^2) with the production cross section
    from DarkNews' Z-mediated DIS (proper massive-N kinematics, PDFs, finite-Z
    propagator), tabulated per (mass, material) and disk-cached. This replaces
    the crude |U|^2 sigma_NC^SM(E) x linear-threshold estimate of MixingPortal.

    The elastic regimes (coherent CEvNS-like + nucleon-elastic) are NOT included:
    at TeV energies they are ~4-5 orders of magnitude below NC-DIS per nucleon.
    NC-DIS is flavor-universal, so ``nuflavor`` only selects the decay widths and
    the mass-reconstructable branching ratio (channel-resolved, hnl_widths).
    """

    E_MAX = 5e3
    M_NUCLEON = const.m_proton

    def __init__(self, m_N, materials, nuflavor="numubar", y_mean=0.3,
                 Q2_min=2.0, n_E=10, ref_U2=1e-4, br_vis=None):
        self.m_N = m_N
        self.nuflavor = nuflavor
        self.flavor = _flavor_of(nuflavor)
        self.y_mean = y_mean
        self.br_visible = (
            hnl_widths.visible_br(m_N, flavor=self.flavor) if br_vis is None else br_vis
        )
        self._ref = ref_U2
        self._tables = {}
        E_thr = 1.05 * (m_N**2 + Q2_min) / (2.0 * self.M_NUCLEON)
        E_lo = max(E_thr, 10.0)
        for mat in materials:
            key = _material_key(mat)
            tgt = _DN_TARGET_FOR_ZA[key]
            if E_lo >= 0.99 * self.E_MAX:      # mass out of DIS reach
                self._tables[key] = None
                continue
            E_grid = np.geomspace(E_lo, self.E_MAX, n_E)
            ck = f"ncdis_{tgt}_{m_N:.6g}_{Q2_min:g}_{n_E}"
            cached = _dn_cache_get(ck)
            if cached is None:
                sig = darknews_nc_dis_xsec(E_grid, m_N, target=tgt,
                                           ref_U2=ref_U2, Q2_min=Q2_min)
                _dn_cache_put(ck, E_grid, sig)
                cached = (E_grid, sig)
            self._tables[key] = cached

    def production_xsec(self, E_nu, U2, material):
        tab = self._tables[_material_key(material)]
        if tab is None:
            return np.zeros_like(np.asarray(E_nu, float))
        E, sig = tab
        return _loglog_interp(E_nu, E, sig) * (U2 / self._ref)

    def target_density(self, material):
        return dt.nuclei_density(material)  # nuclei / cm^3

    def width(self, m_N, U2):
        return mixing_total_width(m_N, U2, flavor=self.flavor)


def signal_events(
    E_nu, w_nu, portal, coupling, detector, chord_cm=None, min_displacement_cm=0.0
):
    """Expected number of visible HNL decays for a flux (E_nu, w_nu) and a portal.

    Args:
        E_nu, w_nu: neutrino energies [GeV] and weights (number of neutrinos crossing
            the detector over the exposure of interest).
        portal: a MixingPortal or DipolePortal (carries m_N and model parameters).
        coupling: |U|^2 (mixing) or d [GeV^-1] (dipole).
        detector: ForwardDetector.
        chord_cm: optional per-neutrino path length through the detector volume [cm]
            (e.g. from mint.detector_tools.CylinderVolume.intersect). Defaults to the
            full detector length for every neutrino; passing the ray-traced chords
            treats edge-clipping trajectories correctly (production and decay both
            happen along the actual chord).
        min_displacement_cm: required displacement between the production
            (upscattering) vertex and the decay vertex, to reject prompt SM
            backgrounds. Removes short-lived HNLs (lambda << cut), closing the reach
            band at strong coupling.
    """
    E_nu = np.atleast_1d(np.asarray(E_nu, float))
    w_nu = np.atleast_1d(np.asarray(w_nu, float))

    sigma = portal.production_xsec(E_nu, coupling, detector.material)
    n_target = portal.target_density(detector.material)
    E_N = (1.0 - portal.y_mean) * E_nu
    lam = decay_length(E_N, portal.m_N, portal.width(portal.m_N, coupling))

    if chord_cm is None:
        L = detector.length_cm
    else:
        L = np.atleast_1d(np.asarray(chord_cm, float))
    # production uniform along L; decay displaced by >= min_displacement_cm and
    # inside the block
    p = n_target * sigma * decay_in_window(lam, L, min_displacement_cm)
    return float(np.sum(w_nu * portal.br_visible * p))


@dataclass
class BeamDumpGeometry:
    """Beam-dump geometry: HNLs are produced by upscattering in the rock column that
    begins at the end of the collider straight section and runs up to the fiducial
    detector, then decay inside the fiducial volume.

    For the analytic portals the rock cross section is computed on ``rock`` directly;
    for a ``DarkNewsPortal`` the precomputed cross section is used, so build that portal
    with a rock-like target (e.g. ``target="Na23"``) matching ``rock``.
    """

    straight_half_length_cm: float          # rock starts here (end of the MuC straight)
    det_distance_cm: float                  # fiducial decay volume starts here
    fiducial_length_cm: float               # decay-volume length
    rock: dt.Material = field(default_factory=lambda: dt.standard_rock)
    fiducial_produces: bool = False         # dense fiducial also upscatters?
    fiducial_material: object = None        # fiducial material (defaults to rock)
    gap_cm: float = 0.0                      # empty gap between rock end and fiducial

    @property
    def rock_column_cm(self):
        # rock ends gap_cm upstream of the fiducial
        return max(self.det_distance_cm - self.gap_cm - self.straight_half_length_cm, 0.0)


def signal_events_beamdump(E_nu, w_nu, portal, coupling, geom, pT_min=0.0):
    """Expected visible HNL decays for the beam-dump geometry.

    HNLs upscatter in the rock column [straight end -> detector] and must decay inside
    the fiducial volume [D, D+L_fid]:

        P/nu = n_rock sigma_rock * lambda (1 - e^{-D_rock/lambda}) * (1 - e^{-L_fid/lambda})
               [ + in-fiducial "block" production if geom.fiducial_produces ]

    In the long-lived regime lambda*(1 - e^{-D_rock/lambda}) -> D_rock, so the signal
    grows ~linearly with detector distance (more upstream production), saturates near
    D_rock ~ lambda, then falls (HNLs decaying before reaching the fiducial). ``w_nu`` is
    the flux crossing the detector face (the HNL inherits the neutrino direction).

    A geom.gap_cm empty gap between the end of the rock and the fiducial decay
    volume adds a survival factor e^{-gap/lambda}: HNLs must live long enough to
    cross the gap and reach the (reconstructable) fiducial. This closes the reach
    at large coupling -- short-lived HNLs decay in the rock/gap before the fiducial
    -- fixing the unphysical high-|U|^2 plateau of the gap-less geometry.

    pT_min [GeV]: rock-produced HNLs have no visible production vertex, so a
    displacement cut is not available; instead require the visible decay pT to
    exceed pT_min (see pt_cut_efficiency). Removes m_N < 2 pT_min.
    """
    E_nu = np.atleast_1d(np.asarray(E_nu, float))
    w_nu = np.atleast_1d(np.asarray(w_nu, float))

    E_N = (1.0 - portal.y_mean) * E_nu
    lam = decay_length(E_N, portal.m_N, portal.width(portal.m_N, coupling))
    lam_safe = np.maximum(lam, 1e-300)
    L_fid = geom.fiducial_length_cm
    D_rock = geom.rock_column_cm
    fid_decay = 1.0 - np.exp(-L_fid / lam_safe)
    gap_survival = np.exp(-geom.gap_cm / lam_safe)

    # upstream rock production; HNL survives the column, crosses the gap, and decays
    # in the fiducial
    sigma_rock = portal.production_xsec(E_nu, coupling, geom.rock)
    n_rock = portal.target_density(geom.rock)
    p = n_rock * sigma_rock * lam * (1.0 - np.exp(-D_rock / lam_safe)) * gap_survival * fid_decay

    # optional: a dense fiducial volume also upscatters (produce-and-decay in place)
    if geom.fiducial_produces:
        mat = geom.fiducial_material or geom.rock
        sigma_fid = portal.production_xsec(E_nu, coupling, mat)
        n_fid = portal.target_density(mat)
        p = p + n_fid * sigma_fid * (L_fid - lam * (1.0 - np.exp(-L_fid / lam_safe)))

    eff_pt = pt_cut_efficiency(portal.m_N, pT_min)
    return float(np.sum(w_nu * portal.br_visible * p) * eff_pt)


def signal_events_detector(
    E_nu, w_nu, portal, coupling, det, min_displacement_cm=0.0,
    prod_kinds=None, decay_kinds=None, n_z=48,
):
    """In-detector HNL signal for a SEGMENTED detector, using its real geometry.

    Supersedes :func:`signal_events_gas_targets`, which assumed the dense
    production targets were embedded inside the gas decay volume. In the
    benchmark detector the vertex tracker sits ~9 m UPSTREAM of the first gas
    module, with air in between, so a decay in the drift space is lost rather
    than reconstructed -- the old geometry would have counted it.

    Production is integrated over every module of ``prod_kinds`` (default: the
    detector's SIGNAL_KINDS, i.e. vertex tracker + gas); the HNL then has to
    decay inside a module of ``decay_kinds`` (default: DECAY_KINDS, the gas),
    at least ``min_displacement_cm`` downstream of its production point.

    Args:
        det: a :class:`mint.detectors.Detector`.
        n_z: production points per module for the z integral.

    Returns the summed weight (events per unit exposure of ``w_nu``).
    """
    from mint.detectors import SIGNAL_KINDS, DECAY_KINDS
    prod_kinds = SIGNAL_KINDS if prod_kinds is None else prod_kinds
    decay_kinds = DECAY_KINDS if decay_kinds is None else decay_kinds

    E_nu = np.atleast_1d(np.asarray(E_nu, float))
    w_nu = np.atleast_1d(np.asarray(w_nu, float))
    E_N = (1.0 - portal.y_mean) * E_nu
    lam = np.maximum(
        decay_length(E_N, portal.m_N, portal.width(portal.m_N, coupling)), 1e-300)
    d = float(min_displacement_cm)

    windows = sorted((m.z0, m.z1) for m in det.modules if m.kind in decay_kinds)
    if not windows:
        return 0.0
    w_a = np.array([a for a, _ in windows])
    w_b = np.array([b for _, b in windows])

    total = np.zeros_like(E_nu)
    for m in det.modules:
        if m.kind not in prod_kinds or m.thickness <= 0:
            continue
        n_t = portal.target_density(m.material)
        if n_t <= 0:
            continue
        sig = portal.production_xsec(E_nu, coupling, m.material)

        # Production points across the module, paired with every decay window.
        # Everything below is one broadcast over (production point x window) and
        # energy: the earlier version looped in Python over z and windows, which
        # cost ~3000 tiny numpy calls per (mass, coupling) point and dominated
        # the whole scan.
        zs = np.linspace(m.z0, m.z1, n_z + 1)
        z_prod = 0.5 * (zs[:-1] + zs[1:])                     # (n_z,)
        lo = np.maximum(w_a[None, :], z_prod[:, None] + d)    # (n_z, n_win)
        hi = np.broadcast_to(w_b[None, :], lo.shape)
        ok = hi > lo
        if not ok.any():
            continue
        d_lo = (lo - z_prod[:, None])[ok][:, None]            # (n_pair, 1)
        d_hi = (hi - z_prod[:, None])[ok][:, None]
        p_dec = (np.exp(-d_lo / lam[None, :])
                 - np.exp(-d_hi / lam[None, :])).sum(axis=0) / n_z
        total = total + w_nu * n_t * sig * m.thickness * p_dec

    return float(np.sum(total) * portal.br_visible)






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


# ============================================================================
# Inverse HNL decay: nubar_e e- -> l- Nbar (s-channel W annihilation)
# ----------------------------------------------------------------------------
# The massive generalization of inverse muon/tau decay: the daughter neutrino
# is an HNL mixing with flavor l, sigma ~ |U_l4|^2. For l = tau this is the
# ONLY production channel for tau-coupled HNLs at this facility (upscattering
# would need a nu_tau flux). Exact tree-level contact limit (s << m_W^2):
#   |M|^2 = 64 G_F^2 |U|^2 (k_nubar_e . k_N)(p_e . p_l)
# (the P_L projectors forbid m_N m_l interference terms), giving
#   sigma = (2 G^2 / pi s)(p_f/p_i) [2 E1 EN E2 El + (2/3) p_i^2 p_f^2]
# which reduces to G^2 s / 3pi (and to xsecs.inverse_lepton_decay_sigma) as
# m_N -> 0. The V-A weight makes the Nbar the SOFT, backward-CM leg (the
# charged lepton is hard) -- mirrored in the E_N sampling below.
# ============================================================================
_M_LEP_OF = {"e": const.m_e, "mu": const.m_mu, "tau": const.m_tau}


def inverse_hnl_decay_sigma(E_nu, m_N, m_lepton):
    """sigma(nubar_e e- -> l- Nbar) per electron [cm^2] at |U_l4|^2 = 1."""
    E_nu = np.atleast_1d(np.asarray(E_nu, float))
    me = const.m_e
    s = me**2 + 2.0 * me * E_nu
    sq = np.sqrt(s)
    E1 = (s - me**2) / (2.0 * sq)
    E2 = (s + me**2) / (2.0 * sq)
    EN = (s + m_N**2 - m_lepton**2) / (2.0 * sq)
    El = (s + m_lepton**2 - m_N**2) / (2.0 * sq)
    lam = np.clip((s - (m_N + m_lepton) ** 2) * (s - (m_N - m_lepton) ** 2), 0.0, None)
    pf = np.sqrt(lam) / (2.0 * sq)
    pi = E1
    sig = (2.0 * const.Gf**2 / (np.pi * s)) * (pf / np.maximum(pi, 1e-300)) * (
        2.0 * (E1 * EN) * (E2 * El) + (2.0 / 3.0) * (pi * pf) ** 2)
    return sig * const.invGeV2_to_cm2


def _ihd_EN_nodes(E_nu, m_N, m_lepton, n_c=12):
    """Lab N energies and normalized V-A weights on cos(theta*) nodes.
    Returns (E_N [n_c, nE], w_c [n_c, nE]); weights sum to 1 per energy."""
    E_nu = np.atleast_1d(np.asarray(E_nu, float))
    me = const.m_e
    s = me**2 + 2.0 * me * E_nu
    sq = np.sqrt(s)
    E1 = (s - me**2) / (2.0 * sq)
    E2 = (s + me**2) / (2.0 * sq)
    EN = (s + m_N**2 - m_lepton**2) / (2.0 * sq)
    El = (s + m_lepton**2 - m_N**2) / (2.0 * sq)
    lam = np.clip((s - (m_N + m_lepton) ** 2) * (s - (m_N - m_lepton) ** 2), 0.0, None)
    pf = np.sqrt(lam) / (2.0 * sq)
    c = (np.arange(n_c) + 0.5) / n_c * 2.0 - 1.0
    c = c[:, None]
    w = (E1 * EN - E1 * pf * c) * (E2 * El - E1 * pf * c)
    # below threshold the CM energies are unphysical (sigma = 0 there anyway);
    # clip so the normalized weights stay finite for every energy
    w = np.clip(w, 0.0, None)
    tot = w.sum(axis=0, keepdims=True)
    w = np.where(tot > 0, w / np.maximum(tot, 1e-300), 1.0 / n_c)
    gam = (E_nu + me) / sq
    beta = np.sqrt(np.clip(1.0 - 1.0 / gam**2, 0.0, 1.0))
    E_N = gam * (EN + beta * pf * c)
    return np.maximum(E_N, m_N * 1.0000001), w


def signal_events_ihd(E, w, m_N, U2, geom, L_det_cm, gas_material, targets,
                      flavor="tau", min_displacement_cm=3.0, n_c=12,
                      br_vis=None):
    """Expected visible N decays from inverse-HNL-decay production on the
    ELECTRONS of the rock column and of the detector (gas + W slabs, where
    the prompt l- also tags the production vertex).

    Visibility = 1 - BR(N -> 3 nu): for tau flavor below m_tau no CC channel
    opens, so the mass peak is not reconstructable -- the signature is a
    displaced vertex with missing energy (stated on the plots).
    """
    E = np.atleast_1d(np.asarray(E, float))
    w = np.atleast_1d(np.asarray(w, float))
    m_l = _M_LEP_OF[flavor]
    sig = U2 * inverse_hnl_decay_sigma(E, m_N, m_l)      # per electron
    if not np.any(sig > 0):
        return 0.0
    Gam = mixing_total_width(m_N, U2, flavor=flavor)
    vis = (1.0 - hnl_widths.invisible_br(m_N, flavor=flavor)
           if br_vis is None else float(br_vis))

    E_N, w_c = _ihd_EN_nodes(E, m_N, m_l, n_c=n_c)
    gb = np.sqrt(np.clip((E_N / m_N) ** 2 - 1.0, 0.0, None))
    lam = np.maximum(gb * const.invGeV_to_cm / Gam, 1e-300)

    rock = geom.rock
    n_e_rock = rock.N * rock.Z / rock.A
    D_rock = geom.rock_column_cm
    L_fid = geom.fiducial_length_cm
    d = float(min_displacement_cm)

    # rock production -> decay inside the fiducial volume
    p_rock = (n_e_rock * lam * (1.0 - np.exp(-D_rock / lam))
              * np.exp(-geom.gap_cm / lam) * (1.0 - np.exp(-L_fid / lam)))
    # in-detector production (gas electrons along L_det + W-slab electrons),
    # decay displaced >= d inside the gas
    n_e_gas = gas_material.N * gas_material.Z / gas_material.A
    p_det = n_e_gas * decay_in_window(lam, L_det_cm, d)
    for z, thick, mat in targets:
        L_down = L_det_cm - z
        if L_down <= d:
            continue
        window = np.clip(np.exp(-d / lam) - np.exp(-L_down / lam), 0.0, None)
        p_det = p_det + mat.N * mat.Z / mat.A * float(thick) * window

    return float(np.sum(w * sig * np.sum(w_c * (p_rock + p_det), axis=0)) * vis)
