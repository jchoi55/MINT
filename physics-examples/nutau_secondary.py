"""Secondary nu_tau flux from primary-neutrino charm production in the rock.

Chain: a primary nu (numubar or nue from the mu+ beam) undergoes charm-producing
CC DIS in the rock column upstream of the detector,

    nu N -> l + c + X,   c -> D_s (fraction f_Ds),   D_s -> tau nu_tau (BR 5.3%),
    tau -> nu_tau + X (BR ~ 1),

yielding TWO tau neutrinos per chain: a soft one from the D_s decay and a harder
one from the tau decay. Both D_s and tau decay promptly at these energies
(gamma c tau ~ cm << interaction length), so the chain is effectively local.

Rough-estimate ingredients (all knobs exposed):
  * sigma_charm from the MINT CC tables (data-driven);
  * f_Ds = 0.08 (charm -> D_s hadronization fraction), BR(D_s -> tau nu) = 0.0532;
  * energy transfer: E_Ds = C_DS * E_nu with C_DS = <y> <z> ~ 0.4 x 0.65 ~ 0.26;
  * D_s -> tau nu: E_nu1 uniform in [0, (1 - m_tau^2/m_Ds^2) E_Ds] (exact for a
    relativistic two-body decay), E_tau = E_Ds - E_nu1;
  * tau -> nu_tau X: E_nu2 uniform in [0, 0.8 E_tau] (mean 0.4 E_tau, mimicking
    the mode-averaged tau decay spectrum);
  * angular acceptance: the nu_tau deviates from the primary direction by
    theta_s ~ PT_EFF / E_nutau (charm pT + decay pT, PT_EFF ~ 0.7 GeV); the
    probability to still cross the detector face of radius R at distance L_rem
    is the Gaussian-cone factor 1 - exp(-theta_det^2 / 2 theta_s^2) with
    theta_det = R / L_rem. This is what kills the soft D_s-line component from
    far-upstream production.

Neglected (sized against De Lellis-Migliozzi-Santorelli, Phys.Rep. 399 (2004) 227):
  * B mesons (D+- -> tau nu IS included in the MC, +7%; D0 has no tau nu mode);
  * QUASI-ELASTIC charm (sigma_QE ~ 4e-40 cm^2, constant: <0.1% of DIS charm
    at TeV energies);
  * DIFFRACTIVE D_s(*): measured (3.1/6.2)e-3 per CC (nu/nubar) at <E> ~
    100-150 GeV; extrapolating sigma_diff ~ s^0.08 vs sigma_CC ~ E gives
    ~2-5% of the DIS D_s yield at 2 TeV -- a few-% systematic on the charm
    nu_tau chains;
  * ASSOCIATED c-cbar (CC: 3.9e-4/CC; NC: 6.5e-3/NC at 154 GeV, gluon-driven
    growth): ~1% of the wrong-sign rates (CC) and an O(5-10%) systematic on
    them at TeV (NC), see the wrong-sign study;
  * secondary attenuation, tau energy loss.

VALIDATION against the charm review (notebook 3, Sec. 5): our
sigma_charm/sigma_CC x B_mu reproduces the world-average dimuon rates
(review Tables 8-9) to a few percent for E > 100 GeV; our effective
semileptonic BR f_h x BR_h = 0.086 matches the measured B_mu = 0.088(6)
(E > 30 GeV) and 0.083(13) (CHORUS). Fragmentation-parameter provenance:
Peterson eps = 0.20 follows the CCFR fit at <E> ~ 140 GeV (eps is
scale-dependent: ~0.075 at sqrt(s) ~ 5 GeV, 0.22(5) at ~10 GeV -- the high
value is the appropriate one for our TeV energies); beta_pT = 1.21 GeV^-2 is
the high-W (W^2 > 30 GeV^2) E531 fit; m_c = 1.70 GeV is the dimuon-standard
(CCFR NLO) effective mass -- the LO-fit average is 1.43(10), but the sampler
threshold factor differs by <2% flux-weighted and the normalization comes
from the data-driven tables either way. Hadronization fractions
(0.60/0.26/0.08 + Lambda_c 0.06 dropped) follow the Bolton E531 reanalysis;
the review's Table 15 (E>30) variant (0.68/0.19/0.07/0.07) shifts B_eff to
0.079, slightly further from the measured B_mu. Units: GeV, cm.
"""

import numpy as np

from mint import const, xsecs

M_TAU = 1.77686
M_DS = 1.96835

F_DS = 0.08              # c -> D_s hadronization fraction
BR_DS_TAUNU = 0.0532     # BR(D_s -> tau nu_tau)
M_DPM = 1.86966          # D+- mass
F_DPM = 0.26             # c -> D+- fraction (E531 re-analysis, via CCFR)
BR_DPM_TAUNU = 1.20e-3   # BR(D+- -> tau nu_tau), PDG
C_DS = 0.26              # E_Ds / E_nu  (= <y> x <z_frag>)
R_TAU2 = (M_TAU / M_DS) ** 2   # 0.815: tau takes most of the D_s
TAU_NU_XMAX = 0.8        # E_nu2 uniform in [0, TAU_NU_XMAX * E_tau] (mean 0.4)
PT_EFF = 0.7             # GeV: effective transverse momentum of the chain


def sigma_charm_cc(E_nu, nuflavor="numubar"):
    """Charm-production CC cross section per nucleon [cm^2].

    Deprecated alias for :func:`mint.xsecs.sigma_charm_CC`, kept so existing
    call sites keep working. Call the mint function directly in new code."""
    return xsecs.sigma_charm_CC(np.asarray(E_nu, float), nuflavor)


def sigma_nutau_cc_avg(E_nu):
    """tau-appearance CC cross section per nucleon [cm^2], nu/nubar average,
    with the tau-mass threshold factor (MINT sigma_nutau_CC)."""
    E_nu = np.asarray(E_nu, float)
    return 0.5 * (xsecs.sigma_nutau_CC(E_nu) + xsecs.sigma_nutaubar_CC(E_nu))


def _cone_acceptance(E_nutau, L_rem_cm, det_radius_cm, pt_eff=PT_EFF):
    """Probability that a nu_tau of energy E with Gaussian angular spread
    theta_s = pt_eff/E (per chain) still crosses the detector face at L_rem."""
    theta_det = det_radius_cm / np.maximum(L_rem_cm, det_radius_cm)
    theta_s = pt_eff / np.maximum(E_nutau, 1e-6)
    return 1.0 - np.exp(-(theta_det**2) / (2.0 * theta_s**2))


def nutau_flux_spectrum(
    Ec, W, nuflavor, rock, s_rock_start_cm, s_rock_end_cm, det_dist_cm,
    det_radius_cm, Enu_edges, n_depth=24, n_kernel=10, pt_eff=PT_EFF,
    c_ds=C_DS,
):
    """Secondary nu_tau + nubar_tau spectrum through the detector face [nu/yr/bin].

    Args:
        Ec, W: binned primary flux through the face (mint.detector_tools.bin_flux).
        rock: mint material (nucleon density rock.N).
        s_rock_start_cm, s_rock_end_cm: rock column along the line of sight.
        det_dist_cm, det_radius_cm: detector face position and radius.
        Enu_edges: output nu_tau energy bins [GeV].
    Returns (flux_Ds_line, flux_tau_line): the two chain components per bin.
    """
    Ec = np.atleast_1d(np.asarray(Ec, float))
    W = np.atleast_1d(np.asarray(W, float))
    sig_c = sigma_charm_cc(Ec, nuflavor)
    chain_norm = F_DS * BR_DS_TAUNU

    s_nodes = np.linspace(s_rock_start_cm, s_rock_end_cm, n_depth + 1)
    s_mid = 0.5 * (s_nodes[1:] + s_nodes[:-1])
    ds = np.diff(s_nodes)

    E_Ds = c_ds * Ec
    # D_s -> tau nu kernel: E_nu1 uniform in [0, (1-R_TAU2) E_Ds]
    u = (np.arange(n_kernel) + 0.5) / n_kernel
    flux1 = np.zeros(len(Enu_edges) - 1)
    flux2 = np.zeros_like(flux1)

    for s, dz in zip(s_mid, ds):
        L_rem = det_dist_cm - s
        prod = rock.N * sig_c * dz * chain_norm * W       # chains/yr from this slab
        for ui in u:
            # nu_tau (1) from the D_s decay
            E1 = ui * (1.0 - R_TAU2) * E_Ds
            w1 = prod / n_kernel
            acc1 = _cone_acceptance(E1, L_rem, det_radius_cm, pt_eff)
            flux1 += np.histogram(E1, bins=Enu_edges, weights=w1 * acc1)[0]
            # nu_tau (2) from the tau decay: E_tau for THIS E1, then a flat kernel
            E_tau = E_Ds - E1
            for uj in u:
                E2 = uj * TAU_NU_XMAX * E_tau
                w2 = prod / n_kernel**2
                acc2 = _cone_acceptance(E2, L_rem, det_radius_cm, pt_eff)
                flux2 += np.histogram(E2, bins=Enu_edges, weights=w2 * acc2)[0]
    return flux1, flux2


# ============================================================================
# Improved, dimuon-inspired kinematics MC
# ----------------------------------------------------------------------------
# Samples the full chain kinematics following the standard nu-charm (dimuon)
# treatments:
#   * LO charm production d2sigma/dxi dy with slow rescaling,
#     xi = x (1 + m_c^2/Q^2), threshold factor (1 - m_c^2/2ME xi);
#     m_c = 1.70 GeV [CCFR NLO, Bazarko et al., hep-ex/9406007];
#   * x sampled from the CCFR NLO strange sea xs(x) ~ (1-x)^6.98 x^-0.164
#     (Table 4, mu^2 = 20 GeV^2); y flat modulated by the threshold factor
#     (nu-qbar / nubar-q scattering off the sea is y-flat at LO);
#   * fragmentation c -> D_s with the Peterson function, eps_P = 0.20, and a
#     Gaussian transverse momentum of the hadron about the quark direction,
#     dn/dpT^2 ~ exp(-beta pT^2), beta = 1.21 GeV^-2 [E531, via CCFR;
#     Kretzer-Mason-Olness hep-ph/0112191 confirm Gaussian smearing is the
#     standard treatment];
#   * current-jet direction from the event kinematics, theta_W ~ Q sqrt(1-y)/(y E)
#     (the charm takes the W direction at LO);
#   * EXACT two-body D_s -> tau nu decay (boost + isotropic cos theta*);
#   * tau -> nu_tau X decays by channel: leptonic (35.2%, dN/dx = 2x^2(3-2x)),
#     pi (10.8%), rho (25.5% + 9.5% "other" treated as rho-like), a1 (19%),
#     each as an exact two-body boost with isotropic cos theta* (tau
#     polarization neglected).
# The chain NORMALIZATION still comes from the data-driven sigma_charm tables
# (the y/x sampling only shapes the kinematics), so no double counting of the
# threshold suppression.
# ============================================================================
M_N_NUCLEON = const.m_proton
M_C_QUARK = 1.70          # GeV, CCFR NLO
PETERSON_EPS = 0.20
BETA_PT = 1.21            # GeV^-2 (E531): dn/dpT^2 ~ exp(-beta pT^2)
XS_B, XS_C = 6.98, 0.164  # xs(x) ~ (1-x)^b x^-c, CCFR Table 4 (mu^2 = 20)

M_PI, M_RHO, M_A1 = 0.1396, 0.775, 1.26
# (label, BR, effective hadron mass or None for leptonic)
TAU_CHANNELS = [
    ("lep", 0.352, None),
    ("pi", 0.108, M_PI),
    ("rho", 0.350, M_RHO),   # rho + "other" hadronic treated as rho-like
    ("a1", 0.190, M_A1),
]


def _inv_cdf_sample(grid, pdf, n, rng):
    cdf = np.cumsum(pdf)
    cdf = cdf / cdf[-1]
    return np.interp(rng.uniform(size=n), cdf, grid)


def _sample_x_strange(n, rng):
    g = np.geomspace(1e-4, 0.95, 400)
    return _inv_cdf_sample(g, (1 - g) ** XS_B * g ** (-XS_C) * g, n, rng)  # extra g: log grid measure


def _sample_peterson(n, rng):
    z = np.linspace(0.02, 0.98, 400)
    pdf = 1.0 / (z * (1 - 1 / z - PETERSON_EPS / (1 - z)) ** 2)
    return _inv_cdf_sample(z, pdf, n, rng)


def _sample_lep_x(n, rng):
    x = np.linspace(1e-3, 1.0, 400)
    return _inv_cdf_sample(x, 2 * x**2 * (3 - 2 * x), n, rng)


def _two_body_nu(E_parent, m_parent, E_star, ct, rng=None):
    """Boost a massless decay product with rest-frame energy E_star and
    cos(theta*) = ct along the parent direction. Returns (E_lab, theta_lab)."""
    gam = E_parent / m_parent
    beta = np.sqrt(np.clip(1 - 1 / gam**2, 0, 1))
    st = np.sqrt(np.clip(1 - ct**2, 0, 1))
    E_lab = gam * E_star * (1 + beta * ct)
    pz = gam * E_star * (ct + beta)
    theta = np.arctan2(E_star * st, np.maximum(pz, 1e-12))
    return E_lab, theta


def _add_angles(rng, *thetas):
    """Combine small-angle deviations as 2D transverse-angle vectors with
    independent random azimuths."""
    tx = np.zeros_like(thetas[0])
    ty = np.zeros_like(thetas[0])
    for th in thetas:
        phi = rng.uniform(0, 2 * np.pi, size=th.shape)
        tx += th * np.cos(phi)
        ty += th * np.sin(phi)
    return np.sqrt(tx**2 + ty**2)


# nu_tau spin-analyzing coefficients alpha_nu per channel: the rest-frame
# angular distribution is (1 + alpha_nu P cos theta)/2 with theta measured from
# the tau spin axis (= lab momentum direction for longitudinal polarization P).
# Two-body: alpha_nu = -alpha_h with alpha_h = (m_tau^2 - 2 m_h^2)/(m_tau^2 + 2 m_h^2)
# (pi: -1 exactly; rho: -0.449; a1: +0.003 ~ 0). Leptonic: the polarized Michel
# form x^2[(3-2x) + P (1-2x) cos theta], whose lab integral reproduces the
# standard Gaisser polynomials g0(y) + P g1(y) (validated in the notebook).
# For tau+ both P and alpha flip sign (CP), so the same code covers both.
ALPHA_NU = {
    "pi": -1.0,
    "rho": -(M_TAU**2 - 2 * M_RHO**2) / (M_TAU**2 + 2 * M_RHO**2),
    "a1": -(M_TAU**2 - 2 * M_A1**2) / (M_TAU**2 + 2 * M_A1**2),
}


def _sample_cos_linear(k, rng):
    """Sample cos theta from the linear pdf (1 + k cos)/2 on [-1, 1] (|k| <= 1),
    via the closed-form inverse CDF (quadratic); k may be an array."""
    k = np.asarray(k, float)
    u = rng.uniform(size=k.shape)
    small = np.abs(k) < 1e-8
    ksafe = np.where(small, 1.0, k)
    c = (-1.0 + np.sqrt(np.clip(1.0 - ksafe * (2.0 - ksafe - 4.0 * u), 0.0, None))) / ksafe
    return np.where(small, 2 * u - 1, np.clip(c, -1, 1))


def _sample_tau_decay(E_tau, rng, P=None, force_channel=None):
    """Sample the nu_tau of a tau decay (channel-resolved, POLARIZED).

    P: longitudinal polarization of the tau along its lab momentum, per event
    (tau- convention; tau+ is identical by CP). None -> unpolarized.
    force_channel: restrict to one channel label (validation use).
    Returns (E_nu lab, angle wrt the tau direction)."""
    n = len(E_tau)
    P = np.zeros(n) if P is None else np.clip(np.asarray(P, float), -1, 1)
    if force_channel is None:
        u = rng.uniform(size=n)
        edges = np.cumsum([br for _, br, _ in TAU_CHANNELS])
        edges = edges / edges[-1]
        ch = np.searchsorted(edges, u)
    else:
        ch = np.full(n, [lab for lab, _, _ in TAU_CHANNELS].index(force_channel))
    E_star = np.empty(n)
    kcos = np.empty(n)
    for k_i, (lab, _, m_h) in enumerate(TAU_CHANNELS):
        sel = ch == k_i
        if not sel.any():
            continue
        if lab == "lep":
            x = _sample_lep_x(sel.sum(), rng)
            E_star[sel] = x * M_TAU / 2
            # conditional angular slope: P (1-2x)/(3-2x)
            kcos[sel] = P[sel] * (1 - 2 * x) / (3 - 2 * x)
        else:
            E_star[sel] = (M_TAU**2 - m_h**2) / (2 * M_TAU)
            kcos[sel] = ALPHA_NU[lab] * P[sel]
    ct = _sample_cos_linear(kcos, rng)
    return _two_body_nu(np.maximum(E_tau, M_TAU * 1.001), M_TAU, E_star, ct)


def _charm_quark(n_mc, Ec, W, nuflavor, rng, min_mass=None):
    """Charm-production kinematics common to all chains: primary energy E
    (sampled from the flux x sigma_charm measure), the charm-quark energy and
    the current-jet angle. Returns (E, E_charm, th_W)."""
    if min_mass is None:
        min_mass = M_DS
    p = W * sigma_charm_cc(Ec, nuflavor)
    p = p / p.sum()
    E = np.asarray(Ec, float)[rng.choice(len(Ec), size=n_mc, p=p)]

    x = _sample_x_strange(n_mc, rng)
    # y flat, rejection-shaped by the slow-rescaling threshold factor
    y = rng.uniform(0.02, 1.0, n_mc)
    for _ in range(20):
        Q2 = 2 * M_N_NUCLEON * E * x * y
        xi = x * (1 + M_C_QUARK**2 / np.maximum(Q2, 1e-3))
        T = np.clip(1 - M_C_QUARK**2 / (2 * M_N_NUCLEON * E * np.clip(xi, x, 1.0)), 0, 1)
        # xi > 1 is kinematically forbidden (no parton) -- hard reject
        redo = (rng.uniform(size=n_mc) > T) | (xi > 1.0)
        if not redo.any():
            break
        y[redo] = rng.uniform(0.02, 1.0, redo.sum())
    Q2 = 2 * M_N_NUCLEON * E * x * y

    # current jet: charm takes the W direction and energy at LO
    E_charm = np.maximum(y * E, min_mass * 1.05)
    th_W = np.sqrt(Q2 * (1 - y)) / (y * E)
    return E, E_charm, th_W


def sample_nutau_chains(n_mc, Ec, W, nuflavor, rng=None):
    """Sample n_mc production chains. Returns (E_nu1, th1, E_nu2, th2): energies
    and angles (wrt the primary direction) of the D_s-line and tau-line nu_tau.
    All chains carry equal weight; normalize externally to the chain rate."""
    rng = np.random.default_rng(rng)
    E, E_charm, th_W = _charm_quark(n_mc, Ec, W, nuflavor, rng, min_mass=M_DS)

    # parent meson: D_s or D+- , picked by chain weight f x BR (kinematics kept
    # exact per parent -- the D+- nu1 endpoint is (1 - m_tau^2/m_D^2) = 0.096
    # vs 0.185 for the D_s)
    w_ds = F_DS * BR_DS_TAUNU
    w_dpm = F_DPM * BR_DPM_TAUNU
    is_ds = rng.uniform(size=n_mc) < w_ds / (w_ds + w_dpm)
    M_P = np.where(is_ds, M_DS, M_DPM)

    # fragmentation: energy via Peterson, pT about the quark direction (Gaussian pT^2)
    z = _sample_peterson(n_mc, rng)
    E_Ds = np.maximum(z * E_charm, M_P * 1.01)
    p_Ds = np.sqrt(E_Ds**2 - M_P**2)
    pT_frag = np.sqrt(rng.exponential(1.0 / BETA_PT, n_mc))
    th_frag = np.arctan2(pT_frag, p_Ds)

    # exact two-body D -> tau nu
    E_star_nu = (M_P**2 - M_TAU**2) / (2 * M_P)
    ct = rng.uniform(-1, 1, n_mc)
    E_nu1, th_d1 = _two_body_nu(E_Ds, M_P, E_star_nu, ct)
    gam = E_Ds / M_P
    beta = np.sqrt(np.clip(1 - 1 / gam**2, 0, 1))
    E_star_tau = (M_P**2 + M_TAU**2) / (2 * M_P)
    p_star = E_star_nu
    E_tau = gam * (E_star_tau - beta * p_star * ct)
    pz_tau = gam * (-p_star * ct + beta * E_star_tau)
    th_tau = np.arctan2(p_star * np.sqrt(np.clip(1 - ct**2, 0, 1)),
                        np.maximum(pz_tau, 1e-12))

    # tau -> nu_tau X by channel, POLARIZED: the tau from the pseudoscalar
    # two-body decay has parent-frame helicity +1 (exact for massless nu).
    # EXACT large-boost spin transport gives the lab longitudinal polarization
    #   P = (ct_tau + beta*)/(1 + beta* ct_tau),  ct_tau = -ct,
    # with beta* = (M_P^2 - m_tau^2)/(M_P^2 + m_tau^2) the tau speed in the
    # parent frame (validated against explicit spin-4-vector boosts to 1e-4;
    # boost-independent for gamma_P >~ 10; reduces to the naive -ct as
    # beta* -> 0; exact at ct = +-1).
    b_star = (M_P**2 - M_TAU**2) / (M_P**2 + M_TAU**2)
    E_nu2, th_d2 = _sample_tau_decay(E_tau, rng,
                                     P=(b_star - ct) / (1.0 - b_star * ct))

    th1 = _add_angles(rng, th_W, th_frag, th_d1)
    th2 = _add_angles(rng, th_W, th_frag, th_tau, th_d2)
    return E_nu1, th1, E_nu2, th2


def nutau_flux_spectrum_mc(
    Ec, W, nuflavor, rock, s_rock_start_cm, s_rock_end_cm, det_dist_cm,
    det_radius_cm, Enu_edges, n_mc=400_000, rng=0,
):
    """MC version of nutau_flux_spectrum with sampled chain kinematics and a
    hard geometric acceptance (on-axis primaries): theta x L_rem < R_det.
    Returns (flux_Ds_line, flux_tau_line) [nu/yr/bin] through the face."""
    rng_ = np.random.default_rng(rng)
    Ec = np.atleast_1d(np.asarray(Ec, float))
    W = np.atleast_1d(np.asarray(W, float))
    # production weighting by sigma_charm(E) is inside the E sampling; the
    # total rate is the column times the flux-weighted sigma_charm
    chains_per_year = (rock.N * (s_rock_end_cm - s_rock_start_cm)
                       * (F_DS * BR_DS_TAUNU + F_DPM * BR_DPM_TAUNU)
                       * float(np.sum(W * sigma_charm_cc(Ec, nuflavor))))

    E1, th1, E2, th2 = sample_nutau_chains(n_mc, Ec, W, nuflavor, rng=rng_)
    s = rng_.uniform(s_rock_start_cm, s_rock_end_cm, n_mc)
    L_rem = det_dist_cm - s
    w = chains_per_year / n_mc

    f1 = np.histogram(E1[th1 * L_rem < det_radius_cm], bins=Enu_edges)[0] * w
    f2 = np.histogram(E2[th2 * L_rem < det_radius_cm], bins=Enu_edges)[0] * w
    return f1, f2


# ============================================================================
# Inverse tau decay: nubar_e e- -> tau- nubar_tau (s-channel W annihilation)
# ----------------------------------------------------------------------------
# Only the mu- beam provides a nubar_e flux, and the threshold is
# E_nu > (m_tau^2 - m_e^2)/(2 m_e) ~ 3.09 TeV -- the multi-TeV tail of the
# Michel spectrum. Each event yields TWO tau-flavored neutrinos: the prompt
# nubar_tau from the vertex and the nu_tau from the tau decay. The CM boost is
# enormous (gamma ~ E/sqrt(s) ~ 2000), so the chain is nearly collinear and the
# geometric acceptance is ~1 even from far-upstream production.
# Total cross section: mint.xsecs.inverse_lepton_decay_sigma (V-A, with mass
# corrections). Angular sampling: exact two-body CM kinematics with the V-A
# weight |M|^2 ~ (k_nubar_e . k_nubar_tau)(p_e . p_tau), tau polarization
# neglected in the subsequent decay.
# ============================================================================
def sigma_itd(E_nu):
    """sigma(nubar_e e- -> tau- nubar_tau) [cm^2/electron]."""
    return xsecs.inverse_lepton_decay_sigma(np.asarray(E_nu, float), "nuebar", "t")


def sample_itd_chains(n_mc, Ec, W, rng=None):
    """Sample inverse-tau-decay chains from a binned nubar_e flux.
    Returns (E_nub_tau, th_prompt, E_nu2, th2): the prompt nubar_tau and the
    tau-decay nu_tau, with angles wrt the primary direction."""
    rng = np.random.default_rng(rng)
    me = const.m_e
    p = W * sigma_itd(Ec)
    if p.sum() <= 0:
        z = np.zeros(0)
        return z, z, z, z
    p = p / p.sum()
    E = np.asarray(Ec, float)[rng.choice(len(Ec), size=n_mc, p=p)]

    s = 2 * me * E + me**2
    sq = np.sqrt(s)
    gam = (E + me) / sq
    beta = np.sqrt(np.clip(1 - 1 / gam**2, 0, 1))
    E1s = (s - me**2) / (2 * sq)              # nubar_e CM energy (along +z)
    Ees = (s + me**2) / (2 * sq)              # electron CM energy (along -z)
    E3s = (s - M_TAU**2) / (2 * sq)           # prompt nubar_tau CM energy
    Ets = (s + M_TAU**2) / (2 * sq)           # tau CM energy
    ps = E3s                                   # CM momentum of the products

    # V-A angular weight: |M|^2 ~ (k1 . k3)(p_e . k_tau); ct = cos theta*(tau)
    ct = rng.uniform(-1, 1, n_mc)
    for _ in range(30):
        k1k3 = E1s * E3s + E1s * ps * ct       # k3 at angle pi - theta*: k1.k3 = E1E3(1 + ct_tau)... sign via -ct3 = ct
        pek2 = Ees * Ets + E1s * ps * ct       # p_e (along -z) . k_tau(theta*)
        w = k1k3 * pek2
        wmax = (E1s * E3s + E1s * ps) * (Ees * Ets + E1s * ps)
        redo = rng.uniform(size=n_mc) > w / np.maximum(wmax, 1e-300)
        if not redo.any():
            break
        ct[redo] = rng.uniform(-1, 1, redo.sum())

    # prompt nubar_tau: CM angle is pi - theta*(tau) -> cos = -ct
    E_nu3 = gam * E3s * (1 - beta * ct)
    pz3 = gam * (-ps * ct + beta * E3s)
    th3 = np.arctan2(ps * np.sqrt(np.clip(1 - ct**2, 0, 1)), np.maximum(pz3, 1e-12))

    # tau
    E_tau = gam * (Ets + beta * ps * ct)
    pzt = gam * (ps * ct + beta * Ets)
    th_tau = np.arctan2(ps * np.sqrt(np.clip(1 - ct**2, 0, 1)), np.maximum(pzt, 1e-12))

    # V-A: the ITD tau- is left-handed in the massless limit -> P = -1
    # (partial depolarization near threshold neglected; ITD is a ~5% source)
    E_nu2, th_d2 = _sample_tau_decay(E_tau, rng, P=-np.ones(n_mc))
    th2 = _add_angles(rng, th_tau, th_d2)
    return E_nu3, th3, E_nu2, th2


def nutau_flux_spectrum_itd_mc(
    Ec, W, rock, s_rock_start_cm, s_rock_end_cm, det_dist_cm, det_radius_cm,
    Enu_edges, n_mc=200_000, rng=0,
):
    """Inverse-tau-decay nu_tau spectra through the detector face [nu/yr/bin],
    from nubar_e annihilation on the ELECTRONS of the rock column.
    Returns (flux_prompt_nubar_tau, flux_tau_line)."""
    rng_ = np.random.default_rng(rng)
    Ec = np.atleast_1d(np.asarray(Ec, float))
    W = np.atleast_1d(np.asarray(W, float))
    n_e = rock.N * rock.Z / rock.A            # electrons / cm^3
    chains_per_year = (n_e * (s_rock_end_cm - s_rock_start_cm)
                       * float(np.sum(W * sigma_itd(Ec))))
    if chains_per_year <= 0:
        z = np.zeros(len(Enu_edges) - 1)
        return z, z

    E3, th3, E2, th2 = sample_itd_chains(n_mc, Ec, W, rng=rng_)
    s = rng_.uniform(s_rock_start_cm, s_rock_end_cm, n_mc)
    L_rem = det_dist_cm - s
    w = chains_per_year / n_mc
    f3 = np.histogram(E3[th3 * L_rem < det_radius_cm], bins=Enu_edges)[0] * w
    f2 = np.histogram(E2[th2 * L_rem < det_radius_cm], bins=Enu_edges)[0] * w
    return f3, f2


# ============================================================================
# Resonant nuebar_e e- -> charged vector charm meson (Brdar, de Gouvea,
# Machado, Plestid, arXiv:2112.03283): s-channel formation on the rock
# electrons -- the sub-PeV analog of the Glashow resonance. Only D*- and
# D_s*- can cascade to nu_tau. Breit-Wigner (paper Eq. 1, J = 1) with
#   Gamma(m -> nuebar_e e-) = G_F^2 f^2 M^3 |V_CKM|^2 / (12 pi)   (Eq. 2).
# The total widths are keV-scale, so the resonance is a ~1e-4 GeV spike in
# E_nu and the flux-integrated rate is evaluated in the narrow-width
# approximation, where the total width cancels:
#   int sigma dE_nu = 24 pi^2 Gamma_enu Br_fi / (2 m_e M),
# with every chain formed at E_res = (M^2 - m_e^2)/(2 m_e) and exactly
# collinear (1 -> 1 formation on an electron at rest).
# nu_tau chains: D_s*- -> D_s- + gamma/pi0 (Br ~ 1), D_s- -> tau- nubar_tau
# (5.32%); D*- -> D- + pi0/gamma (32.3%), D- -> tau- nubar_tau (0.12%) --
# doubly suppressed (|V_cd|^2 formation and the D- leptonic BR). The direct
# m -> tau- nubar_tau mode has Br ~ 4e-6 (the EM width dominates the tiny
# total width) and is neglected. The vector-meson polarization from the V-A
# formation makes the R -> P gamma decay slightly anisotropic; this moves
# E_P by < +-2% and is neglected (isotropic decay).
# ============================================================================
M_DSSTAR = 2.1122
M_DSTAR = 2.0103
F_DSSTAR = 0.274          # GeV, vector decay constants (Bondarenko et al.)
F_DSTAR = 0.25
M_PI0 = 0.13498

_RESONANCES = {
    # M, f, |V_CKM|^2, Br(R -> P + light), m_P, BR(P -> tau nu), m_light
    "Ds*": (M_DSSTAR, F_DSSTAR, 0.975**2, 1.0, M_DS, BR_DS_TAUNU, 0.0),
    "D*": (M_DSTAR, F_DSTAR, 0.221**2, 0.323, M_DPM, BR_DPM_TAUNU, M_PI0),
}


def resonance_line(which):
    """(E_res [GeV], S [cm^2 GeV]): formation energy and narrow-width line
    strength S = int sigma dE for nuebar_e e- -> R -> P + X (Br_fi = Br to
    the tau-parent pseudoscalar P; the P -> tau nu BR is applied later)."""
    M, f, v2, br_P, *_ = _RESONANCES[which]
    me = const.m_e
    gam_enu = const.Gf**2 * f**2 * M**3 * v2 / (12 * np.pi)
    E_res = (M**2 - me**2) / (2 * me)
    S = 24 * np.pi**2 * gam_enu * br_P / (2 * me * M) * const.invGeV2_to_cm2
    return E_res, S


def sample_resonance_chains(n_mc, which, rng=None):
    """nu_tau kinematics of resonance chains (all formed at E_res, collinear).
    Returns (E_nu1, th1, E_nu2, th2): the prompt nubar_tau of P -> tau nu and
    the tau-line nu_tau, with angles wrt the primary direction."""
    rng = np.random.default_rng(rng)
    M, f, v2, br_P, m_P, br_taunu, m_light = _RESONANCES[which]
    E_res, _ = resonance_line(which)

    # R -> P + gamma/pi0: exact two-body, isotropic
    p_star = np.sqrt((M**2 - (m_P + m_light) ** 2)
                     * (M**2 - (m_P - m_light) ** 2)) / (2 * M)
    E_star_P = np.sqrt(m_P**2 + p_star**2)
    ctP = rng.uniform(-1, 1, n_mc)
    gam = E_res / M
    beta = np.sqrt(np.clip(1 - 1 / gam**2, 0, 1))
    E_P = gam * (E_star_P + beta * p_star * ctP)
    th_P = np.arctan2(p_star * np.sqrt(np.clip(1 - ctP**2, 0, 1)),
                      np.maximum(gam * (p_star * ctP + beta * E_star_P), 1e-12))

    # P -> tau nubar_tau: exact two-body (same treatment as the charm chains)
    E_star_nu = (m_P**2 - M_TAU**2) / (2 * m_P)
    ct = rng.uniform(-1, 1, n_mc)
    E_nu1, th_d1 = _two_body_nu(E_P, m_P, E_star_nu, ct)
    gamP = E_P / m_P
    betaP = np.sqrt(np.clip(1 - 1 / gamP**2, 0, 1))
    E_star_tau = (m_P**2 + M_TAU**2) / (2 * m_P)
    E_tau = gamP * (E_star_tau - betaP * E_star_nu * ct)
    pz_tau = gamP * (-E_star_nu * ct + betaP * E_star_tau)
    th_tau = np.arctan2(E_star_nu * np.sqrt(np.clip(1 - ct**2, 0, 1)),
                        np.maximum(pz_tau, 1e-12))

    # tau polarization: same exact spin transport as the charm chain,
    # P = (beta* - ct)/(1 - beta* ct)
    b_star = (m_P**2 - M_TAU**2) / (m_P**2 + M_TAU**2)
    E_nu2, th_d2 = _sample_tau_decay(E_tau, rng,
                                     P=(b_star - ct) / (1.0 - b_star * ct))

    th1 = _add_angles(rng, th_P, th_d1)
    th2 = _add_angles(rng, th_P, th_tau, th_d2)
    return E_nu1, th1, E_nu2, th2


def nutau_flux_spectrum_res_mc(
    E_raw, w_raw, rock, s_rock_start_cm, s_rock_end_cm, det_dist_cm,
    det_radius_cm, Enu_edges, which="Ds*", n_mc=100_000, window_gev=250.0,
    rng=0,
):
    """Resonance-chain nu_tau spectra through the face [nu/yr/bin], from the
    RAW (unbinned) nubar_e face flux (E_raw [GeV], w_raw [nu/yr]). The chain
    rate is n_e * L * dPhi/dE|_{E_res} * S * BR(P -> tau nu), with dPhi/dE
    estimated in a +-window_gev window around E_res.
    Returns (flux_prompt_nubar_tau, flux_tau_line)."""
    rng_ = np.random.default_rng(rng)
    E_raw = np.asarray(E_raw, float)
    w_raw = np.asarray(w_raw, float)
    E_res, S = resonance_line(which)
    br_taunu = _RESONANCES[which][5]

    in_win = np.abs(E_raw - E_res) < window_gev
    dphi_de = w_raw[in_win].sum() / (2 * window_gev)      # nu/yr/GeV at E_res
    n_e = rock.N * rock.Z / rock.A
    chains_per_year = (n_e * (s_rock_end_cm - s_rock_start_cm)
                       * dphi_de * S * br_taunu)
    if chains_per_year <= 0:
        z = np.zeros(len(Enu_edges) - 1)
        return z, z

    E1, th1, E2, th2 = sample_resonance_chains(n_mc, which, rng=rng_)
    s = rng_.uniform(s_rock_start_cm, s_rock_end_cm, n_mc)
    L_rem = det_dist_cm - s
    w = chains_per_year / n_mc
    f1 = np.histogram(E1[th1 * L_rem < det_radius_cm], bins=Enu_edges)[0] * w
    f2 = np.histogram(E2[th2 * L_rem < det_radius_cm], bins=Enu_edges)[0] * w
    return f1, f2


# ============================================================================
# nu_tau from l-tau trident production in the rock (NEPTUNE)
# ----------------------------------------------------------------------------
# nu_mu A -> nu_tau mu- tau+ A' and nu_e A -> nu_tau e- tau+ A' (W-mediated
# "mixed" tridents; conjugate channels for nubar beams): every event yields
# TWO tau-flavored neutrinos -- the prompt nu_tau at the vertex and the
# opposite-sign one from the tau decay. tau+tau- (NC) tridents are neglected
# (threshold-suppressed well below the ltau channels). Cross sections and
# event kinematics from NEPTUNE (coherent Woods-Saxon + diffractive) on
# standard rock (Z = 11, A = 22). Lab four-vectors are reconstructed exactly
# from the generator invariants x1..x6 (closure ~1e-10 on all invariants and
# on p'_nu^2 = 0); the two azimuths the invariants do not fix are uniform.
# Rates are normalized to TridentProcess.sigma_total per nucleus -- the raw
# generator weights are used for shape only (their absolute normalization is
# not preserved by the truncated vegas batch), with the coherent and
# diffractive samples mixed by their cross sections.
# ============================================================================

TRIDENT_Z_ROCK, TRIDENT_A_ROCK = 11, 22        # standard rock
TRIDENT_EG = np.geomspace(60.0, 6000.0, 10)    # anchor energies [GeV]
M_AMU = 0.9315
_M_E_LEP, _M_MU_LEP = 0.000511, 0.1056584
# per (primary flavor, channel): NEPTUNE (nu_flavor, l1, m1, l2, m2, is_nubar).
# "ltau": nu_l A -> nu_tau l- tau+ A (the nu_tau study); "emu": the mixed
# e-mu tridents whose prompt neutrino is the WRONG-SIGN flavor, e.g.
# nu_mu A -> nu_e mu- e+ A and nuebar A -> numubar e+ mu- A.
_TRIDENT_CHANNELS = {
    ("numu", "ltau"): ("mu", "mu", _M_MU_LEP, "tau", M_TAU, False),
    ("numubar", "ltau"): ("mu", "mu", _M_MU_LEP, "tau", M_TAU, True),
    ("nue", "ltau"): ("e", "e", _M_E_LEP, "tau", M_TAU, False),
    ("nuebar", "ltau"): ("e", "e", _M_E_LEP, "tau", M_TAU, True),
    ("numu", "emu"): ("mu", "mu", _M_MU_LEP, "e", _M_E_LEP, False),
    ("numubar", "emu"): ("mu", "mu", _M_MU_LEP, "e", _M_E_LEP, True),
    ("nue", "emu"): ("e", "e", _M_E_LEP, "mu", _M_MU_LEP, False),
    ("nuebar", "emu"): ("e", "e", _M_E_LEP, "mu", _M_MU_LEP, True),
}
_trident_cache = {}


def _neptune():
    try:
        import neptune
    except ImportError as e:
        raise ImportError(
            "the l-tau trident source needs NEPTUNE "
            "(pip install -e ~/Repos/NEPTUNE)") from e
    return neptune


def _trident_model(nuflavor, channel="ltau"):
    nep = _neptune()
    nu_f, l1, _, l2, _, is_nubar = _TRIDENT_CHANNELS[(nuflavor, channel)]
    return nep.TridentSMModel(nu_flavor=nu_f, l1_flavor=l1, l2_flavor=l2,
                              is_nubar=is_nubar)


def _trident_sigmas(nuflavor, channel="ltau"):
    """(sigma_coherent, sigma_diffractive) [cm^2 / rock nucleus] on TRIDENT_EG."""
    key = ("sig", nuflavor, channel)
    if key not in _trident_cache:
        nep = _neptune()
        model = _trident_model(nuflavor, channel)
        coh, dif = [], []
        for E in TRIDENT_EG:
            proc = nep.TridentProcess(model, Z=TRIDENT_Z_ROCK, A=TRIDENT_A_ROCK,
                                      Enu=float(E))
            out = proc.sigma_total(nitn=4, neval=5000)
            coh.append(max(out["coherent"][0], 0.0))
            dif.append(max(out["diffractive"][0], 0.0))
        _trident_cache[key] = (np.array(coh), np.array(dif))
    return _trident_cache[key]


def sigma_trident(E_nu, nuflavor, channel="ltau"):
    """Trident cross section per rock nucleus [cm^2] (coherent + diffractive,
    log-log interpolation of the NEPTUNE table). channel: "ltau" (nu_tau
    study) or "emu" (wrong-sign mixed tridents)."""
    coh, dif = _trident_sigmas(nuflavor, channel)
    tot = coh + dif
    E_nu = np.asarray(E_nu, float)
    lo = np.log(np.clip(tot, 1e-60, None))
    out = np.exp(np.interp(np.log(np.clip(E_nu, 1e-3, None)),
                           np.log(TRIDENT_EG), lo))
    return np.where(E_nu < TRIDENT_EG[0], 0.0, out)


def _trident_lab_kinematics(ev, m1, rng, m2=M_TAU, return_vectors=False):
    """Reconstruct lab four-vector observables from the NEPTUNE invariants.

    Frame chain: q = P_N - P'_N with the elastic target (nucleus for coherent,
    nucleon for diffractive) fixing q0 = -x1/2M; the lepton system
    P = p_nu + q has W^2 = 2 x2 - x1; the P-rest-frame energies follow from
    P.p_i = (x4+x6, x3+x5) and the polar angles wrt the beam direction from
    x6 = p_nu.p_l1 and x5 = p_nu.p_l2; the relative azimuth is fixed by
    p'_nu^2 = 0 (sign sampled), the overall azimuth is uniform.
    Returns (E_nuprompt, th_nuprompt, E_tau, th_tau) in the lab, angles wrt
    the beam neutrino direction."""
    x1, x2, x3 = ev["x1"], ev["x2"], ev["x3"]
    x4, x5, x6 = ev["x4"], ev["x5"], ev["x6"]
    E = ev["Enu"]
    n = len(E)
    M = np.where(np.asarray(ev["mode"]) == "coherent",
                 TRIDENT_A_ROCK * M_AMU, M_N_NUCLEON)

    q0 = -x1 / (2 * M)
    qz = q0 - x2 / E
    qT = np.sqrt(np.clip(q0**2 + x1 - qz**2, 0, None))
    P0, Px, Pz = E + q0, qT, E + qz
    W2 = np.clip(2 * x2 - x1, 1e-9, None)
    W = np.sqrt(W2)

    EnuW = x2 / W
    E1W = (x4 + x6) / W
    E2W = (x3 + x5) / W
    EnpW = W - E1W - E2W
    p1W = np.sqrt(np.clip(E1W**2 - m1**2, 0, None))
    p2W = np.sqrt(np.clip(E2W**2 - m2**2, 0, None))

    c2 = np.clip((EnuW * E2W - x5) / np.clip(EnuW * p2W, 1e-30, None), -1, 1)
    c1 = np.clip((EnuW * E1W - x6) / np.clip(EnuW * p1W, 1e-30, None), -1, 1)
    s1, s2 = np.sqrt(1 - c1**2), np.sqrt(1 - c2**2)
    d12 = (2 * (x4 + x6) + 2 * (x3 + x5) - W2 - m1**2 - m2**2) / 2
    cphi = np.clip((E1W * E2W - d12 - p1W * p2W * c1 * c2)
                   / np.clip(p1W * p2W * s1 * s2, 1e-30, None), -1, 1)
    sphi = np.sqrt(1 - cphi**2) * np.where(rng.uniform(size=n) < 0.5, 1, -1)

    # P-rest-frame 3-vectors, z along the beam-neutrino direction
    v2 = np.stack([p2W * s2, np.zeros(n), p2W * c2], 1)
    v1 = np.stack([p1W * s1 * cphi, p1W * s1 * sphi, p1W * c1], 1)
    vp = -(v1 + v2)
    psi = rng.uniform(0, 2 * np.pi, n)
    cps, sps = np.cos(psi), np.sin(psi)

    def rotz(v):
        return np.stack([cps * v[:, 0] - sps * v[:, 1],
                         sps * v[:, 0] + cps * v[:, 1], v[:, 2]], 1)

    v1, v2, vp = rotz(v1), rotz(v2), rotz(vp)

    beta = np.stack([Px, np.zeros(n), Pz], 1) / P0[:, None]
    b2 = np.sum(beta**2, 1)
    gam = 1 / np.sqrt(np.clip(1 - b2, 1e-12, None))

    def boost(E_, v, sign):
        bv = sign * beta
        bp = np.sum(bv * v, 1)
        gfac = (gam - 1) / np.clip(b2, 1e-30, None)
        return gam * (E_ + bp), v + (gfac * bp + gam * E_)[:, None] * bv

    # beam-neutrino direction as seen in the P rest frame -> frame axes
    pnu_lab = np.stack([np.zeros(n), np.zeros(n), E], 1)
    _, nuW = boost(E, pnu_lab, -1.0)
    zW = nuW / np.linalg.norm(nuW, axis=1)[:, None]
    ref = np.where(np.abs(zW[:, 2:3]) < 0.9,
                   np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    xW = np.cross(ref, zW)
    xW /= np.linalg.norm(xW, axis=1)[:, None]
    yW = np.cross(zW, xW)

    def tolab(EW, v):
        vw = v[:, 0:1] * xW + v[:, 1:2] * yW + v[:, 2:3] * zW
        return boost(EW, vw, +1.0)

    Etau, utau = tolab(E2W, v2)
    Enp, unp = tolab(EnpW, vp)

    def th(u):
        return np.arccos(np.clip(u[:, 2] / np.linalg.norm(u, axis=1), -1, 1))

    if return_vectors:
        E1l, u1 = tolab(E1W, v1)
        return dict(E_nup=Enp, u_nup=unp, E_tau=Etau, u_tau=utau,
                    E_l1=E1l, u_l1=u1, q0=q0, qT=qT, qz=qz)
    return Enp, th(unp), Etau, th(utau)


def _trident_pool(nuflavor, i_anchor, n_pool=40_000, seed=7, channel="ltau"):
    """Unweighted (E_nuprompt, th_nuprompt, E_tau, th_tau) pool at anchor
    energy TRIDENT_EG[i_anchor]: NEPTUNE weighted events resampled by weight
    per regime, regimes mixed by their cross sections. Vary ``seed`` to build
    statistically independent pools (MC error bands): the weighted->unweighted
    resampling has an effective sample size well below n_pool, so repetitions
    sharing one pool underestimate the MC spread."""
    key = ("pool", nuflavor, channel, i_anchor, seed)
    if key not in _trident_cache:
        nep = _neptune()
        E_A = float(TRIDENT_EG[i_anchor])
        _, _, m1, _, m2, _ = _TRIDENT_CHANNELS[(nuflavor, channel)]
        gen = nep.TridentGenerator(_trident_model(nuflavor, channel), Z=TRIDENT_Z_ROCK,
                                   A=TRIDENT_A_ROCK, Enu=E_A, n_events=n_pool,
                                   nitn=4, neval=40_000,
                                   seed=seed + 1000 * i_anchor)
        ev = {k: np.asarray(v) for k, v in gen.generate(verbose=False).items()}
        rng = np.random.default_rng(seed + i_anchor)
        Enp, thnp, Etau, thtau = _trident_lab_kinematics(ev, m1, rng, m2=m2)

        coh, dif = _trident_sigmas(nuflavor, channel)
        f_coh = coh[i_anchor] / max(coh[i_anchor] + dif[i_anchor], 1e-300)
        w = np.asarray(ev["weight"], float)
        rows = []
        for mode, frac in (("coherent", f_coh), ("diffractive", 1 - f_coh)):
            sel = np.asarray(ev["mode"]) == mode
            n_take = int(round(frac * n_pool))
            if n_take == 0 or not sel.any():
                continue
            p = w[sel] / w[sel].sum()
            rows.append(rng.choice(np.flatnonzero(sel), size=n_take, p=p))
        idx = np.concatenate(rows)
        _trident_cache[key] = (Enp[idx], thnp[idx], Etau[idx], thtau[idx])
    return _trident_cache[key]


def sample_trident_chains(n_mc, Ec, W, nuflavor, rng=None, pool_seed=7):
    """Sample n_mc trident chains from the binned primary flux (Ec, W).
    Returns (E_nu1, th1, E_nu2, th2): the prompt vertex nu_tau and the
    tau-decay-line nu_tau (energies and angles wrt the primary direction).
    Equal-weight chains; normalize externally to the chain rate.
    ``pool_seed`` selects the NEPTUNE kinematics pool (vary for error bands)."""
    rng = np.random.default_rng(rng)
    Ec = np.asarray(Ec, float)
    p = np.asarray(W, float) * sigma_trident(Ec, nuflavor)
    p = p / p.sum()
    E = Ec[rng.choice(len(Ec), size=n_mc, p=p)]

    # stochastic two-anchor interpolation: pick the lower/upper bracketing
    # anchor with probability set by the log-E distance, then scale energies
    # as E/E_A and angles as E_A/E. Averaging over the two anchors removes
    # the first-order bias of nearest-anchor scaling (the residual is second
    # order in the anchor spacing).
    iL = np.clip(np.searchsorted(TRIDENT_EG, E) - 1, 0, len(TRIDENT_EG) - 2)
    t = np.log(E / TRIDENT_EG[iL]) / np.log(TRIDENT_EG[iL + 1] / TRIDENT_EG[iL])
    t = np.clip(t, 0.0, 1.0)
    ia = iL + (rng.uniform(size=n_mc) < t)
    E1 = np.empty(n_mc)
    th1 = np.empty(n_mc)
    E_tau = np.empty(n_mc)
    th_tau = np.empty(n_mc)
    for i in np.unique(ia):
        sel = ia == i
        Enp, thnp, Et, tht = _trident_pool(nuflavor, int(i), seed=pool_seed)
        j = rng.integers(0, len(Enp), sel.sum())
        r = E[sel] / TRIDENT_EG[i]
        E1[sel], th1[sel] = Enp[j] * r, thnp[j] / r
        E_tau[sel], th_tau[sel] = Et[j] * r, tht[j] / r

    # tau decay: the W-vertex tau is relativistic and dominantly left-handed
    # (tau- convention P = -1; the tau+ case is identical by CP)
    E2, th_d2 = _sample_tau_decay(E_tau, rng, P=-np.ones(n_mc))
    th2 = _add_angles(rng, th_tau, th_d2)
    return E1, th1, E2, th2


def nutau_flux_spectrum_trident_mc(
    Ec, W, nuflavor, rock, s_rock_start_cm, s_rock_end_cm, det_dist_cm,
    det_radius_cm, Enu_edges, n_mc=200_000, rng=0, pool_seed=7,
):
    """ltau-trident nu_tau spectra through the face [nu/yr/bin] from the
    binned primary flux (Ec, W) of `nuflavor`, produced on the rock NUCLEI
    of the column. Returns (flux_prompt_line, flux_tau_line)."""
    rng_ = np.random.default_rng(rng)
    Ec = np.atleast_1d(np.asarray(Ec, float))
    W = np.atleast_1d(np.asarray(W, float))
    n_nuc = rock.N / TRIDENT_A_ROCK                     # nuclei / cm^3
    chains_per_year = (n_nuc * (s_rock_end_cm - s_rock_start_cm)
                       * float(np.sum(W * sigma_trident(Ec, nuflavor))))
    if chains_per_year <= 0:
        z = np.zeros(len(Enu_edges) - 1)
        return z, z

    E1, th1, E2, th2 = sample_trident_chains(n_mc, Ec, W, nuflavor, rng=rng_,
                                             pool_seed=pool_seed)
    s = rng_.uniform(s_rock_start_cm, s_rock_end_cm, n_mc)
    L_rem = det_dist_cm - s
    w = chains_per_year / n_mc
    f1 = np.histogram(E1[th1 * L_rem < det_radius_cm], bins=Enu_edges)[0] * w
    f2 = np.histogram(E2[th2 * L_rem < det_radius_cm], bins=Enu_edges)[0] * w
    return f1, f2


# ============================================================================
# Wrong-sign neutrinos from neutrino interactions in the rock
# ----------------------------------------------------------------------------
# "Wrong sign" relative to the beam content (mu- beam: {numu, nuebar} -> wrong
# sign = numubar, nue; mu+ beam: {numubar, nue} -> wrong sign = numu, nuebar).
# Two prompt sources survive the rock (shower pi/K are absorbed before they
# decay; tertiary muons stop and decay at rest, far below threshold):
#
#  * CHARM: nu CC charm production, c/cbar -> D0, D+-, D_s hadrons, followed
#    by the 3-body semileptonic decay D -> X l nu. The nu carries the lepton
#    number of the charm hadron: e.g. nuebar -> cbar -> D- -> mu- NUMUBAR X
#    (wrong-sign numubar in the mu- beam) and nue -> c -> D+ -> mu+ NUMU X
#    (wrong-sign numu in the mu+ beam). Same production machinery as the
#    nu_tau chains (_charm_quark + Peterson + pT); leptonic 2-body and 4-body
#    modes neglected; Lambda_c (f ~ 6%, BR_sl ~ 4%) neglected.
#
#  * IMD on the rock electrons (mu- beam flavors only, by lepton number):
#    the PROMPT partner neutrino of inverse muon decay,
#      numu e- -> mu- NU_E (t-channel, isotropic CM)
#      nuebar e- -> mu- NUMUBAR (s-channel annihilation, (1+ct*)^2 weight).
#    Collinear (gamma_CM ~ 10^3) and hard -- near-unit acceptance, like ITD.
# ============================================================================

M_D0 = 1.86484
F_D0 = 0.60              # c -> D0 fraction (dimuon standard)
# inclusive semileptonic BR per lepton flavor (PDG electron-mode values; the
# muon modes agree within uncertainties)
BR_SL = {"D0": 0.0649, "Dpm": 0.1607, "Ds": 0.065}
_SL_SPECIES = ("D0", "Dpm", "Ds")
_SL_FRAC = {"D0": F_D0, "Dpm": F_DPM, "Ds": F_DS}
_SL_MASS = {"D0": M_D0, "Dpm": M_DPM, "Ds": M_DS}
_SL_MHAD = {"D0": 0.4937, "Dpm": 0.4976, "Ds": 0.5479}   # K-, K0bar, eta


def wrongsign_br_eff():
    """Effective semileptonic BR per charm CC event and per lepton flavor:
    sum_h f_h BR_h."""
    return sum(_SL_FRAC[s] * BR_SL[s] for s in _SL_SPECIES)


def _sample_semileptonic_nu(M_D, m_X, n, rng):
    """Neutrino energy in the D rest frame for D -> X l nu with a constant
    vector form factor (massless lepton): accept-reject on the (E_l, E_nu)
    Dalitz plane with |M|^2 = 2 (V.p_l)(V.p_nu) - V^2 (p_l.p_nu),
    V = p_D + p_X. In invariants (D frame): V.p_l = 2 M E_l - q^2/2,
    V.p_nu = 2 M E_nu - q^2/2, p_l.p_nu = q^2/2,
    V^2 = 2(M^2 + m_X^2) - q^2, q^2 = m_X^2 - M^2 + 2M(E_l + E_nu)."""
    E_max = (M_D**2 - m_X**2) / (2.0 * M_D)
    out = np.empty(n)
    got = 0
    # conservative envelope for |M|^2 on the physical region
    wmax = 2.0 * (2 * M_D * E_max) ** 2
    while got < n:
        m = max(2 * (n - got), 1000)
        El = rng.uniform(0.0, E_max, m)
        Ev = rng.uniform(0.0, E_max, m)
        q2 = m_X**2 - M_D**2 + 2.0 * M_D * (El + Ev)
        ok = (q2 >= 0.0) & (q2 <= 4.0 * El * Ev)
        V2 = 2.0 * (M_D**2 + m_X**2) - q2
        w = 2.0 * (2 * M_D * El - q2 / 2) * (2 * M_D * Ev - q2 / 2) - V2 * q2 / 2
        keep = ok & (rng.uniform(0.0, wmax, m) < np.clip(w, 0.0, None))
        k = np.flatnonzero(keep)[: n - got]
        out[got:got + len(k)] = Ev[k]
        got += len(k)
    return out


def sample_wrongsign_charm_chains(n_mc, Ec, W, nuflavor, rng=None):
    """Wrong-sign-neutrino chains from charm: primary `nuflavor` -> charm CC
    -> D hadron -> X l nu (3-body semileptonic). Returns (E_nu, th): decay
    neutrino energy and angle wrt the primary. Equal-weight chains; the chain
    rate is rock.N * L * sum(W sigma_charm) * wrongsign_br_eff(). Kinematics
    are identical for the e and mu decay modes (massless-lepton limit)."""
    rng = np.random.default_rng(rng)
    Ec = np.asarray(Ec, float)
    W = np.asarray(W, float)
    E, E_charm, th_W = _charm_quark(n_mc, Ec, W, nuflavor, rng, min_mass=M_D0)

    # species picked by chain weight f_h x BR_h (kinematics exact per species)
    wts = np.array([_SL_FRAC[s] * BR_SL[s] for s in _SL_SPECIES])
    cum = np.cumsum(wts / wts.sum())
    isp = np.searchsorted(cum, rng.uniform(size=n_mc))
    M_P = np.array([_SL_MASS[s] for s in _SL_SPECIES])[isp]
    m_X = np.array([_SL_MHAD[s] for s in _SL_SPECIES])[isp]

    # fragmentation: Peterson z + Gaussian pT^2 about the quark direction
    z = _sample_peterson(n_mc, rng)
    E_D = np.maximum(z * E_charm, M_P * 1.01)
    p_D = np.sqrt(E_D**2 - M_P**2)
    pT_frag = np.sqrt(rng.exponential(1.0 / BETA_PT, n_mc))
    th_frag = np.arctan2(pT_frag, p_D)

    # 3-body semileptonic decay, isotropic orientation in the D frame
    E_star = np.empty(n_mc)
    for i, s in enumerate(_SL_SPECIES):
        sel = isp == i
        if sel.any():
            E_star[sel] = _sample_semileptonic_nu(_SL_MASS[s], _SL_MHAD[s],
                                                  int(sel.sum()), rng)
    ct = rng.uniform(-1.0, 1.0, n_mc)
    E_nu, th_d = _two_body_nu(E_D, M_P, E_star, ct)

    th = _add_angles(rng, th_W, th_frag, th_d)
    return E_nu, th


def wrongsign_flux_charm_mc(
    Ec, W, nuflavor, rock, s_rock_start_cm, s_rock_end_cm, det_dist_cm,
    det_radius_cm, Enu_edges, n_mc=400_000, rng=0,
):
    """Wrong-sign nu flux through the face [nu/yr/bin] from charm semileptonic
    decays in the rock column, for ONE decay-lepton flavor (e or mu modes have
    the same kinematics and the same BR_SL here -- call once and reuse, or
    scale by the flavor's BR ratio if desired)."""
    rng_ = np.random.default_rng(rng)
    Ec = np.atleast_1d(np.asarray(Ec, float))
    W = np.atleast_1d(np.asarray(W, float))
    chains_per_year = (rock.N * (s_rock_end_cm - s_rock_start_cm)
                       * wrongsign_br_eff()
                       * float(np.sum(W * sigma_charm_cc(Ec, nuflavor))))
    E_nu, th = sample_wrongsign_charm_chains(n_mc, Ec, W, nuflavor, rng=rng_)
    s = rng_.uniform(s_rock_start_cm, s_rock_end_cm, n_mc)
    L_rem = det_dist_cm - s
    w = chains_per_year / n_mc
    return np.histogram(E_nu[th * L_rem < det_radius_cm], bins=Enu_edges)[0] * w


def sample_imd_prompt(n_mc, Ec, W, nuflavor, rng=None):
    """Prompt partner neutrino of inverse muon decay on the rock electrons:
    numu e- -> mu- nu_e (t-channel; isotropic in the CM at these energies) or
    nuebar e- -> mu- numubar (s-channel annihilation; (1+ct*)^2 V-A weight).
    Returns (E_nu, th) wrt the primary direction."""
    from mint import const as _c
    from mint import xsecs as _x
    rng = np.random.default_rng(rng)
    Ec = np.asarray(Ec, float)
    p = np.asarray(W, float) * _x.inverse_lepton_decay_sigma(Ec, nuflavor, "m")
    p = np.clip(p, 0.0, None)
    p = p / p.sum()
    E = Ec[rng.choice(len(Ec), size=n_mc, p=p)]

    m_e, m_mu = _c.m_e, _c.m_mu
    s = 2.0 * m_e * E + m_e**2
    sqs = np.sqrt(s)
    E_star = (s - m_mu**2) / (2.0 * sqs)     # massless prompt nu
    if "bar" in nuflavor:
        # (1+ct)^2 weight -> inverse CDF of the (1+ct)^3 cumulative
        ct = 2.0 * rng.uniform(size=n_mc) ** (1.0 / 3.0) - 1.0
    else:
        ct = rng.uniform(-1.0, 1.0, n_mc)
    # boost the CM: "parent" = the nu+e system (mass sqrt(s), energy E + m_e)
    E_nu, th = _two_body_nu(E + m_e, sqs, E_star, ct)
    return E_nu, th


def wrongsign_flux_imd_mc(
    Ec, W, nuflavor, rock, s_rock_start_cm, s_rock_end_cm, det_dist_cm,
    det_radius_cm, Enu_edges, n_mc=200_000, rng=0,
):
    """Wrong-sign prompt-IMD neutrino flux through the face [nu/yr/bin]:
    nu_e (from numu) or numubar (from nuebar), produced on the rock
    ELECTRONS. Near-collinear, so the whole column contributes."""
    from mint import xsecs as _x
    rng_ = np.random.default_rng(rng)
    Ec = np.atleast_1d(np.asarray(Ec, float))
    W = np.atleast_1d(np.asarray(W, float))
    n_e = rock.N * rock.Z / rock.A
    sig = np.clip(_x.inverse_lepton_decay_sigma(Ec, nuflavor, "m"), 0.0, None)
    events_per_year = (n_e * (s_rock_end_cm - s_rock_start_cm)
                       * float(np.sum(W * sig)))
    if events_per_year <= 0:
        return np.zeros(len(Enu_edges) - 1)
    E_nu, th = sample_imd_prompt(n_mc, Ec, W, nuflavor, rng=rng_)
    s = rng_.uniform(s_rock_start_cm, s_rock_end_cm, n_mc)
    L_rem = det_dist_cm - s
    w = events_per_year / n_mc
    return np.histogram(E_nu[th * L_rem < det_radius_cm], bins=Enu_edges)[0] * w


def wrongsign_flux_trident_mc(
    Ec, W, nuflavor, rock, s_rock_start_cm, s_rock_end_cm, det_dist_cm,
    det_radius_cm, Enu_edges, n_mc=200_000, rng=0, pool_seed=7,
):
    """Wrong-sign PROMPT neutrino flux through the face [nu/yr/bin] from the
    mixed e-mu tridents on the rock nuclei: nu_mu A -> nu_e mu- e+ A,
    nuebar A -> numubar e+ mu- A, and conjugates -- the outgoing neutrino IS
    the wrong-sign flavor. Same anchors/pools/interpolation as the ltau
    tridents (channel="emu"); the charged leptons range out in the rock."""
    rng_ = np.random.default_rng(rng)
    Ec = np.atleast_1d(np.asarray(Ec, float))
    W = np.atleast_1d(np.asarray(W, float))
    n_nuc = rock.N / TRIDENT_A_ROCK
    sig = sigma_trident(Ec, nuflavor, channel="emu")
    chains_per_year = (n_nuc * (s_rock_end_cm - s_rock_start_cm)
                       * float(np.sum(W * sig)))
    if chains_per_year <= 0:
        return np.zeros(len(Enu_edges) - 1)

    p = W * sig
    p = p / p.sum()
    E = Ec[rng_.choice(len(Ec), size=n_mc, p=p)]
    iL = np.clip(np.searchsorted(TRIDENT_EG, E) - 1, 0, len(TRIDENT_EG) - 2)
    t = np.log(E / TRIDENT_EG[iL]) / np.log(TRIDENT_EG[iL + 1] / TRIDENT_EG[iL])
    ia = iL + (rng_.uniform(size=n_mc) < np.clip(t, 0.0, 1.0))
    E1 = np.empty(n_mc)
    th1 = np.empty(n_mc)
    for i in np.unique(ia):
        sel = ia == i
        Enp, thnp, _, _ = _trident_pool(nuflavor, int(i), seed=pool_seed,
                                        channel="emu")
        j = rng_.integers(0, len(Enp), sel.sum())
        r = E[sel] / TRIDENT_EG[i]
        E1[sel], th1[sel] = Enp[j] * r, thnp[j] / r

    s = rng_.uniform(s_rock_start_cm, s_rock_end_cm, n_mc)
    L_rem = det_dist_cm - s
    w = chains_per_year / n_mc
    return np.histogram(E1[th1 * L_rem < det_radius_cm], bins=Enu_edges)[0] * w
