"""Channel-resolved decay widths for a Heavy Neutral Lepton mixing with the
active neutrinos, valid across the full m_N ~ 0.1-50 GeV range used in the
muon-collider forward-flux study.

Provenance
----------
* Meson decay constants and the meson-coupling structure follow Coloma,
  Fernandez-Martinez, Gonzalez-Lopez, Hernandez-Garcia & Pavlovic,
  "GeV-scale neutrinos: interactions with mesons and DUNE sensitivity",
  JHEP (2021) [arXiv:2007.03701] -- their Table 1 (decay constants, eta/eta'
  mixing) and neutral-current coupling coefficients (C_ij, the vector NC
  factors).  That paper provides effective operators up to m_N ~ 2 GeV.
* The closed-form partial-width kinematics (2-body meson modes, 3-body
  leptonic functions f1/f2, L(x), the CC muon-decay function) are the standard
  results (Atre, Han, Pascoli, Zhang 2009 [0901.3589]; Bondarenko, Boyarsky,
  Gorbunov, Ruchayskyy 2018 [1805.08567]), which are consistent with the
  Coloma et al. operators.
* Above ~1 GeV the exclusive meson sum is replaced by the inclusive
  quark-level widths (with a QCD correction), matched to the exclusive result
  by quark-hadron duality.

Normalization anchor
--------------------
Every width is written in units of
    Gamma_0 = G_F^2 m_N^5 / (192 pi^3) * |U_aN|^2 .
In these units a charged-current channel with massless final states has
coefficient 1 (identical to muon decay), and a neutral-current f fbar channel
has coefficient (gL^2 + gR^2) with the chiral couplings gL = T3 - Q s_w^2,
gR = -Q s_w^2. These reproduce the Coloma/Bondarenko C1, C2 coefficients
exactly (validated numerically in the notebook: BRs sum to 1, closed-form
two-body meson widths, and the m_N -> m_mu muon-decay limit).

All masses in GeV, widths in GeV. Mixing is specified per flavor via
``flavor in {"e", "mu", "tau"}``; only that flavor's |U_aN|^2 is turned on.
"""

import numpy as np

from mint import const

# ----------------------------------------------------------------------------
# Constants and particle data
# ----------------------------------------------------------------------------
S2W = const.s2w                 # sin^2(theta_W), on-shell (PDG)
GF = const.Gf
NC = 3                          # quark color factor
GAMMA0_PREFAC = GF**2 / (192.0 * np.pi**3)

_MLEP = {"e": const.m_e, "mu": const.m_mu, "tau": const.m_tau}

# Meson masses [GeV] (PDG); a few not in mint.const are given here.
_M = {
    "pi": const.m_charged_pion, "pi0": const.m_neutral_pion,
    "K": const.m_charged_kaon, "eta": const.m_neutral_eta, "etap": 0.95778,
    "D": 1.86966, "Ds": 1.96835,
    "rho": const.m_charged_rho, "rho0": const.m_neutral_rho,
    "omega": 0.78266, "phi": 1.019461, "Kstar": const.m_charged_kaonstar,
}

# Decay constants from Coloma et al. (2007.03701), Table 1.
# Pseudoscalars in GeV; vectors in GeV^2 (f_V = m_V * g_V convention).
_FP = {"pi": 0.130, "K": 0.156, "D": 0.212, "Ds": 0.249}
_FV = {"rho": 0.171, "omega": 0.155, "phi": 0.232, "Kstar": 0.178}

# eta / eta' effective decay constants for N -> nu (eta, eta'), from the
# octet/singlet mixing of Coloma et al. (Table 1): theta8 = -21.2 deg,
# theta0 = -6.9 deg, f8 = 0.165 GeV, f0 = 0.148 GeV.
_TH8, _TH0, _F8, _F0 = np.deg2rad(-21.2), np.deg2rad(-6.9), 0.165, 0.148
_FP["eta"] = _F8 / np.sqrt(3.0) * np.cos(_TH8) - _F0 / np.sqrt(6.0) * np.sin(_TH0)
_FP["etap"] = _F8 / np.sqrt(3.0) * np.sin(_TH8) + _F0 / np.sqrt(6.0) * np.cos(_TH0)
_FP["pi0"] = 0.130  # same as f_pi

# CKM magnitudes (PDG) for the charged-current meson modes.
_VCKM = {"pi": 0.97373, "K": 0.2243, "D": 0.221, "Ds": 0.987}
# charged vector CKM: rho ~ V_ud, K* ~ V_us
_VCKM_V = {"rho": 0.97373, "Kstar": 0.2243}

# Neutral-vector NC coupling factors kappa_V (Coloma et al. Eqs. 52-54):
# rho0: (1 - 2 s_w^2);  omega: (2/3) s_w^2;  phi: sqrt(2)(1/2 - 2/3 s_w^2).
_KAPPA_V = {
    "rho0": 1.0 - 2.0 * S2W,
    "omega": (2.0 / 3.0) * S2W,
    "phi": np.sqrt(2.0) * (0.5 - (2.0 / 3.0) * S2W),
}
_MV_NEUTRAL = {"rho0": _M["rho0"], "omega": _M["omega"], "phi": _M["phi"]}
_FV_NEUTRAL = {"rho0": _FV["rho"], "omega": _FV["omega"], "phi": _FV["phi"]}

# Matching scale between exclusive-meson and inclusive quark-level hadronic
# widths (quark-hadron duality). ~1 GeV is the standard choice.
M_MATCH = 1.0

# Quarks: (mass [GeV], T3, Q). Top excluded (far above range).
_QUARKS = {
    "u": (0.0022, 0.5, 2.0 / 3.0), "c": (1.27, 0.5, 2.0 / 3.0),
    "d": (0.0047, -0.5, -1.0 / 3.0), "s": (0.095, -0.5, -1.0 / 3.0),
    "b": (4.18, -0.5, -1.0 / 3.0),
}
# CC quark pairs (up-type, down-type, |V_qq'|)
_QUARK_CC = [
    ("u", "d", 0.97373), ("u", "s", 0.2243), ("u", "b", 0.00382),
    ("c", "d", 0.221), ("c", "s", 0.975), ("c", "b", 0.0408),
]


# ----------------------------------------------------------------------------
# Kinematic helper functions
# ----------------------------------------------------------------------------
def _lam(a, b, c):
    """Kallen lambda(a,b,c) = a^2+b^2+c^2-2ab-2ac-2bc."""
    return a * a + b * b + c * c - 2 * (a * b + a * c + b * c)


def _gLgR(T3, Q):
    """Chiral neutral-current couplings gL = T3 - Q s_w^2, gR = -Q s_w^2."""
    return T3 - Q * S2W, -Q * S2W


# Below this x = m/m_N the mass corrections are < 1e-9; use the massless limits.
_XEPS = 1e-6


def _L_func(x):
    """L(x) for the 3-body leptonic pair-production phase space (Bondarenko et al.).
    Defined for 0 < x < 1/2; returns 0 outside.

    Uses the algebraically stable form: the numerator of the standard expression,
    1 - 3x^2 - (1-x^2) sqrt(1-4x^2), rationalizes exactly to
    4x^6 / (1 - 3x^2 + (1-x^2) sqrt(1-4x^2)), removing the catastrophic
    cancellation as x -> 0.
    """
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    m = (x > _XEPS) & (x < 0.5)
    xr = x[m]
    root = np.sqrt(1.0 - 4.0 * xr**2)
    denom = (1.0 - 3.0 * xr**2 + (1.0 - xr**2) * root) * (1.0 + root)
    out[m] = np.log(4.0 * xr**4 / denom)
    return out


def _f1(x):
    """Pair-production phase-space function; f1(0)=1, f1(x>=1/2)=0."""
    x = np.asarray(x, float)
    out = np.ones_like(x)          # massless limit
    out[x >= 0.5] = 0.0            # channel closed
    m = (x > _XEPS) & (x < 0.5)    # full expression only where mass matters
    xr = x[m]
    root = np.sqrt(1.0 - 4.0 * xr**2)
    out[m] = ((1.0 - 14.0 * xr**2 - 2.0 * xr**4 - 12.0 * xr**6) * root
              + 12.0 * xr**4 * (xr**4 - 1.0) * _L_func(xr))
    return out  # noqa: RET504


def _f2(x):
    """Pair-production mass-interference function; f2(0)=0, f2(x>=1/2)=0."""
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    m = (x > _XEPS) & (x < 0.5)
    xr = x[m]
    root = np.sqrt(1.0 - 4.0 * xr**2)
    out[m] = 4.0 * (xr**2 * (2.0 + 10.0 * xr**2 - 12.0 * xr**4) * root
                    + 6.0 * xr**4 * (1.0 - 2.0 * xr**2 + 2.0 * xr**4) * _L_func(xr))
    return out


def _I_cc(y):
    """Muon-decay phase-space function for one massive final fermion,
    y = (m/m_N)^2:  I(y) = 1 - 8y + 8y^3 - y^4 - 12 y^2 ln(y).  I(0)=1."""
    y = np.asarray(y, float)
    out = np.zeros_like(y)
    m = (y > 0) & (y < 1.0)
    yr = y[m]
    out[m] = 1.0 - 8.0 * yr + 8.0 * yr**3 - yr**4 - 12.0 * yr**2 * np.log(yr)
    out[y <= 0] = 1.0
    return out


def _alpha_s(mu):
    """1-loop alpha_s(mu), nf=4, Lambda=0.2 GeV -- for the QCD correction."""
    mu = np.asarray(mu, float)
    b0 = 11.0 - 2.0 / 3.0 * 4.0
    t = np.log(np.maximum(mu, 1.0) ** 2 / 0.2**2)
    return 4.0 * np.pi / (b0 * np.maximum(t, 1.0))


# ----------------------------------------------------------------------------
# Two-body meson widths (exclusive; used below M_MATCH)
# ----------------------------------------------------------------------------
def width_l_pseudoscalar(m_N, flavor, meson, majorana=True):
    """N -> l_a^- P^+ (P = pi, K, D, Ds), charged-current."""
    m_N = np.asarray(m_N, float)
    ml, mP, fP, V = _MLEP[flavor], _M[meson], _FP[meson], _VCKM[meson]
    xl2, xP2 = (ml / m_N) ** 2, (mP / m_N) ** 2
    open_ = m_N > (ml + mP)
    kin = ((1.0 - xl2) ** 2 - xP2 * (1.0 + xl2)) * np.sqrt(
        np.clip(_lam(1.0, xP2, xl2), 0.0, None))
    G = GF**2 * fP**2 * V**2 * m_N**3 / (16.0 * np.pi) * np.clip(kin, 0.0, None)
    G = np.where(open_, G, 0.0)
    return (2.0 if majorana else 1.0) * G


def width_nu_pseudoscalar0(m_N, meson):
    """N -> nu_a P^0 (P^0 = pi0, eta, eta'), neutral-current."""
    m_N = np.asarray(m_N, float)
    mP, fP = _M[meson], _FP[meson]
    xP2 = (mP / m_N) ** 2
    G = GF**2 * fP**2 * m_N**3 / (32.0 * np.pi) * (1.0 - xP2) ** 2
    return np.where(m_N > mP, G, 0.0)


def width_l_vector(m_N, flavor, meson, majorana=True):
    """N -> l_a^- V^+ (V = rho, K*), charged-current."""
    m_N = np.asarray(m_N, float)
    ml, mV, fV, V = _MLEP[flavor], _M[meson], _FV[meson], _VCKM_V[meson]
    xl2, xV2 = (ml / m_N) ** 2, (mV / m_N) ** 2
    open_ = m_N > (ml + mV)
    kin = ((1.0 - xl2) ** 2 + xV2 * (1.0 + xl2) - 2.0 * xV2**2) * np.sqrt(
        np.clip(_lam(1.0, xV2, xl2), 0.0, None))
    G = GF**2 * fV**2 * V**2 * m_N**3 / (16.0 * np.pi * mV**2) * np.clip(kin, 0.0, None)
    G = np.where(open_, G, 0.0)
    return (2.0 if majorana else 1.0) * G


def width_nu_vector0(m_N, meson):
    """N -> nu_a V^0 (V^0 = rho0, omega, phi), neutral-current."""
    m_N = np.asarray(m_N, float)
    mV, fV, kappa = _MV_NEUTRAL[meson], _FV_NEUTRAL[meson], _KAPPA_V[meson]
    xV2 = (mV / m_N) ** 2
    G = (GF**2 * kappa**2 * fV**2 * m_N**3 / (32.0 * np.pi * mV**2)
         * (1.0 + 2.0 * xV2) * (1.0 - xV2) ** 2)
    return np.where(m_N > mV, G, 0.0)


def _cc_hadronic_exclusive(m_N, flavor, majorana=True):
    """Charged-current exclusive meson modes N -> l^- P^+ / l^- V^+ (NO final
    neutrino -> fully mass-reconstructable)."""
    total = np.zeros_like(np.asarray(m_N, float))
    for P in ("pi", "K", "D", "Ds"):
        total = total + width_l_pseudoscalar(m_N, flavor, P, majorana)
    for V in ("rho", "Kstar"):
        total = total + width_l_vector(m_N, flavor, V, majorana)
    return total


def _nc_hadronic_exclusive(m_N):
    """Neutral-current exclusive meson modes N -> nu P^0 / nu V^0 (missing neutrino)."""
    total = np.zeros_like(np.asarray(m_N, float))
    for P in ("pi0", "eta", "etap"):
        total = total + width_nu_pseudoscalar0(m_N, P)
    for V in ("rho0", "omega", "phi"):
        total = total + width_nu_vector0(m_N, V)
    return total


def _hadronic_exclusive(m_N, flavor, majorana=True):
    """Sum of exclusive meson channels (charged-current semileptonic + neutral)."""
    return _cc_hadronic_exclusive(m_N, flavor, majorana) + _nc_hadronic_exclusive(m_N)


# ----------------------------------------------------------------------------
# Inclusive quark-level hadronic widths (used above M_MATCH)
# ----------------------------------------------------------------------------
def _cc_hadronic_inclusive(m_N, flavor, majorana=True):
    """Inclusive charged-current N -> l qqbar' (l + hadrons, no neutrino ->
    fully mass-reconstructable), with a QCD correction."""
    m_N = np.asarray(m_N, float)
    ml = _MLEP[flavor]
    qcd = 1.0 + _alpha_s(m_N) / np.pi
    G0 = GAMMA0_PREFAC * m_N**5
    total = np.zeros_like(m_N)
    for qu, qd, V in _QUARK_CC:
        mu_, md_ = _QUARKS[qu][0], _QUARKS[qd][0]
        open_ = m_N > (ml + mu_ + md_)
        y = ((max(mu_, md_) + ml) / m_N) ** 2   # crude single-scale threshold
        total = total + np.where(open_, NC * V**2 * _I_cc(y), 0.0)
    mfac = 2.0 if majorana else 1.0
    return mfac * G0 * qcd * total


def _nc_hadronic_inclusive(m_N, majorana=True):
    """Inclusive neutral-current N -> nu qqbar (missing neutrino), with QCD."""
    m_N = np.asarray(m_N, float)
    qcd = 1.0 + _alpha_s(m_N) / np.pi
    G0 = GAMMA0_PREFAC * m_N**5
    total = np.zeros_like(m_N)
    for q, (mq, T3, Q) in _QUARKS.items():
        gL, gR = _gLgR(T3, Q)
        x = mq / m_N
        open_ = m_N > 2.0 * mq
        total = total + np.where(open_, NC * ((gL**2 + gR**2) * _f1(x) + gL * gR * _f2(x)), 0.0)
    mfac = 2.0 if majorana else 1.0
    return mfac * G0 * qcd * total


def _hadronic_inclusive(m_N, flavor, majorana=True):
    """Inclusive N -> l qqbar' (CC) + N -> nu qqbar (NC)."""
    return (_cc_hadronic_inclusive(m_N, flavor, majorana)
            + _nc_hadronic_inclusive(m_N, majorana))


# ----------------------------------------------------------------------------
# Leptonic widths (always exact / perturbative)
# ----------------------------------------------------------------------------
def _leptonic(m_N, flavor, majorana=True):
    """3-nu + N -> nu_a l_b l_b (NC[+CC]) + N -> l_a l_b nu_b (CC), in GeV/|U|^2."""
    m_N = np.asarray(m_N, float)
    G0 = GAMMA0_PREFAC * m_N**5
    mfac = 2.0 if majorana else 1.0

    total = np.zeros_like(m_N)
    # invisible N -> nu nu nu : 3 neutrino species, coeff 1/4 each
    total = total + 3.0 * 0.25

    gL_l, gR_l = _gLgR(-0.5, -1.0)  # charged-lepton chiral couplings
    for beta in ("e", "mu", "tau"):
        mb = _MLEP[beta]
        x = mb / m_N
        # N -> nu_a l_b^+ l_b^-  (NC; CC added when beta == flavor via gL -> gL+1)
        gL = gL_l + (1.0 if beta == flavor else 0.0)
        pair = (gL**2 + gR_l**2) * _f1(x) + gL * gR_l * _f2(x)
        total = total + np.where(m_N > 2.0 * mb, np.clip(pair, 0.0, None), 0.0)
        # N -> l_a^- l_b^+ nu_b  (CC, different flavor only)
        if beta != flavor:
            ma = _MLEP[flavor]
            y = (max(ma, mb) / m_N) ** 2
            total = total + np.where(m_N > (ma + mb), _I_cc(y), 0.0)
    return mfac * G0 * total


# ----------------------------------------------------------------------------
# Total width
# ----------------------------------------------------------------------------
def hnl_total_width(m_N, U2, flavor="mu", majorana=True):
    """Channel-resolved total HNL decay width [GeV].

    Parameters
    ----------
    m_N : float or array [GeV]
    U2 : float
        |U_aN|^2 for the single active flavor ``flavor``.
    flavor : {"e", "mu", "tau"}
    majorana : bool
        Majorana (default) sums charge-conjugate final states (factor 2 where
        applicable); Dirac counts each once.

    Returns
    -------
    Gamma_tot [GeV], same shape as m_N.
    """
    m_N = np.asarray(m_N, float)
    scalar = m_N.ndim == 0
    m_N = np.atleast_1d(m_N)

    lep = _leptonic(m_N, flavor, majorana)

    # hadronic: exclusive mesons below the matching scale, inclusive above
    had = np.where(
        m_N < M_MATCH,
        _hadronic_exclusive(m_N, flavor, majorana),
        _hadronic_inclusive(m_N, flavor, majorana),
    )
    # below ~2 m_pi there is no open hadronic channel at all
    had = np.where(m_N > _M["pi0"], had, 0.0)

    total = U2 * (lep + had)
    return float(total[0]) if scalar else total


def reconstructable_width(m_N, flavor="mu", majorana=True):
    """Width into fully mass-reconstructable final states: the charged-current
    hadronic modes N -> l^- + hadrons (N -> l P, l V below ~1 GeV; N -> l qqbar'
    above), which have NO final-state neutrino, so the HNL four-momentum (hence
    m_N) can be reconstructed from the visible decay products. Returns GeV/|U|^2.

    Excludes: all N -> nu ... modes (neutral-current mesons, nu l+ l-, 3nu) and
    the CC leptonic mode l_a l_b nu_b -- all carry missing energy.
    """
    m_N = np.atleast_1d(np.asarray(m_N, float))
    return np.where(
        m_N < M_MATCH,
        _cc_hadronic_exclusive(m_N, flavor, majorana),
        _cc_hadronic_inclusive(m_N, flavor, majorana),
    )


def _total_width_per_U2(m_N, flavor, majorana):
    """Total width / |U|^2 (helper; array in, array out)."""
    m_N = np.atleast_1d(np.asarray(m_N, float))
    lep = _leptonic(m_N, flavor, majorana)
    had = np.where(m_N < M_MATCH,
                   _hadronic_exclusive(m_N, flavor, majorana),
                   _hadronic_inclusive(m_N, flavor, majorana))
    had = np.where(m_N > _M["pi0"], had, 0.0)
    return lep + had


def visible_br(m_N, flavor="mu", majorana=True):
    """Branching ratio into fully mass-reconstructable (no-neutrino) final states.
    This is the signal efficiency used for a displaced-vertex + mass-peak search.
    Scalar in -> scalar out; array in -> array out. |U|^2 cancels."""
    m_N_arr = np.atleast_1d(np.asarray(m_N, float))
    br = reconstructable_width(m_N_arr, flavor, majorana) / _total_width_per_U2(
        m_N_arr, flavor, majorana)
    return float(br[0]) if np.ndim(m_N) == 0 else br


def invisible_br(m_N, flavor="mu", majorana=True):
    """Branching ratio into the fully invisible mode N -> 3 nu."""
    m_N_arr = np.atleast_1d(np.asarray(m_N, float))
    inv = 3.0 * 0.25 * (2.0 if majorana else 1.0) * GAMMA0_PREFAC * m_N_arr**5
    br = inv / _total_width_per_U2(m_N_arr, flavor, majorana)
    return float(br[0]) if np.ndim(m_N) == 0 else br


def branching_ratios(m_N, flavor="mu", majorana=True):
    """Dict of branching-ratio groups at mass m_N (|U|^2 cancels):
    leptonic, hadronic, reconstructable (CC hadronic), invisible (3nu)."""
    lep = float(_leptonic(np.atleast_1d(m_N), flavor, majorana)[0])
    had = _hadronic_exclusive if m_N < M_MATCH else _hadronic_inclusive
    had = float(had(np.atleast_1d(m_N), flavor, majorana)[0]) if m_N > _M["pi0"] else 0.0
    tot = lep + had
    return {
        "leptonic": lep / tot,
        "hadronic": had / tot,
        "reconstructable": visible_br(m_N, flavor, majorana),
        "invisible": invisible_br(m_N, flavor, majorana),
    }
