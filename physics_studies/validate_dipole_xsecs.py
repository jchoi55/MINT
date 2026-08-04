"""Validation suite for the dipole-portal production cross sections in hnl_tools.py.

Run from physics_studies/:  python validate_dipole_xsecs.py

Three independent checks (requires the DarkNews dev branch with the DIS regime,
and the CT18NNLO grid for the `parton` backend: `python -m parton install CT18NNLO`):

(A) COUPLING CONVENTION -- DarkNews proton-elastic on hydrogen vs the exact
    tree-level elastic formula of Jodlowski & Trojanowski (arXiv:2011.04751,
    Eq. A.10, in the d-convention of our hnl_tools) weighted by the standard
    proton dipole form factor. Result: ratio = 1/4 to ~2%, i.e. DarkNews'
    vertex coupling Tij (= mu_tr/2) IS our d. DarkNewsPortal accounts for this.

(B) DIS -- the exact PDF-convolved dipole DIS cross section (DarkNews
    `dis_diff_xsec_dxdy`, after Huang-Jana-Lindner-Rodejohann, CT18NNLO,
    Q^2 > 2 GeV^2) vs the leading-log estimate `hnl_tools.dipole_dis_xsec`.
    Result: agreement to ~10% at valence-dominated energies (E ~ 250 GeV,
    m_N <~ 1 GeV) -- which independently confirms both the leading-log
    coefficient (4 alpha e_q^2 per quark) and the convention of (A) -- growing
    to ~2.5x CONSERVATIVE (underestimate) at 3 TeV as sea quarks enter, and
    OVERESTIMATING by ~2-7x only near the high-mass threshold (m_N ~ 20 GeV
    at E <~ 1 TeV, where the crude threshold treatment is too loose).

(C) COHERENT -- DarkNews' exact coherent cross section on tungsten (real form
    factors), convention-corrected, vs the analytic leading-log
    `hnl_tools.dipole_coherent_xsec` with the default coeff = 2. Result:
    exact/analytic = 0.8-1.5 for m_N = 0.1-2 GeV at 0.5-3 TeV. The naive
    log-accuracy coefficient is 4 (Magill et al. 1803.03262, Eq. 2), but the
    nuclear form factor erodes it; coeff = 2 reproduces the exact result to
    +-50% and is kept as the default.

Since the signal scales as d^2 (short-lived) to d^4 (long-lived), a factor-k
cross-section error moves the reach contours by only k^(1/2) to k^(1/4):
the validated accuracies above correspond to <~25% shifts in d.
"""

import numpy as np
from scipy.integrate import quad

import hnl_tools as hnl
from mint import const

ALPHA = const.alphaQED
M_P = const.m_proton


# ======================================================================
# (A) convention pin: DarkNews p-el on hydrogen vs exact A.10 x proton FF
# ======================================================================
def dsigma_dt_A10(t, E, m, M, d):
    """Exact elastic dsigma/dt [GeV^-4] off a point fermion (charge 1, mass M),
    Eq. (A.10) of arXiv:2011.04751 with m_e -> M and mu_N = d (our convention)."""
    num = (2 * M**2 * (4 * E**2 * t + m**4 - m**2 * t)
           + 4 * E * M * t * (t - m**2) + m**2 * t * (m**2 - t))
    return -ALPHA * d**2 * num / (2 * M**2 * E**2 * t**2)


def t_range(E, m, M):
    """Physical t range for nu(0) M -> N(m) M."""
    s = M**2 + 2 * M * E
    sq = np.sqrt(s)
    p1 = (s - M**2) / (2 * sq)
    E3 = (s + m**2 - M**2) / (2 * sq)
    p3 = np.sqrt(max(E3**2 - m**2, 0.0))
    t0 = ((0 - m**2) / (2 * sq)) ** 2 - (p1 - p3) ** 2   # least negative
    t1 = ((0 - m**2) / (2 * sq)) ** 2 - (p1 + p3) ** 2   # most negative
    return t1, t0


def F_dip2(Q2, L2=0.71):
    """Standard proton dipole form factor squared (squared again for |M|^2)."""
    return 1.0 / (1.0 + Q2 / L2) ** 4


def sigma_A10_proton(E, m, d):
    """A.10 integrated over t (in log|t|, resolving the forward peak) with the
    proton dipole form factor [cm^2]."""
    t1, t0 = t_range(E, m, M_P)
    lo, hi = np.log(-t0), np.log(-t1)
    val, _ = quad(
        lambda u: np.exp(u) * dsigma_dt_A10(-np.exp(u), E, m, M_P, d) * F_dip2(np.exp(u)),
        lo, hi, limit=400,
    )
    return val * const.invGeV2_to_cm2


def check_convention(mu0=1e-3):
    print("=" * 72)
    print("(A) Convention: DarkNews p-el (H1, mu_tr = mu0) vs A.10 (d = mu0)")
    print("    Tij = mu_tr/2 == our d  <=>  ratio = 1/4")
    for m in [0.1, 0.5]:
        for E in [500.0, 2000.0]:
            dn = hnl.darknews_upscattering_xsec(
                np.array([E]), m, "dipole", "H1", regimes=("p-el",),
                ref_coupling=mu0, NEVAL=4000)[0]
            a10 = sigma_A10_proton(E, m, mu0)
            print(f"  m={m:4.1f} E={E:6.0f}:  DN/A10 = {dn / a10:.4f}   (x4 = {4 * dn / a10:.3f})")


# ======================================================================
# (B) DIS: exact (DarkNews + CT18NNLO) vs leading-log estimate
# ======================================================================
def check_dis(d0=1e-3, Q2cut=2.0):
    from DarkNews import phase_space as ps
    from DarkNews import pdf as dnpdf
    from DarkNews import amplitudes as amps
    from DarkNews import pdg
    from DarkNews.nuclear_tools import NuclearTarget
    from DarkNews.model import ThreePortalModel
    from DarkNews.processes import UpscatteringProcess

    pdf = dnpdf.mkPDF("CT18NNLO")
    O16 = NuclearTarget("O16")
    mu_tr0 = 2.0 * d0   # DarkNews mu_tr for our d = d0 (see check_convention)

    def total_dis_xsec(Enu, mN, nx=80, ny=80):
        mdl = ThreePortalModel(m4=mN, mu_tr_mu4=mu_tr0, mzprime=1.25)
        proc = UpscatteringProcess(
            nu_projectile=pdg.numu, nu_upscattered=pdg.neutrino4,
            nuclear_target=O16, scattering_regime="DIS",
            TheoryModel=mdl, helicity="conserving")
        proc.target.pdf = pdf
        M = proc.target.mass
        xlo = ps.dis_xmin(Enu, mN, M)
        if xlo >= 1.0:
            return 0.0
        xs = np.geomspace(max(xlo, 1e-6), 1.0, nx)
        dsdx = np.zeros_like(xs)
        for i, xv in enumerate(xs):
            ymin, ymax = ps.dis_ylimits(Enu, xv, mN, M)
            ylo = max(max(float(ymin), 0.0), Q2cut / (2 * M * Enu * xv))
            if ylo >= ymax:
                continue
            ys = np.geomspace(ylo, float(ymax), ny)
            d2 = amps.dis_diff_xsec_dxdy(Enu, np.full_like(ys, xv), ys, proc)
            dsdx[i] = np.trapezoid(d2, ys)
        return np.trapezoid(dsdx, xs)

    print("=" * 72)
    print(f"(B) DIS per nucleon: exact (CT18NNLO, Q2>{Q2cut:g}) vs hnl.dipole_dis_xsec")
    print(f"{'m_N':>6} {'E_nu':>7} | {'exact/nucl':>11} {'LL':>11} {'exact/LL':>8}")
    for m in [0.1, 1.0, 5.0, 20.0]:
        for E in [250.0, 1000.0, 3000.0]:
            ex = total_dis_xsec(E, m) / O16.A
            ll = float(hnl.dipole_dis_xsec(np.array([E]), m, d0, 8, 16, Q2_min=Q2cut)[0])
            r = ex / ll if ll > 0 else float("nan")
            print(f"{m:6.1f} {E:7.0f} | {ex:11.3e} {ll:11.3e} {r:8.2f}")


# ======================================================================
# (C) coherent: DarkNews exact (W, convention-corrected) vs analytic coeff=2
# ======================================================================
def check_coherent(d0=1e-3):
    print("=" * 72)
    print("(C) Coherent on W: DarkNews (mu_tr = 2 d0) / analytic (coeff=2, d0)")
    E = np.array([500.0, 1000.0, 3000.0])
    for m in [0.1, 0.5, 2.0]:
        dn = hnl.darknews_upscattering_xsec(
            E, m, "dipole", "W184", regimes=("coherent",),
            ref_coupling=2.0 * d0, NEVAL=2000)
        ana = hnl.dipole_coherent_xsec(E, m, d0, 74, 184, coeff=2.0)
        print(f"  m={m}: exact/analytic =", np.round(dn / ana, 3), f" at E = {E} GeV")




# ======================================================================
# (D) electron formula: independent spinor-level verification of Eq. (A.10)
# ======================================================================
# Build |M|^2 for nu e- -> N e- directly from explicit Dirac spinors and the
# dipole vertex 2 d sigma^{mu nu} q_nu (no trace algebra taken from any paper).
# The machinery is anchored by requiring it reproduce Gamma(N -> nu gamma)
# = d^2 m^3 / 4 pi from the same vertex, then compared to
# hnl_tools.dipole_electron_dsigma_dt point by point, and to the classic
# Vogel-Engel magnetic-moment limit (mu_nu = 2d) at m_N -> 0.

_ETA = np.diag([1.0, -1.0, -1.0, -1.0])
_S0 = np.eye(2, dtype=complex)
_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)
_SVEC = [_SX, _SY, _SZ]

_GAMMA = [np.zeros((4, 4), dtype=complex) for _ in range(4)]
_GAMMA[0][:2, 2:] = _S0
_GAMMA[0][2:, :2] = _S0
for i in range(3):
    _GAMMA[i + 1][:2, 2:] = _SVEC[i]
    _GAMMA[i + 1][2:, :2] = -_SVEC[i]
_G5 = np.zeros((4, 4), dtype=complex)
_G5[:2, :2] = -_S0
_G5[2:, 2:] = _S0
_PL = (np.eye(4) - _G5) / 2.0

_SIGMA = [[0.25j * (_GAMMA[m] @ _GAMMA[n] - _GAMMA[n] @ _GAMMA[m]) * 2
           for n in range(4)] for m in range(4)]   # sigma^{mu nu} = i/2 [g^mu, g^nu]


def _sqrtm2(h):
    """Matrix sqrt of a 2x2 hermitian PSD matrix."""
    w, v = np.linalg.eigh(h)
    return (v * np.sqrt(np.clip(w, 0, None))) @ v.conj().T


def _u_spinor(p, m, xi):
    """Dirac spinor (Weyl basis): u = (sqrt(p.sigma) xi, sqrt(p.sigmabar) xi)."""
    E, px, py, pz = p
    psig = E * _S0 - (px * _SX + py * _SY + pz * _SZ)
    psigbar = E * _S0 + (px * _SX + py * _SY + pz * _SZ)
    return np.concatenate([_sqrtm2(psig) @ xi, _sqrtm2(psigbar) @ xi])


def _helicity_spinors(p):
    """(chi_plus, chi_minus) two-spinors along the direction of p."""
    _, px, py, pz = p
    pp = np.sqrt(px**2 + py**2 + pz**2)
    theta = np.arccos(np.clip(pz / pp, -1, 1))
    phi = np.arctan2(py, px)
    chi_p = np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])
    chi_m = np.array([-np.exp(-1j * phi) * np.sin(theta / 2), np.cos(theta / 2)])
    return chi_p, chi_m


def _dipole_vertex(q, d):
    """Gamma^mu = 2 d sigma^{mu nu} q_nu (contravariant mu), as 4 matrices."""
    q_lo = _ETA @ q
    return [2.0 * d * sum(_SIGMA[mu][nu] * q_lo[nu] for nu in range(4))
            for mu in range(4)]


def _ubar(u):
    return u.conj() @ _GAMMA[0]


def numeric_width_Ngamma(m_N, d):
    """Gamma(N -> nu gamma) from the explicit-spinor amplitude [GeV]."""
    # N at rest; photon along +z with energy m/2; nu along -z (left-handed).
    q = np.array([m_N / 2, 0, 0, m_N / 2])
    k = np.array([m_N / 2, 0, 0, -m_N / 2])
    eps = [np.array([0, 1, 1j, 0]) / np.sqrt(2),
           np.array([0, 1, -1j, 0]) / np.sqrt(2)]   # transverse polarizations
    chi_p, chi_m = _helicity_spinors(k)
    u_nu = [_u_spinor(k, 0.0, chi_m), _u_spinor(k, 0.0, chi_p)]  # LH spinor is chi_m
    V = _dipole_vertex(q, d)
    m2 = 0.0
    for sN in range(2):
        xi = np.array([1.0, 0.0]) if sN == 0 else np.array([0.0, 1.0])
        uN = _u_spinor(np.array([m_N, 0, 0, 0.0]), m_N, xi)
        for unu in u_nu:
            for e in eps:
                amp = sum((_ETA @ e.conj())[mu] * (_ubar(unu) @ V[mu] @ (_PL @ uN))
                          for mu in range(4))
                # NOTE: nu_L projection: operator is nubar_L sigma N -> P_R acting
                # left of uN is equivalent to using LH nu spinor; PL here is a
                # belt-and-braces projector on the massless nu side via ubar PL... 
                m2 += abs(amp) ** 2
    m2 /= 2.0   # average over N spin
    return m2 / (16.0 * np.pi * m_N)


def numeric_dsigma_dt(E_nu, m_N, t, d):
    """dsigma/dt [GeV^-4] for nu e- -> N e- from explicit spinors (CM frame)."""
    me = const.m_e
    s = me**2 + 2 * me * E_nu
    sq = np.sqrt(s)
    # CM momenta
    p_i = (s - me**2) / (2 * sq)
    E_e_i = np.sqrt(p_i**2 + me**2)
    E_N = (s + m_N**2 - me**2) / (2 * sq)
    p_f = np.sqrt(max(E_N**2 - m_N**2, 0))
    E_e_f = np.sqrt(p_f**2 + me**2)
    # scattering angle from t = (k1 - k2)^2 = m_N^2 - 2(E_nu_cm E_N - p_i p_f cos)
    E_nu_cm = p_i
    cth = (m_N**2 - t - 2 * E_nu_cm * E_N) / (2 * p_i * p_f) * (-1.0)
    sth = np.sqrt(max(1 - cth**2, 0))
    k1 = np.array([E_nu_cm, 0, 0, p_i])
    p1 = np.array([E_e_i, 0, 0, -p_i])
    k2 = np.array([E_N, p_f * sth, 0, p_f * cth])
    p2 = k1 + p1 - k2
    q = k1 - k2
    tt = q[0]**2 - q[1]**2 - q[2]**2 - q[3]**2

    V = _dipole_vertex(q, d)
    chi_p1, chi_m1 = _helicity_spinors(k1)
    u_nu = _u_spinor(k1, 0.0, chi_m1)     # left-handed neutrino
    e_charge = np.sqrt(4 * np.pi * const.alphaQED)

    m2 = 0.0
    for sN in range(2):
        xiN = np.array([1.0, 0.0]) if sN == 0 else np.array([0.0, 1.0])
        uN = _u_spinor(k2, m_N, xiN)
        hnl_cur = [(_ubar(uN) @ V[mu] @ (_PL @ u_nu)) for mu in range(4)]
        for s1 in range(2):
            xi1 = np.array([1.0, 0.0]) if s1 == 0 else np.array([0.0, 1.0])
            ue1 = _u_spinor(p1, me, xi1)
            for s2 in range(2):
                xi2 = np.array([1.0, 0.0]) if s2 == 0 else np.array([0.0, 1.0])
                ue2 = _u_spinor(p2, me, xi2)
                e_cur = [(_ubar(ue2) @ _GAMMA[mu] @ ue1) for mu in range(4)]
                # contract with metric: J_HNL^mu g_{mu nu} J_e^nu / t
                amp = sum(_ETA[mu, mu] * hnl_cur[mu] * e_cur[mu] for mu in range(4))
                m2 += abs(e_charge * amp / tt) ** 2
    m2 /= 2.0   # average over initial electron spins (nu is fixed LH)
    return m2 / (16 * np.pi * (s - me**2) ** 2)


def check_electron_formula(d=1e-6):
    print("=" * 72)
    print("(D) Electron formula, independent spinor-level check")
    print("  D1: width anchor Gamma(N->nu gamma) vs d^2 m^3/4pi")
    for m in [0.1, 1.0]:
        gnum = numeric_width_Ngamma(m, d)
        gana = hnl.dipole_radiative_width(m, d)
        print(f"    m={m}: numeric/analytic = {gnum/gana:.6f}")
    print("  D2: dsigma/dt vs Eq. (A.10) at random phase-space points")
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(12):
        E = float(rng.uniform(50, 4000))
        m = float(rng.uniform(0.01, np.sqrt(2 * const.m_e * E) * 0.9))
        Te_lo, Te_hi = hnl.dipole_electron_Te_limits(E, m)
        Te = float(np.exp(rng.uniform(np.log(max(Te_lo * 1.01, 1e-6)), np.log(Te_hi * 0.99))))
        t = -2 * const.m_e * Te
        num = numeric_dsigma_dt(E, m, t, d)
        ana = -const.alphaQED * d**2 * (
            2 * const.m_e**2 * (4 * E**2 * t + m**4 - m**2 * t)
            + 4 * E * const.m_e * t * (t - m**2) + m**2 * t * (m**2 - t)
        ) / (2 * const.m_e**2 * E**2 * t**2)
        r = num / ana
        worst = max(worst, abs(r - 1))
        print(f"    E={E:7.1f} m={m:6.3f} Te={Te:9.3f}:  numeric/A.10 = {r:.6f}")
    print(f"    worst deviation: {worst:.2e}")
    print("  D3: m_N -> 0 limit vs Vogel-Engel magnetic moment (mu_nu = 2d):")
    E, Te = 1000.0, 30.0
    a10 = float(hnl.dipole_electron_dsigma_dTe(Te, E, 1e-4, d))
    ve = const.alphaQED * (2 * d) ** 2 * (1 / Te - 1 / E) * const.invGeV2_to_cm2
    print(f"    A.10(m_N=1e-4)/VogelEngel = {a10/ve:.6f}")




# ======================================================================
# (E) mixing portal: DarkNews Z-mediated DIS vs SM tables and crude threshold
# ======================================================================
def check_mixing_nc_dis():
    """Validates hnl_tools.darknews_nc_dis_xsec (the DarkNewsMixingPortal input):
    E1 light-N limit reproduces the SM NC-DIS tables (mint.xsecs, Alfonso data)
    to ~4%; E2 exact U^2 scaling; E3 the massive-N suppression vs the crude
    linear-threshold factor of the legacy MixingPortal (which overestimates the
    m_N ~ 10-40 GeV production by x2.4-13)."""
    from mint import xsecs
    print("=" * 72)
    print("(E) Mixing NC-DIS validation")
    E = np.array([250., 1000., 3000.])
    sig_dn = hnl.darknews_nc_dis_xsec(E, 0.05, target="O16", ref_U2=1.0) / 16.0
    sig_sm = xsecs.sigma_NC_nu(E)
    for e, a, b in zip(E, sig_dn, sig_sm):
        print(f"  E1 light-N: E={e:6.0f}  DN/SM-table = {a / b:.3f}")
    s1 = hnl.darknews_nc_dis_xsec(np.array([1000.]), 1.0, target="O16", ref_U2=1e-4)[0]
    s2 = hnl.darknews_nc_dis_xsec(np.array([1000.]), 1.0, target="O16", ref_U2=4e-4)[0]
    print(f"  E2 U^2 scaling: sigma(4e-4)/sigma(1e-4) = {s2 / s1:.4f} (expect 4)")
    s0 = hnl.darknews_nc_dis_xsec(np.array([1000.]), 0.05, target="O16", ref_U2=1.0)[0]
    for m in [2.0, 10.0, 20.0, 40.0]:
        sm_ = hnl.darknews_nc_dis_xsec(np.array([1000.]), m, target="O16", ref_U2=1.0)[0]
        crude = float(np.clip(1 - m**2 / (2 * 0.938 * 1000.0), 0, 1))
        print(f"  E3 m={m:5.1f}: exact suppression {sm_ / s0:.3f} | crude {crude:.3f}")

if __name__ == "__main__":
    check_convention()
    check_dis()
    check_coherent()
    check_electron_formula()
    check_mixing_nc_dis()
    print("=" * 72)
    print("done")
