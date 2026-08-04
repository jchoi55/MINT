"""Tertiary muon "beam" from primary-neutrino CC interactions in the rock.

The forward neutrino flux undergoes CC scattering in the rock shielding in
front of the detector. Hadronic showers and electrons are absorbed within
meters, but the CC muon carries (1 - y) E_nu on average and can penetrate
km-scale rock: the detector therefore sits in a tertiary muon beam.

Production: nubar_mu CC -> mu+ (mu+ beam, downstream) and nu_mu CC -> mu-
(mu- beam, upstream). Kinematics (y and the muon angle theta_munu wrt the
neutrino) are sampled from a LO DIS model, d sigma/dx dy ~
[xQ(x) + xQbar(x) (1-y)^2] (m_W^2/(Q^2+m_W^2))^2 (Q <-> Qbar for nubar),
with isoscalar CT18NNLO PDFs via `parton` (toy-shape fallback).
Only the (x, y) SHAPE is used -- the rate normalization comes from the
mint.xsecs CC tables (valid above 50 GeV; the sub-50 GeV flux is dropped).
The muon angle is theta = Q / (E_nu sqrt(1-y)).

Propagation: nupyprop (arXiv:2209.15581) muon-in-standard-rock tables, using
the same MUSIC-style stochastic scheme as its internal propagator: continuous
ionization + soft radiative losses (alpha, beta_cut) between catastrophic
brem / pair / photonuclear events with fractional loss y > 1e-3 sampled from
the ALLM integrated-cross-section CDFs; decay in flight included. On top of
the 1D energy loss we accumulate Highland multiple scattering per step
(correlated angle + lateral displacement, PDG prescription) to track where
and at what angle the muon crosses the detector face.

Units: GeV, cm, rad. Muons are followed down to E = 1 GeV (nupyprop table
floor); slower muons range out within meters and are counted as lost.
"""

import numpy as np

import nupyprop.data as _npp_data
import nupyprop.propagate as _npp_prop

from mint import const, xsecs

_T = _npp_prop.transport

M_MU = 0.1056583755          # GeV
CTAU_MU = 6.58638e4          # cm
C_CM_PER_NS = const.c_LIGHT * 1e-9
RHO_ROCK = 2.65              # g/cm^3 (nupyprop standard rock)
X0_ROCK = 26.54              # g/cm^2 radiation length of standard rock
E_MIN = 1.0                  # GeV: nupyprop table floor = tracking threshold

PN_MODEL = "allm"

_xc = np.asfortranarray(_npp_data.get_xc("muon", PN_MODEL, "rock"))
_ixc = np.asfortranarray(_npp_data.get_ixc("muon", PN_MODEL, "rock"))
_alpha = np.asfortranarray(_npp_data.get_alpha("muon", "rock"))
_beta_cut = np.asfortranarray(_npp_data.get_beta("muon", PN_MODEL, "rock", "cut"))
_beta_tot = np.asfortranarray(_npp_data.get_beta("muon", PN_MODEL, "rock", "total"))
_E_lep = np.asarray(_npp_data.E_lep, float)


def _ab(E, beta=_beta_cut):
    """Interpolated ionization alpha [GeV cm^2/g] and radiative beta [cm^2/g]."""
    a = np.interp(E, _E_lep, _alpha)
    b = (np.interp(E, _E_lep, beta[:, 0]) + np.interp(E, _E_lep, beta[:, 1])
         + np.interp(E, _E_lep, beta[:, 2]))
    return a, b


def _range_table():
    E = np.asarray(_E_lep, float)
    a, b = _ab(E, beta=_beta_tot)
    dE = np.diff(E)
    integrand = 1.0 / (a + b * E)
    X = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dE)])
    return E, X


_RANGE_E, _RANGE_X = _range_table()


def continuous_range(E0):
    """Continuous-slowing-down range [g/cm^2] from E0 down to E_MIN."""
    E0 = np.asarray(E0, float)
    return np.interp(np.clip(E0, E_MIN, _RANGE_E[-1]), _RANGE_E, _RANGE_X)


def continuous_E_profile(E0, X_grid, dX=100.0):
    """Deterministic continuous-loss energy profile E(X) [GeV] on the column
    depth grid X_grid [g/cm^2], integrating dE/dX = -(alpha + beta_total E).
    Entries where the muon has ranged out are set to E_MIN."""
    X_grid = np.asarray(X_grid, float)
    out = np.empty_like(X_grid)
    E, x = float(E0), 0.0
    for i, xg in enumerate(X_grid):
        while x < xg and E > E_MIN:
            h = min(dX, xg - x)
            a, b = _ab(E, beta=_beta_tot)
            E = max(E - (a + b * E) * h, 0.0)
            x += h
        out[i] = max(E, E_MIN)
    return out


# ============================================================================
# CC kinematics: LO DIS (x, y) sampler
# ============================================================================
M_N = const.m_avg             # GeV, isoscalar nucleon
M_W = 80.38

_NX, _NY = 240, 120
_lx_edges = np.linspace(-4.0, 0.0, _NX + 1)
_y_edges = np.linspace(1e-3, 0.999, _NY + 1)
_lx_c = 0.5 * (_lx_edges[:-1] + _lx_edges[1:])
_y_c = 0.5 * (_y_edges[:-1] + _y_edges[1:])


def _xq(x):
    return 2.2 * x**0.55 * (1.0 - x) ** 2.9 + _xqbar(x)


def _xqbar(x):
    return 0.6 * (1.0 - x) ** 7


def _pdf_grids():
    """CC parton groupings on the (log10 x, y) grid: CT18NNLO via `parton`,
    evaluated at Q^2(x, y) for a reference E_nu = 1.5 TeV (scaling violations
    across the 0.3-5 TeV band move these moments at the percent level).
    Isoscalar LO CC groupings are returned as
    (nu_flat, nu_y2, nubar_flat, nubar_y2), where the y2 term is multiplied by
    (1-y)^2. Falls back to toy Q/Qbar shapes if `parton` is unavailable."""
    lx, y = np.meshgrid(_lx_c, _y_c, indexing="ij")
    x = 10.0**lx
    try:
        from parton import mkPDF
        pdf = mkPDF("CT18NNLO", 0)
        E_REF = 1500.0
        Q2 = np.clip(2.0 * M_N * E_REF * x * y, 2.0, None)
        nu_flat = np.empty_like(x)
        nu_y2 = np.empty_like(x)
        nubar_flat = np.empty_like(x)
        nubar_y2 = np.empty_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                xi, q2 = float(x[i, j]), float(Q2[i, j])
                Dq = 0.5 * (pdf.xfxQ2(2, xi, q2) + pdf.xfxQ2(1, xi, q2))
                Dqb = 0.5 * (pdf.xfxQ2(-2, xi, q2) + pdf.xfxQ2(-1, xi, q2))
                s, sb = pdf.xfxQ2(3, xi, q2), pdf.xfxQ2(-3, xi, q2)
                c, cb = pdf.xfxQ2(4, xi, q2), pdf.xfxQ2(-4, xi, q2)
                nu_flat[i, j] = Dq + s
                nu_y2[i, j] = Dqb + cb
                nubar_flat[i, j] = Dqb + sb
                nubar_y2[i, j] = Dq + c
    except Exception:
        xq, xqb = _xq(x), _xqbar(x)
        nu_flat, nu_y2, nubar_flat, nubar_y2 = xq, xqb, xqb, xq
    return x, y, nu_flat, nu_y2, nubar_flat, nubar_y2


def _build_samplers():
    x, y, nu_flat, nu_y2, nubar_flat, nubar_y2 = _pdf_grids()
    out = {}
    for nubar in (False, True):
        flat, y2 = (nubar_flat, nubar_y2) if nubar else (nu_flat, nu_y2)
        w = flat + y2 * (1.0 - y) ** 2
        w = np.clip(w, 0.0, None) * x * np.log(10.0)   # d(log10 x) jacobian
        out[nubar] = (w / w.sum()).ravel()
    return out


_P_XY = _build_samplers()


def sample_cc_xy(E_nu, nuflavor, rng):
    """Sample (x, y) of a CC DIS event for each neutrino energy in E_nu.

    Base sample from the energy-independent LO grid, thinned by the
    W-propagator factor 1/(1 + Q^2/m_W^2)^2 via rejection.
    """
    E_nu = np.asarray(E_nu, float)
    nubar = "bar" in nuflavor
    p = _P_XY[nubar]
    n = E_nu.size
    x_out = np.empty(n)
    y_out = np.empty(n)
    todo = np.arange(n)
    while todo.size:
        idx = rng.choice(p.size, size=todo.size, p=p)
        ix, iy = np.unravel_index(idx, (_NX, _NY))
        lx = _lx_c[ix] + (rng.uniform(size=todo.size) - 0.5) * (_lx_edges[1] - _lx_edges[0])
        y = _y_c[iy] + (rng.uniform(size=todo.size) - 0.5) * (_y_edges[1] - _y_edges[0])
        x = 10.0**lx
        Q2 = 2.0 * M_N * E_nu[todo] * x * y
        acc = rng.uniform(size=todo.size) < 1.0 / (1.0 + Q2 / M_W**2) ** 2
        x_out[todo[acc]] = x[acc]
        y_out[todo[acc]] = y[acc]
        todo = todo[~acc]
    return x_out, y_out


def muon_kinematics(E_nu, nuflavor, rng):
    """CC muon energy and angle wrt the neutrino: (E_mu, theta_munu [rad])."""
    x, y = sample_cc_xy(E_nu, nuflavor, rng)
    E_nu = np.asarray(E_nu, float)
    E_mu = (1.0 - y) * E_nu
    Q = np.sqrt(2.0 * M_N * E_nu * x * y)
    theta = Q / (E_nu * np.sqrt(np.clip(1.0 - y, 1e-12, None)))
    return E_mu, np.clip(theta, 0.0, np.pi / 2)


def secondary_muon_delay_ns(E_prod, E_final, d_face_cm, theta_face):
    """Extra arrival delay [ns] of a secondary muon after production.

    The delay is added to the parent-neutrino face time. The mass term uses the
    usual high-energy expansion integrated with the start/end energy product,
    and the angular term assumes the face-plane angle was accumulated roughly
    linearly through the rock, giving int theta^2 dz / 2c ~= theta_face^2 L/4c.
    """
    E_prod = np.maximum(np.asarray(E_prod, float), M_MU)
    E_final = np.maximum(np.asarray(E_final, float), M_MU)
    d_face_cm = np.asarray(d_face_cm, float)
    theta_face = np.asarray(theta_face, float)
    mass_delay = M_MU**2 * d_face_cm / (2.0 * E_prod * E_final * C_CM_PER_NS)
    angle_delay = 0.25 * theta_face**2 * d_face_cm / C_CM_PER_NS
    return mass_delay + angle_delay


# total CC cross section per nucleon: single implementation in mint.xsecs
sigma_cc = xsecs.sigma_CC


# ============================================================================
# Inverse muon decay on the electrons of the rock
# ============================================================================
M_E = 5.109989461e-4


def sigma_imd(E_nu, channel):
    """IMD cross section per electron [cm^2]: channel = "numu" for
    nu_mu e- -> mu- nu_e (t-channel, flat in y) or "nuebar" for
    nubar_e e- -> mu- nubar_mu (s-channel annihilation, (1-y)^2)."""
    return xsecs.inverse_lepton_decay_sigma(np.asarray(E_nu, float), channel, "m")


def imd_kinematics(E_nu, channel, rng):
    """Muon energy and angle wrt the neutrino for IMD, exact two-body CM
    kinematics. y = 1 - E_mu/E_nu is flat ("numu") or (1-y)^2 ("nuebar")
    on [0, 1 - m_mu^2/s]; the muon angle follows from cos(theta*)."""
    E = np.asarray(E_nu, float)
    s = M_E**2 + 2.0 * M_E * E
    ymax = np.clip(1.0 - M_MU**2 / s, 0.0, None)
    u = rng.uniform(size=E.size)
    if channel == "numu":
        y = u * ymax
    else:
        cdf_max = 1.0 - (1.0 - ymax) ** 3
        y = 1.0 - (1.0 - u * cdf_max) ** (1.0 / 3.0)
    E_mu = (1.0 - y) * E

    sqs = np.sqrt(s)
    E_star = (s + M_MU**2) / (2.0 * sqs)
    p_star = (s - M_MU**2) / (2.0 * sqs)
    gam = (E + M_E) / sqs
    beta = np.sqrt(np.clip(1.0 - 1.0 / gam**2, 0.0, None))
    ct = np.clip((E_mu / gam - E_star) / np.maximum(beta * p_star, 1e-300), -1.0, 1.0)
    pT = p_star * np.sqrt(np.clip(1.0 - ct**2, 0.0, None))
    pL = gam * (p_star * ct + beta * E_star)
    theta = np.arctan2(pT, np.maximum(pL, 1e-300))
    return E_mu, theta


# ============================================================================
# Muon propagation through a rock slab (nupyprop primitives + Highland MCS)
# ============================================================================
M_RHO2 = 0.60                # GeV^2: VMD scale for photonuclear q_T sampling


def _stochastic_kick(it, y, E_after, rng):
    """Muon deflection angle [rad] from a single stochastic interaction.

    Kinematic estimates in the spirit of Van Ginneken / PROPOSAL: for
    bremsstrahlung the muon recoils against a photon of typical transverse
    momentum ~ y m_mu; pair production transfers about half that; for
    photonuclear scattering q_T is drawn from a vector-meson-dominance
    propagator ~ 1/(Q^2 + m_rho^2)^2 (log-enhanced tail, single scatters can
    reach mrad-scale even at TeV energies -- cf. Gutjahr et al. 2208.11902).
    """
    if it == 3:                                   # bremsstrahlung
        qT = y * M_MU
    elif it == 4:                                 # pair production
        qT = 0.5 * y * M_MU
    else:                                         # photonuclear
        u = rng.uniform()
        Q2 = M_RHO2 * u / max(1.0 - u, 1e-6)
        # crude kinematic cap for the DEFLECTION estimate only (not a cross
        # section): the max() floor deliberately keeps the VMD scale for very
        # soft kicks even where 2 m_N y E dips below m_rho^2
        Q2 = min(Q2, max(2.0 * 0.94 * y * E_after, M_RHO2))
        qT = np.sqrt(Q2)
    return qT / max(E_after, M_MU)


def propagate_muon(E0, X_total, rng, rho=RHO_ROCK, record=False,
                   stochastic_deflection=True):
    """One muon through X_total [g/cm^2] of rock.

    Returns (E_fin, thx, thy, dispx, dispy): final energy (0 if ranged out or
    decayed) and the accumulated multiple-scattering deflection angles [rad]
    and lateral displacements [cm] in two transverse planes. With
    record=True, also returns a trajectory list of (X [g/cm^2], E [GeV],
    theta [rad]) checkpoints after every continuous segment and stochastic
    interaction.
    """
    E, x0 = float(E0), 0.0
    thx = thy = dx = dy = 0.0
    traj = [(0.0, E, 0.0)] if record else None

    def _ret(E_out):
        if record:
            return E_out, thx, thy, dx, dy, traj
        return E_out, thx, thy, dx, dy

    if X_total <= 0.0:
        return _ret(E)
    # Cap on a single continuous segment. At low E the stochastic interaction
    # length blows up (~1e5 g/cm^2 at a few GeV) and an uncapped segment would
    # smear the stopping point by hundreds of meters and hold alpha, beta
    # fixed over a huge energy change. Truncating the exponential step is
    # exact (memoryless): if the sampled step exceeds the cap, advance by the
    # cap with no interaction and resample.
    STEP_CAP = 5e3    # g/cm^2 (~19 m rock)
    while True:
        lam = _T.int_depth_lep(E, _xc, rho, M_MU, CTAU_MU)
        step = -lam * np.log(rng.uniform())
        interact = step <= STEP_CAP
        adv = step if interact else STEP_CAP
        last = x0 + adv >= X_total
        seg = X_total - x0 if last else adv

        a, b = _ab(E)
        e_mid = max(E - (E * b + a) * seg, E_MIN)      # MUSIC average-energy trick
        a, b = _ab(10.0 ** (0.5 * (np.log10(E) + np.log10(e_mid))))
        E_new = _T.em_cont_part(E, a, b, seg, M_MU)

        # Highland MCS over this segment (PDG correlated angle + offset pair)
        e_scat = np.sqrt(E * max(E_new, E_MIN))
        t = seg / X0_ROCK
        s_th = 0.0136 / e_scat * np.sqrt(t) * (1.0 + 0.038 * np.log(max(t, 1e-3)))
        z_cm = seg / rho
        g1, g2, g3, g4 = rng.standard_normal(4)
        dx += z_cm * (thx + s_th * (g1 / np.sqrt(12.0) + g2 / 2.0))
        thx += s_th * g2
        dy += z_cm * (thy + s_th * (g3 / np.sqrt(12.0) + g4 / 2.0))
        thy += s_th * g4

        if record:
            traj.append((x0 + seg, E_new, np.hypot(thx, thy)))
        if E_new <= E_MIN:
            return _ret(0.0)
        if last:
            return _ret(E_new)
        E, x0 = E_new, x0 + adv
        if not interact:
            continue

        it = _T.interaction_type_lep(E, _xc, rho, M_MU, CTAU_MU)
        # it == 6 is nupyprop's defensive fallthrough (CC/NC disabled): no-op
        if it == 2:                                    # decayed in flight
            return _ret(0.0)
        if it in (3, 4, 5):                            # brem / pair / photonuclear
            y_st = _T.find_y(E, _ixc, it)
            E *= 1.0 - y_st
            if stochastic_deflection:
                th_k = _stochastic_kick(it, y_st, E, rng)
                phi_k = rng.uniform(0.0, 2.0 * np.pi)
                thx += th_k * np.cos(phi_k)
                thy += th_k * np.sin(phi_k)
            if record:
                traj.append((x0, E, np.hypot(thx, thy)))
            if E <= E_MIN:
                return _ret(0.0)


# ============================================================================
# Full chain: face-crossing neutrino rays -> muons at the detector face
# ============================================================================
def muon_beam_mc(E_nu, w_nu, rx, ry, sx, sy, nuflavor, rng,
                 l_prop_cm=2.5e5, gap_cm=1000.0, det_r=200.0,
                 n_mc=50_000, rho=RHO_ROCK, n_nucleon=None, channel="dis",
                 stochastic_deflection=True, n_electron=None):
    """MC of the rock-induced muon beam for one primary flavor.

    Inputs are per-neutrino-ray arrays at the detector face plane: energy
    [GeV], weight [nu/yr], transverse crossing position (rx, ry) [cm] and
    slopes (sx, sy) = (px/pz, py/pz) wrt the propagation axis. Rays are
    resampled down to n_mc equal-weight events; each gets one interaction
    vertex, uniform in the last l_prop_cm of rock (muons from deeper cannot
    reach: l_prop_cm exceeds the range at the beam energy), carrying weight
    w = W_tot/n_mc * n_target * sigma(E) * l_prop_cm.

    channel = "dis" (CC on nucleons) or "imd" (inverse muon decay on the
    rock electrons; nuflavor must then be "numu" or "nuebar").

    Returns a dict with production-level arrays (all sampled muons):
    E_mu, theta_munu, theta_mu (wrt z), theta_nu (parent wrt z), w, d_face,
    E_nu and parent_index (index into the input ray arrays). Plane-survivor
    arrays (E > 1 GeV at the face plane, any radius) carry the ``srv_`` prefix:
    srv_x, srv_y [cm], srv_E, srv_theta (wrt z, incl. MCS), srv_w,
    srv_dface, srv_E_prod, srv_E_nu, srv_theta_prod, srv_theta_nu,
    srv_parent_index, and srv_r_parent.
    """
    E_nu, w_nu = np.asarray(E_nu, float), np.asarray(w_nu, float)
    rx, ry = np.asarray(rx, float), np.asarray(ry, float)
    sx, sy = np.asarray(sx, float), np.asarray(sy, float)
    if n_nucleon is None:
        from mint import detector_tools as dt
        n_nucleon = dt.standard_rock.N
    if n_electron is None:
        from mint import detector_tools as dt
        n_electron = dt.standard_rock.e
    n_mc = int(n_mc)

    if channel == "imd" and nuflavor not in ("numu", "nuebar"):
        raise ValueError("IMD muon production is available only for 'numu' and 'nuebar'")

    empty = np.zeros(0)
    if n_mc <= 0 or E_nu.size == 0 or w_nu.sum() <= 0.0:
        return {"E_mu": empty, "theta_munu": empty, "theta_mu": empty,
                "theta_nu": empty, "w": empty, "E_nu": empty,
                "d_face": empty, "parent_index": empty.astype(int),
                "srv_x": empty, "srv_y": empty, "srv_E": empty,
                "srv_theta": empty, "srv_w": empty, "srv_dface": empty,
                "srv_E_prod": empty, "srv_E_nu": empty,
                "srv_theta_prod": empty, "srv_theta_nu": empty,
                "srv_parent_index": empty.astype(int), "srv_r_parent": empty}

    # weighted resampling to n_mc equal-weight rays
    W_tot = w_nu.sum()
    idx = rng.choice(E_nu.size, size=n_mc, p=w_nu / W_tot)
    E, rx, ry, sx, sy = E_nu[idx], rx[idx], ry[idx], sx[idx], sy[idx]

    # one vertex per ray in the last l_prop_cm of rock (before the gap)
    s_rock = rng.uniform(0.0, l_prop_cm, size=n_mc)     # depth from rock exit
    d_face = gap_cm + s_rock                            # distance to face plane

    if channel == "dis":
        w = (W_tot / n_mc) * n_nucleon * sigma_cc(E, nuflavor) * l_prop_cm
        E_mu, th_munu = muon_kinematics(E, nuflavor, rng)
    elif channel == "imd":
        w = (W_tot / n_mc) * n_electron * sigma_imd(E, nuflavor) * l_prop_cm
        E_mu, th_munu = imd_kinematics(E, nuflavor, rng)
    else:
        raise ValueError(f"unknown channel {channel!r}")
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n_mc)
    mux = sx + th_munu * np.cos(phi)
    muy = sy + th_munu * np.sin(phi)
    theta_nu = np.sqrt(sx**2 + sy**2)
    theta_mu = np.sqrt(mux**2 + muy**2)

    out = {"E_mu": E_mu, "theta_munu": th_munu, "theta_mu": theta_mu,
           "theta_nu": theta_nu, "w": w, "E_nu": E, "d_face": d_face,
           "parent_index": idx}

    # propagate the plausible ones (skip hopeless: > 1.5 x CSDA range, or w = 0)
    X_tot = s_rock * rho
    live = (E_mu > E_MIN) & (w > 0) & (X_tot < 1.5 * continuous_range(E_mu))
    E_fin = np.zeros(n_mc)
    thx = np.zeros(n_mc)
    thy = np.zeros(n_mc)
    dxs = np.zeros(n_mc)
    dys = np.zeros(n_mc)
    for i in np.nonzero(live)[0]:
        E_fin[i], thx[i], thy[i], dxs[i], dys[i] = propagate_muon(
            E_mu[i], X_tot[i], rng, rho, stochastic_deflection=stochastic_deflection)

    # arrival position on the face plane and angle wrt z: initial slope over
    # the full path + MCS displacement in the rock + drift through the air gap
    # with the MCS-deflected exit angle
    px = rx - d_face * sx                               # production point
    py = ry - d_face * sy
    ax = px + mux * d_face + dxs + thx * gap_cm
    ay = py + muy * d_face + dys + thy * gap_cm
    th_arr = np.sqrt((mux + thx) ** 2 + (muy + thy) ** 2)

    srv = E_fin > E_MIN
    out.update(srv_x=ax[srv], srv_y=ay[srv], srv_E=E_fin[srv],
               srv_theta=th_arr[srv], srv_w=w[srv], srv_dface=d_face[srv],
               srv_E_prod=E_mu[srv], srv_E_nu=E[srv],
               srv_theta_prod=theta_mu[srv], srv_theta_nu=theta_nu[srv],
               srv_parent_index=idx[srv],
               srv_r_parent=np.sqrt(rx[srv] ** 2 + ry[srv] ** 2))
    return out


# ============================================================================
# Profile-aware version: production anywhere between the IP and the face
# ----------------------------------------------------------------------------
# The rock is no longer the only target. A detector at 250 m sits BEFORE the
# rock, and everything upstream of it is the tungsten shielding wrapped around
# the machine beam pipe (mint.beamline) plus tunnel air. This version takes the
# piecewise-constant material profile along the line of sight and samples the
# production vertex anywhere in it, then transports the muon through whatever
# materials remain between the vertex and the face.
# ============================================================================
def propagate_through_slabs(E0, X_slabs, rho_slabs, len_slabs, rng,
                            stochastic_deflection=True, X_skip=1.0):
    """One muon through a sequence of (grammage, density, length) slabs.

    Slabs with less than ``X_skip`` g/cm^2 (air) are treated as pure drift: no
    energy loss, no scattering, but the accumulated angle still displaces the
    muon. Returns (E_fin, thx, thy, dx, dy) as :func:`propagate_muon`.
    """
    E = float(E0)
    thx = thy = dx = dy = 0.0
    for X, rho, L in zip(X_slabs, rho_slabs, len_slabs):
        if L <= 0.0:
            continue
        if X < X_skip:                       # drift
            dx += thx * L
            dy += thy * L
            continue
        E, tx, ty, sx, sy = propagate_muon(
            E, X, rng, rho=rho, stochastic_deflection=stochastic_deflection)
        # displacement inside the slab, plus the angle carried in from upstream
        dx += thx * L + sx
        dy += thy * L + sy
        thx += tx
        thy += ty
        if E <= E_MIN:
            return 0.0, thx, thy, dx, dy
    return E, thx, thy, dx, dy


def muon_beam_profile_mc(E_nu, w_nu, rx, ry, sx, sy, nuflavor, rng, profile,
                         dist, det_r=200.0, n_mc=50_000, channel="dis",
                         stochastic_deflection=True, X_window=None):
    """Secondary muons at a face at ``dist``, for an arbitrary material profile.

    ``profile`` is a dict with 1-D arrays ``z0, z1, n_nuc, n_e, rho`` (e.g.
    ``mint.beamline.Beamline.flux_weighted_profile``). Production vertices are
    sampled uniformly in TARGET COLUMN (equivalently, in grammage) over the
    last ``X_window`` g/cm^2 before the face -- muons produced deeper than
    their range cannot arrive, so restricting the window is pure variance
    reduction and is corrected for exactly in the weight.

    Returns the same dict as :func:`muon_beam_mc`.
    """
    E_nu, w_nu = np.asarray(E_nu, float), np.asarray(w_nu, float)
    rx, ry = np.asarray(rx, float), np.asarray(ry, float)
    sx, sy = np.asarray(sx, float), np.asarray(sy, float)

    z0 = np.asarray(profile["z0"], float)
    z1 = np.minimum(np.asarray(profile["z1"], float), dist)
    L = np.clip(z1 - z0, 0.0, None)
    n_t = np.asarray(profile["n_e" if channel == "imd" else "n_nuc"], float)
    rho = np.asarray(profile["rho"], float)

    W_tot = w_nu.sum()
    n_mc = int(n_mc)
    empty = np.zeros(0)
    if channel == "imd" and nuflavor not in ("numu", "nuebar"):
        raise ValueError("IMD muon production is available only for 'numu' and 'nuebar'")
    if n_mc <= 0 or E_nu.size == 0 or W_tot <= 0.0:
        return {"E_mu": empty, "theta_munu": empty, "theta_mu": empty,
                "theta_nu": empty, "w": empty, "E_nu": empty,
                "d_face": empty, "z_prod": empty, "parent_index": empty.astype(int),
                "srv_x": empty, "srv_y": empty, "srv_E": empty,
                "srv_theta": empty, "srv_w": empty, "srv_dface": empty,
                "srv_E_prod": empty, "srv_E_nu": empty,
                "srv_theta_prod": empty, "srv_theta_nu": empty,
                "srv_parent_index": empty.astype(int), "srv_z": empty,
                "srv_r_parent": empty}
    idx = rng.choice(E_nu.size, size=n_mc, p=w_nu / W_tot)
    E, rx, ry, sx, sy = E_nu[idx], rx[idx], ry[idx], sx[idx], sy[idx]

    if channel == "dis":
        E_mu, th_munu = muon_kinematics(E, nuflavor, rng)
        sig = sigma_cc(E, nuflavor)
    elif channel == "imd":
        E_mu, th_munu = imd_kinematics(E, nuflavor, rng)
        sig = sigma_imd(E, nuflavor)
    else:
        raise ValueError(f"unknown channel {channel!r}")

    # Restrict production to the last X_window g/cm^2 before the face: a muon
    # made deeper than its own range cannot arrive, so this is pure variance
    # reduction, exactly corrected for by using the column of the window in the
    # weight. The cut is applied INSIDE the straddling slab -- cutting on whole
    # slabs would throw away the entire rock as soon as the window landed in it.
    if X_window is None:                        # 1.5 x the CSDA range is ample
        X_window = 1.5 * float(continuous_range(np.max(E_mu) if E_mu.size else 1e3))
    # grammage from each slab boundary to the face (decreasing; 0 at the face)
    gram_edge = np.concatenate([(rho * L)[::-1].cumsum()[::-1], [0.0]])
    j = int(np.searchsorted(-gram_edge, -X_window))   # first edge inside window
    if j <= 0:
        z_win = z0[0] if z0.size else 0.0
    else:
        i = j - 1                                     # slab straddling the cut
        z_win = (z1[i] - (X_window - gram_edge[j]) / rho[i] if rho[i] > 0
                 else z0[i])
        z_win = float(np.clip(z_win, z0[i], z1[i]))
    # effective slab lengths inside the window
    L = np.clip(z1 - np.maximum(z0, z_win), 0.0, None)
    col = (n_t * L)[::-1].cumsum()[::-1]        # col[i] = column from z0[i] to face
    col_win = col[0] if col.size else 0.0

    if col_win <= 0 or W_tot <= 0:
        empty = np.zeros(0)
        return {"E_mu": E_mu, "theta_munu": th_munu,
                "theta_mu": np.hypot(sx, sy), "theta_nu": np.hypot(sx, sy),
                "w": np.zeros(n_mc), "E_nu": E, "d_face": np.zeros(n_mc),
                "parent_index": idx,
                "srv_x": empty, "srv_y": empty, "srv_E": empty,
                "srv_theta": empty, "srv_w": empty, "srv_dface": empty,
                "srv_E_prod": empty, "srv_E_nu": empty,
                "srv_theta_prod": empty, "srv_theta_nu": empty,
                "srv_parent_index": empty.astype(int), "srv_r_parent": empty}

    # sample the vertex uniformly in target column, then invert to z
    u = rng.uniform(0.0, col_win, size=n_mc)
    edges = np.concatenate([col, [0.0]])                  # decreasing
    k = np.clip(np.searchsorted(-edges, -u) - 1, 0, L.size - 1)
    frac = np.where((n_t[k] * L[k]) > 0,
                    (col[k] - u) / np.maximum(n_t[k] * L[k], 1e-300), 0.0)
    z_v = np.maximum(z0[k], z_win) + np.clip(frac, 0.0, 1.0) * L[k]
    w = (W_tot / n_mc) * sig * col_win

    phi = rng.uniform(0.0, 2.0 * np.pi, size=n_mc)
    mux = sx + th_munu * np.cos(phi)
    muy = sy + th_munu * np.sin(phi)
    d_face = dist - z_v

    out = {"E_mu": E_mu, "theta_munu": th_munu,
           "theta_mu": np.hypot(mux, muy), "theta_nu": np.hypot(sx, sy),
           "w": w, "E_nu": E, "d_face": d_face, "z_prod": z_v,
           "parent_index": idx}

    # downstream material for every event, slab by slab
    seg = np.clip(np.minimum(z1[None, :], dist) - np.maximum(z0[None, :], z_v[:, None]),
                  0.0, None)
    X_seg = seg * rho[None, :]
    X_tot = X_seg.sum(axis=1)
    live = (E_mu > E_MIN) & (w > 0) & (X_tot < 1.5 * continuous_range(E_mu))
    E_fin = np.zeros(n_mc); thx = np.zeros(n_mc); thy = np.zeros(n_mc)
    dxs = np.zeros(n_mc); dys = np.zeros(n_mc)
    for i in np.nonzero(live)[0]:
        E_fin[i], thx[i], thy[i], dxs[i], dys[i] = propagate_through_slabs(
            E_mu[i], X_seg[i], rho, seg[i], rng,
            stochastic_deflection=stochastic_deflection)

    px = rx - d_face * sx                       # parent ray at the vertex
    py = ry - d_face * sy
    ax = px + mux * d_face + dxs
    ay = py + muy * d_face + dys
    th_arr = np.hypot(mux + thx, muy + thy)

    srv = E_fin > E_MIN
    out.update(srv_x=ax[srv], srv_y=ay[srv], srv_E=E_fin[srv],
               srv_theta=th_arr[srv], srv_w=w[srv], srv_dface=d_face[srv],
               srv_E_prod=E_mu[srv], srv_E_nu=E[srv],
               srv_theta_prod=np.hypot(mux[srv], muy[srv]),
               srv_theta_nu=np.hypot(sx[srv], sy[srv]),
               srv_parent_index=idx[srv], srv_z=z_v[srv],
               srv_r_parent=np.hypot(rx[srv], ry[srv]))
    return out
