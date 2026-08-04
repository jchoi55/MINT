"""LO DIS total neutrino cross sections from PDFs (default CT18NNLO).

Provides an in-house, PDF-consistent alternative to the tabulated cross
sections shipped in ``mint.xsecs`` ("Alfonso" tables): leading-order DIS on an
isoscalar nucleon with

  d^2 sigma^{nu}/dx dQ^2 = (G_F^2 / pi x) (M_V^2/(Q^2+M_V^2))^2
                           [ Q_flat(x,Q^2) + Q_bar(x,Q^2) (1-y)^2 ],

CKM-weighted final-flavor decomposition (light / charm / bottom), slow
rescaling xi = x (1 + m_q^2/Q^2) for heavy-quark production, a real NC
calculation (chiral couplings + Z propagator; the tabulated backend uses the
flat 0.335 x CC approximation), and the same PDF set, member and Q^2 cut as
the differential samplers used across the physics examples.

The (x, Q^2) integration uses a fixed log-log mesh on which the PDFs are
evaluated ONCE (parton's fast grid mode); each neutrino energy only reweights
the mesh, so building the full table takes seconds. Results are cached to
``~/.cache/mint/`` and loaded as interpolators afterwards.

Intended use via the backend switch:

    import mint
    mint.xsecs.use_backend("ct18")     # in-house LO+CT18NNLO tables
    mint.xsecs.use_backend("alfonso")  # shipped tables (default)

Accuracy note: this is a LO (alpha_s^0) calculation with NNLO-fit PDFs; it is
expected to agree with the dedicated NNLO tables at the few-percent to ~10%
level, which doubles as a cross-check of the DIS model used for the
event-by-event samplers.
"""

import os
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d

from mint import const

# PDG CKM magnitudes
V2 = {
    "ud": 0.97435**2, "us": 0.22500**2, "ub": 0.00369**2,
    "cd": 0.22100**2, "cs": 0.97500**2, "cb": 0.04080**2,
}
SW2 = 0.2312
M_N = const.m_avg          # isoscalar nucleon
M_W, M_Z = 80.379, 91.1876
M_C, M_B = 1.27, 4.5

# NC chiral couplings
_GLu, _GRu = 0.5 - 2.0 / 3.0 * SW2, -2.0 / 3.0 * SW2
_GLd, _GRd = -0.5 + 1.0 / 3.0 * SW2, 1.0 / 3.0 * SW2

_VERSION = "v1"


def _cache_path(pdf_name, member):
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "mint"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"lo_dis_{pdf_name}_m{member}_{_VERSION}.npz"


def _pdf_interpolators(pdf_name, member, x_grid, Q2_grid):
    """RegularGridInterpolators of x*f(x, Q^2) on (ln x, ln Q^2), per flavor."""
    from parton import mkPDF

    pdf = mkPDF(pdf_name, member)
    out = {}
    for pid in (1, 2, 3, 4, 5, -1, -2, -3, -4, -5):
        vals = np.asarray(pdf.xfxQ2(pid, x_grid, Q2_grid))  # (nx, nQ) mesh
        out[pid] = RegularGridInterpolator(
            (np.log(x_grid), np.log(Q2_grid)), np.clip(vals, 0.0, None),
            bounds_error=False, fill_value=0.0)
    return out


def compute_tables(pdf_name="CT18NNLO", member=0, E_grid=None,
                   Q2_min=2.0, W2_min=4.0, nx=220, nq=220, verbose=False):
    """Compute all LO DIS totals [cm^2/nucleon] on E_grid. Returns dict."""
    if E_grid is None:
        E_grid = np.geomspace(20.0, 1.0e7, 96)
    E_grid = np.asarray(E_grid, float)

    x_grid = np.geomspace(1e-6, 0.985, nx)
    Q2_grid = np.geomspace(Q2_min, 2.2 * M_N * E_grid[-1], nq)
    F = _pdf_interpolators(pdf_name, member, x_grid, Q2_grid)

    LX, LQ = np.meshgrid(np.log(x_grid), np.log(Q2_grid), indexing="ij")
    X, Q2 = np.exp(LX), np.exp(LQ)
    dlx = np.log(x_grid[1] / x_grid[0])
    dlq = np.log(Q2_grid[1] / Q2_grid[0])

    def xf(pid, xi=None):
        pts = np.stack([np.log(np.clip(xi if xi is not None else X, 1e-12, None)),
                        LQ], axis=-1)
        return F[pid](pts)

    # isoscalar combinations on the mesh (massless flavors at x, heavy at xi)
    Dq = 0.5 * (xf(1) + xf(2))
    Dqb = 0.5 * (xf(-1) + xf(-2))
    s, sb = xf(3), xf(-3)
    xi_c = np.clip(X * (1.0 + M_C**2 / Q2), None, 1.0)
    xi_b = np.clip(X * (1.0 + M_B**2 / Q2), None, 1.0)
    Dq_c, s_c = 0.5 * (xf(1, xi_c) + xf(2, xi_c)), xf(3, xi_c)
    Dqb_c, sb_c = 0.5 * (xf(-1, xi_c) + xf(-2, xi_c)), xf(-3, xi_c)
    c, cb = xf(4, xi_c), xf(-4, xi_c)
    Dqb_b, u_b = 0.5 * (xf(-1, xi_b) + xf(-2, xi_b)), 0.5 * (xf(1, xi_b) + xf(2, xi_b))
    c_b, cb_b = xf(4, xi_b), xf(-4, xi_b)

    propW = (M_W**2 / (Q2 + M_W**2)) ** 2
    propZ = (M_Z**2 / (Q2 + M_Z**2)) ** 2

    # CC flavor decomposition: (flat-in-y term, (1-y)^2 term) per channel
    CC = {
        "light_nu": (Dq * V2["ud"] + s * V2["us"],
                     Dqb * (V2["ud"] + V2["us"]) + cb * (V2["cd"] + V2["cs"])),
        "charm_nu": (Dq_c * V2["cd"] + s_c * V2["cs"], 0.0 * X),
        "bottom_nu": (0.0 * X, Dqb_b * V2["ub"] + cb_b * V2["cb"]),
        "light_nubar": (Dqb * V2["ud"] + sb * V2["us"],
                        Dq * (V2["ud"] + V2["us"]) + c * (V2["cd"] + V2["cs"])),
        "charm_nubar": (Dqb_c * V2["cd"] + sb_c * V2["cs"], 0.0 * X),
        "bottom_nubar": (0.0 * X, u_b * V2["ub"] + c_b * V2["cb"]),
    }
    # NC: nu has (gL^2 q + gR^2 qbar) flat and (gR^2 q + gL^2 qbar) (1-y)^2
    gu2, gub2 = _GLu**2 + 0 * X, _GRu**2 + 0 * X
    up_q, up_qb = Dq + c, Dqb + cb            # up-type: u (+charm at xi_c)
    dn_q, dn_qb = Dq + s, Dqb + sb            # down-type: d + s (b negligible)
    NC_nu_flat = (_GLu**2 * up_q + _GRu**2 * up_qb
                  + _GLd**2 * dn_q + _GRd**2 * dn_qb)
    NC_nu_y2 = (_GRu**2 * up_q + _GLu**2 * up_qb
                + _GRd**2 * dn_q + _GLd**2 * dn_qb)
    NC = {
        "NC_nu": (NC_nu_flat, NC_nu_y2),
        "NC_nubar": (NC_nu_y2, NC_nu_flat),
        "NC_charm_nu": (_GLu**2 * c + _GRu**2 * cb, _GRu**2 * c + _GLu**2 * cb),
        "NC_charm_nubar": (_GRu**2 * c + _GLu**2 * cb, _GLu**2 * c + _GRu**2 * cb),
    }

    pref = const.Gf**2 / np.pi * Q2 * dlx * dlq * const.invGeV2_to_cm2
    W2 = M_N**2 + Q2 * (1.0 / X - 1.0)
    base_ok = W2 >= W2_min

    tables = {k: np.zeros_like(E_grid) for k in list(CC) + list(NC)}
    for iE, E in enumerate(E_grid):
        y = Q2 / (2.0 * M_N * E * X)
        ok = base_ok & (y < 1.0)
        y2 = np.where(ok, (1.0 - y) ** 2, 0.0)
        for k, (flat, wy2) in CC.items():
            tables[k][iE] = np.sum(np.where(ok, pref * propW * (flat + wy2 * y2), 0.0))
        for k, (flat, wy2) in NC.items():
            tables[k][iE] = np.sum(np.where(ok, pref * propZ * (flat + wy2 * y2), 0.0))
        if verbose and iE % 20 == 0:
            print(f"  E = {E:.3g} GeV done")
    tables["E_grid"] = E_grid
    return tables


def load_tables(pdf_name="CT18NNLO", member=0, force=False):
    """Load (or compute + cache) the LO DIS tables; returns dict of interp1d
    matching the ``mint.xsecs`` naming: sigma_lightquark_CC_nu, ...,
    sigma_NC_nu, sigma_NC_nubar, sigma_NC_charm_nu(bar)."""
    path = _cache_path(pdf_name, member)
    if path.exists() and not force:
        data = dict(np.load(path))
    else:
        data = compute_tables(pdf_name, member)
        np.savez_compressed(path, **data)
    E = data["E_grid"]
    kw = {"kind": "linear", "fill_value": 0.0, "bounds_error": False}
    name_map = {
        "sigma_lightquark_CC_nu": "light_nu",
        "sigma_lightquark_CC_nubar": "light_nubar",
        "sigma_charm_CC_nu": "charm_nu",
        "sigma_charm_CC_nubar": "charm_nubar",
        "sigma_bottom_CC_nu": "bottom_nu",
        "sigma_bottom_CC_nubar": "bottom_nubar",
        "sigma_NC_nu": "NC_nu",
        "sigma_NC_nubar": "NC_nubar",
        "sigma_NC_charm_nu": "NC_charm_nu",
        "sigma_NC_charm_nubar": "NC_charm_nubar",
    }
    return {pub: interp1d(E, data[key], **kw) for pub, key in name_map.items()}
