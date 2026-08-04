"""Diagnostics for characterising a MINT detector against a beam.

These are measurement helpers, not part of the detector model: they take a
detector and a simulated beam and report how well the two match. They live
here, alongside the notebooks that use them, rather than in ``mint`` itself.

    import mint
    from detector_diagnostics import spot_quantiles, fit_sigma_div

    det = mint.detectors.benchmark
    ring, sim, ipy = mint.beams.standard_beam("numubar")

    r39, r86, r96 = spot_quantiles(det, sim)
    print(f"the aperture ({det.radius:.0f} cm) contains "
          f"{containment(det, sim, det.radius):.0%} of the beam")
"""

import numpy as np

M_MU = 0.1056583755  # GeV


def spot_quantiles(det, sim, sign=+1, E_min=0.0, r_max=2e3, dist=None,
                   quantiles=(0.393, 0.865, 0.956)):
    """Radii [cm] containing given fractions of the flux at the front face.

    The default quantiles are the 1, 2 and 2.5 sigma containments of a 2D
    Gaussian, ``1 - exp(-n^2/2)``, so the radii returned can be compared
    directly against ``n * det.sigma_spot``. Where they come out larger, the
    true angular distribution has a heavier tail than a Gaussian.

    ``r_max`` bounds the denominator, and the choice matters. Muons decay all
    the way around the ring, and rays from the arcs cross the face plane at
    enormous radius without ever having been part of the forward beam.
    Normalising to "every ray that crosses the plane" would understate the
    containment of the beam the aperture was actually built for.
    """
    E, w, rx, ry, sx, sy = det.face_rays(sim, sign=sign, E_min=E_min,
                                         r_sel=r_max, dist=dist)
    if w.sum() <= 0:
        return np.full(len(quantiles), np.nan)
    r = np.hypot(rx, ry)
    order = np.argsort(r)
    cdf = np.cumsum(w[order])
    cdf = cdf / cdf[-1]
    return np.interp(np.asarray(quantiles, float), cdf, r[order])


def containment(det, sim, radius, sign=+1, E_min=0.0, r_max=2e3, dist=None):
    """Fraction of the forward beam falling inside ``radius`` [cm] at the face.

    The denominator is the flux within ``r_max`` of the axis -- see
    :func:`spot_quantiles` for why that bound is needed.
    """
    E, w, rx, ry, sx, sy = det.face_rays(sim, sign=sign, E_min=E_min,
                                         r_sel=r_max, dist=dist)
    if w.sum() <= 0:
        return np.nan
    return float(w[np.hypot(rx, ry) < radius].sum() / w.sum())


def fit_sigma_div(det, sim, sign=+1, E_min=0.0, r_max=2e3, dist=None):
    """Beam divergence [rad] implied by the measured spot size.

    Inverts ``sigma_spot = L sqrt(sigma_div^2 + 1/gamma^2)`` using the radius
    containing 39.3% of the flux as the measured spot.

    This is a quantile estimator rather than a moment, deliberately: the
    neutrino angular distribution from muon decay has a far heavier tail than
    a Gaussian, so an RMS would be dominated by rays that never come near the
    detector. Returns NaN if the measured spot is narrower than the
    irreducible 1/gamma opening angle.
    """
    D = det.dist if dist is None else dist
    (r39,) = spot_quantiles(det, sim, sign=sign, E_min=E_min, r_max=r_max,
                            dist=dist, quantiles=(0.393,))
    val = (r39 / D) ** 2 - (M_MU / det.E_beam) ** 2
    return float(np.sqrt(val)) if val > 0 else np.nan
