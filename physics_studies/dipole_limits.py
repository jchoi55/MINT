"""Existing constraints on the muon-flavor dipole portal, d_{mu N} [GeV^-1].

Digitized limit files copied from the nu-dipole repository
(``digitized/ExistingConstraints`` and the MiniBooNE preferred regions from
``ContourTextFiles/dtau_0``). The drawing logic follows
``nu-dipole/xsecs/plot_tools.initiate_main_plot`` (dtau = 0 case), with the
MINERvA limits omitted.

Axes convention: x = m_N [MeV], y = d_{mu N} [GeV^-1], both log scale.
"""

import os
import colorsys

from matplotlib.pyplot import annotate
import numpy as np
import matplotlib.colors as mc
from scipy.interpolate import splprep, splev

from mint import const

DATA_DIR = os.path.join(os.path.dirname(__file__), "dipole_limits_data")


def lighten_color(color, amount=0.5):
    """Lighten a matplotlib color (string, hex, or RGB tuple)."""
    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


LIGHT_GREY = lighten_color("lightgrey", 0.5)


def plot_closed_region(points, logx=False, logy=False):
    """Order a scatter of boundary points into a closed region (angle sort +
    linear spline), optionally in log space. From nu-dipole/xsecs/plot_tools."""
    x, y = points
    if logy:
        if (y == 0).any():
            raise ValueError("y values cannot contain any zeros in log mode.")
        sy = np.sign(y)
        ssy = (np.abs(y) < 1) * (-1) + (np.abs(y) > 1) * (1)
        y = ssy * np.log(y * sy)
    if logx:
        if (x == 0).any():
            raise ValueError("x values cannot contain any zeros in log mode.")
        sx = np.sign(x)
        ssx = (x < 1) * (-1) + (x > 1) * (1)
        x = ssx * np.log(x * sx)

    points = np.array([x, y]).T
    points_s = points - points.mean(0)
    angles = np.angle(points_s[:, 0] + 1j * points_s[:, 1])
    points_sort = points_s[angles.argsort()] + points.mean(0)

    tck, u = splprep(points_sort.T, u=None, s=0.0, per=0, k=1)
    u_new = np.linspace(u.min(), u.max(), len(points[:, 0]))
    x_new, y_new = splev(u_new, tck, der=0)

    if logx:
        x_new = sx * np.exp(ssx * x_new)
    if logy:
        y_new = sy * np.exp(ssy * y_new)
    return x_new, y_new


def plot_existing_limits(
    ax,
    data_dir=DATA_DIR,
    x_scale=1.0,
    limit_color="dimgrey",
    facecolor=LIGHT_GREY,
    annotate=True,
    miniboone=True,
    annotation_fontsize=8,
):
    """Draw the existing d_{mu N} constraints on ``ax`` (m_N vs d [GeV^-1]).

    ``x_scale`` multiplies every mass before plotting: 1.0 draws in MeV (the
    native unit of the digitised files), 1e-3 draws in GeV.

    Grey filled regions: NOMAD, CHARM-II, SN1987A, SuperK (Gustafson et al.),
    Borexino (Plestid). If ``miniboone``, overlay the MiniBooNE 95% CL preferred
    regions (E_nu^QE pink, cos(theta) green) and the best-fit star.

    Returns (l1, l2): the MiniBooNE fill handles (None if miniboone=False),
    for building the 'MiniBooNE 95% CL' legend.
    """
    ec = os.path.join(data_dir, "ExistingConstraints")

    # NOMAD (file: m_N [MeV], d [GeV^-1]; the sqrt(2) matches the published limit)
    e, l = np.genfromtxt(os.path.join(ec, "NOMAD.txt"), unpack=True)
    ax.fill_between(
        e * x_scale,
        l * np.sqrt(2),
        np.ones(len(l)),
        edgecolor="None",
        lw=0.25,
        facecolor=facecolor,
        zorder=1,
    )
    ax.fill_between(
        e * x_scale,
        l * np.sqrt(2),
        np.ones(len(l)),
        edgecolor=limit_color,
        lw=0.25,
        facecolor="None",
        zorder=3,
    )
    if annotate:
        ax.annotate(
            "NOMAD",
            xy=(790 * x_scale, 1.0e-6),
            color="grey",
            xycoords="data",
            fontsize=annotation_fontsize,
        )

    # CHARM-II (file: m_N [GeV], neutrino magnetic moment [mu_B]; convert to GeV^-1)
    to_dipole = const.eQED**2 / 2 / const.m_e
    e, l = np.genfromtxt(os.path.join(ec, "CHARM.txt"), unpack=True)
    ax.fill_between(
        e * 1e3 * x_scale,
        l * to_dipole,
        np.ones(len(l)),
        edgecolor="None",
        lw=0.25,
        facecolor=facecolor,
        zorder=1,
    )
    ax.fill_between(
        e * 1e3 * x_scale,
        l * to_dipole,
        np.ones(len(l)),
        edgecolor=limit_color,
        lw=0.25,
        facecolor="None",
        zorder=3,
    )
    if annotate:
        ax.annotate(
            "CHARM-II",
            xy=(15 * x_scale, 2.8e-7),
            color="grey",
            xycoords="data",
            fontsize=annotation_fontsize,
        )

    # SN1987A (file: m_N [MeV], d [GeV^-1]; excludes couplings BELOW the curve)
    e, l = np.genfromtxt(os.path.join(ec, "SN1987A.txt"), unpack=True)
    ax.fill_between(
        e * x_scale,
        np.zeros(len(l)),
        l,
        edgecolor="None",
        lw=0.25,
        facecolor=facecolor,
        zorder=1,
    )
    ax.fill_between(
        e * x_scale,
        np.zeros(len(l)),
        l,
        edgecolor=limit_color,
        lw=0.25,
        facecolor="None",
        zorder=1,
    )
    if annotate:
        ax.annotate(
            "SN1987A",
            xy=(11 * x_scale, 3.5e-8),
            color="grey",
            xycoords="data",
            fontsize=annotation_fontsize,
        )

    # SuperK (Gustafson et al.; file: m_N [GeV], d [MeV^-1] -> MeV, GeV^-1)
    e, l = np.genfromtxt(os.path.join(ec, "Gustafson_SK.dat"), unpack=True)
    x, y = plot_closed_region((e * 1e3, l * 1e3))
    x = x * x_scale
    ax.plot(x, y, color=limit_color, lw=0.25, zorder=3)
    ax.fill(x, y, edgecolor="None", lw=0.25, facecolor=facecolor, zorder=1)
    if annotate:
        ax.annotate(
            "SuperK",
            xy=(62 * x_scale, 0.6e-6),
            color="grey",
            xycoords="data",
            fontsize=annotation_fontsize,
            rotation=53,
        )

    # Borexino (Plestid; file: m_N [GeV], d [MeV^-1] -> MeV, GeV^-1)
    e, l = np.genfromtxt(os.path.join(ec, "Plestid_borexino.dat"), unpack=True)
    x, y = plot_closed_region((e * 1e3, l * 1e3))
    x = x * x_scale
    ax.plot(x, y, color=limit_color, lw=0.25, zorder=3)
    ax.fill(x, y, edgecolor="None", lw=0.25, facecolor=facecolor, zorder=1)
    if annotate:
        ax.annotate(
            "Borexino",
            xy=(10.2 * x_scale, 1e-7),
            color="grey",
            xycoords="data",
            fontsize=annotation_fontsize,
            rotation=80,
        )

    # LEP
    lep_d_lim = 2e-5  # GeV^-1
    ax.fill_between(
        [10 * x_scale, 5e5 * x_scale], lep_d_lim, 1.0, color=facecolor, zorder=1
    )
    ax.axhline(lep_d_lim, color=limit_color, lw=0.25, ls="-", zorder=3)
    if annotate:
        ax.text(12 * x_scale, 2.15e-5, "LEP", color="gray", fontsize=8, va="bottom")

    l1 = l2 = None
    if miniboone:
        mb = os.path.join(data_dir, "MiniBooNE_dtau_0")
        m95, d95 = np.genfromtxt(
            os.path.join(mb, "EnuQE_0.950_CL_Path0.txt"), unpack=True
        )
        ax.plot(m95 * x_scale, d95, c="deeppink", ls="-", zorder=2, lw=0.5)
        (l1,) = ax.fill(m95 * x_scale, d95, facecolor="deeppink", alpha=0.5, zorder=2)

        m95, d95 = np.genfromtxt(
            os.path.join(mb, "CosTheta_0.950_CL_Path0.txt"), unpack=True
        )
        ax.plot(m95 * x_scale, d95, c="limegreen", ls="-", zorder=2, lw=0.5)
        (l2,) = ax.fill(
            m95 * x_scale, d95, facecolor="limegreen", ls="-", alpha=0.5, zorder=2
        )

        # MiniBooNE best-fit point
        ax.scatter(
            [472 * x_scale],
            [1.25e-6],
            marker="*",
            s=40,
            lw=0.5,
            facecolor="black",
            edgecolor="black",
            zorder=3,
        )

    return l1, l2


def plot_existing_limits_de(
    ax,
    data_dir=DATA_DIR,
    x_scale=1.0,
    limit_color="dimgrey",
    facecolor=LIGHT_GREY,
    annotate=True,
    annotation_fontsize=8,
):
    """Draw the existing d_{e N} constraints on ``ax`` (m_N vs d [GeV^-1]).

    ``x_scale`` multiplies every mass before plotting (1.0 = MeV, 1e-3 = GeV).

    Filled regions: Borexino (solar neutrinos; the oscillated flux constrains
    all flavors at the same order), SN1987A (flavor-blind production), LSND,
    XENON1T (solar neutrinos upscattering in the xenon target) and the
    flavor-independent LEP bound on d.
    """
    ec = os.path.join(data_dir, "ExistingConstraints")

    # Borexino (Plestid; file: m_N [GeV], d [MeV^-1] -> MeV, GeV^-1)
    e, l = np.genfromtxt(os.path.join(ec, "Plestid_borexino.dat"), unpack=True)
    x, y = plot_closed_region((e * 1e3, l * 1e3), logx=True, logy=True)
    x = x * x_scale
    ax.plot(x, y, color=limit_color, lw=0.25, zorder=3)
    ax.fill_between(x, y, 1.0, edgecolor="None", lw=0.25, facecolor=facecolor, zorder=1)
    if annotate:
        ax.annotate(
            "Borexino",
            xy=(1.3 * x_scale, 2e-7),
            color="grey",
            xycoords="data",
            fontsize=annotation_fontsize,
        )

    # SN1987A (flavor-blind; excludes couplings BELOW the curve)
    e, l = np.genfromtxt(os.path.join(ec, "SN1987A.txt"), unpack=True)
    ax.fill_between(
        e * x_scale,
        np.zeros(len(l)),
        l,
        edgecolor="None",
        lw=0.25,
        facecolor=facecolor,
        zorder=1,
    )
    ax.fill_between(
        e * x_scale,
        np.zeros(len(l)),
        l,
        edgecolor=limit_color,
        lw=0.25,
        facecolor="None",
        zorder=3,
    )
    if annotate:
        ax.annotate(
            "SN1987A",
            xy=(11 * x_scale, 3.5e-8),
            color="grey",
            xycoords="data",
            fontsize=annotation_fontsize,
        )

    # LSND and XENON1T (files: m_N [GeV], d [GeV^-1] -- despite the "dmu"
    # header, which is a copy-paste in the digitisation). Both are single-valued
    # boundary curves with the usual reach shape (best in the middle, degrading
    # at both mass ends), so the excluded region lies ABOVE the curve, as for
    # NOMAD/CHARM-II in the d_{mu N} case.
    for fname, name, xy in (
        ("LSND_de.dat", "LSND", (30, 3.0e-6)),
        ("XENON1T_de.dat", "XENON1T", (1.5, 4.0e-6)),
    ):
        e, l = np.genfromtxt(os.path.join(data_dir, fname), unpack=True)
        ax.fill_between(
            e * 1e3 * x_scale,
            l,
            np.ones(len(l)),
            edgecolor="None",
            lw=0.25,
            facecolor=facecolor,
            zorder=1,
        )
        ax.fill_between(
            e * 1e3 * x_scale,
            l,
            np.ones(len(l)),
            edgecolor=limit_color,
            lw=0.25,
            facecolor="None",
            zorder=3,
        )
        if annotate:
            ax.annotate(
                name,
                xy=(xy[0] * x_scale, xy[1]),
                color="grey",
                xycoords="data",
                fontsize=annotation_fontsize,
            )

    # LEP (flavor independent: e+e- -> nu nu gamma)
    lep_d_lim = 2e-5  # GeV^-1
    ax.fill_between(
        [10 * x_scale, 5e5 * x_scale], lep_d_lim, 1.0, color=facecolor, zorder=1
    )
    ax.axhline(lep_d_lim, color=limit_color, lw=0.25, ls="-", zorder=3)
    if annotate:
        ax.text(12 * x_scale, 2.15e-5, "LEP", color="gray", fontsize=8, va="bottom")

    # NOTE: the SHiP projections are NOT drawn here -- the notebook draws them
    # for both couplings so they get a single, consistent legend entry.
