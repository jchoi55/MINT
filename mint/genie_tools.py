"""Export MINT neutrino fluxes as GENIE flux drivers.

GENIE's tabulated-flux drivers read a spectrum as a 1-D ROOT histogram: one
``TH1D`` per neutrino species, bin content proportional to :math:`d\\Phi/dE`,
energies in GeV.  :func:`export_flux` takes the per-flavor fluxes MINT produces
(``neutrinos / cm^2 / GeV / year``, on whatever binning the flux was computed),
resamples them onto the uniform grid GENIE wants, and writes the ROOT file plus
a JSON sidecar recording where the numbers came from.

Two things about GENIE's histogram flux are easy to get wrong, and both are why
this module resamples instead of writing the MINT bins straight out:

*Bin widths.*  GENIE samples the spectrum with ``TH1::GetRandom()``, and ROOT's
implementation treats each bin's *content* as that bin's probability -- it never
multiplies by the bin width.  A log-spaced histogram of a flux *density* is
therefore sampled wrongly unless the widths are folded in first.  Recent GENIE
detects variable-width axes and folds them in for you (the ``WIDTH``/``NOWIDTH``
suffix overrides it), but older builds do not, so the safe format is a uniform
grid, where density and per-bin count differ only by a constant.

*Normalization.*  GENIE uses the histogram as a shape only; the absolute scale
is dropped (all that survives is the *relative* normalization between species in
the multi-flavor drivers).  Turning a ``gevgen`` sample back into a rate is
therefore up to you, so :func:`export_flux` reports, per flavor, the flux
integral and -- if you hand it the interacting-event rates -- the weight
``events_per_year / n_events`` that each generated event carries.

Neither driver here carries ray directions or vertices: ``gevgen`` puts every
vertex at the origin and fires along ``+z``.  That is enough to generate an
interaction sample with the right energy and flavor content, which is what a
flux histogram can give you.  A full detector simulation with beam geometry
needs a ray-level format (``dk2nu`` or ``gsimple``) and ``gevgen_fnal``.

Given a local GENIE build, :class:`GenieInstall` and :func:`run_gevgen_all`
drive ``gevgen`` (one process per flavor) and ``gntpc``, :func:`read_gst`
loads the resulting summary trees, and :func:`genie_normalization` restores
the rate GENIE dropped, from the flux histograms and GENIE's own total cross
sections.

Writing ROOT files needs ``uproot`` (``pip install mint-muc[genie]``); it is
imported only when you actually write one.  :func:`write_flux_text` needs
nothing beyond numpy, at the cost of GENIE re-histogramming the table.
"""

import collections
import concurrent.futures
import datetime
import json
import os
import subprocess
import time

import numpy as np

import mint


# GENIE identifies species by PDG code; these are the six a muon beam can give
# you after oscillation.
PDG_CODES = {
    "nue": 12,
    "nuebar": -12,
    "numu": 14,
    "numubar": -14,
    "nutau": 16,
    "nutaubar": -16,
}

NU_FLAVORS = tuple(PDG_CODES)

# GENIE target codes (PDG nuclear format 10LZZZAAAI).
AR40 = 1000180400   # liquid argon: the DUNE far detector

FLUX_UNITS = "neutrinos / cm^2 / GeV / year"

# Directory names gspl2root gives each probe in a cross-section graph file
# (``<probe>_<target>``, e.g. ``nu_mu_bar_Ar40``).
GRAPH_PROBE_NAMES = {
    "nue": "nu_e",
    "nuebar": "nu_e_bar",
    "numu": "nu_mu",
    "numubar": "nu_mu_bar",
    "nutau": "nu_tau",
    "nutaubar": "nu_tau_bar",
}

# Mass number of the GENIE target codes we hand out, for nucleon -> nucleus.
TARGET_A = {AR40: 40}
TARGET_NAMES = {AR40: "Ar40"}


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def uniform_energy_edges(e_max, bin_width=0.1, e_min=0.0):
    """Uniform bin edges from ``e_min`` to ``e_max`` in steps of ``bin_width``.

    The uniform grid is what makes GENIE's sampling unambiguous (see the module
    docstring).  ``bin_width = 0.1`` GeV matches the convention of the public
    DUNE beam flux files; use something finer if the oscillation structure you
    care about is narrower than that.

    ``e_max`` is rounded up to the next whole bin so the last edge is not
    truncated.
    """
    if bin_width <= 0:
        raise ValueError("bin_width must be positive")
    if e_max <= e_min:
        raise ValueError("e_max must exceed e_min")
    n_bins = int(np.ceil((e_max - e_min) / bin_width - 1e-9))
    return e_min + bin_width * np.arange(n_bins + 1, dtype=float)


def flux_integral(energy_edges, flux):
    """Integrate a flux density over its bins: ``sum(flux * dE)``."""
    edges = np.asarray(energy_edges, dtype=float)
    return float(np.sum(np.asarray(flux, dtype=float) * np.diff(edges)))


def _loglog_bin_averages(centers, values, e_lo, e_hi, new_edges):
    """Mean density in each output bin of the log-log interpolant.

    The interpolant is a power law :math:`y_i (E/x_i)^p` on each interval
    between input bin centers, held flat out to ``e_lo``/``e_hi`` and zero
    outside them.  Each piece is integrated in closed form and the cumulative
    differenced on the output edges, so the result is the exact bin average --
    which matters at the ends of the support and wherever the spectrum turns
    over inside one output bin, where sampling the interpolant at the bin
    center instead is badly wrong.
    """
    xs = np.concatenate([[e_lo], centers, [e_hi]])
    ys = np.concatenate([[values[0]], values, [values[-1]]])

    with np.errstate(divide="ignore", invalid="ignore"):
        index = np.diff(np.log(ys)) / np.diff(np.log(xs))
    # The two end pieces are flat by construction; reading their index off a
    # (possibly zero) outer edge would take log(0).
    index[0] = 0.0
    index[-1] = 0.0

    def _partial(i, upper):
        """Integral of segment ``i`` from its left node up to ``upper``."""
        a, ya, p = xs[i], ys[i], index[i]
        q = p + 1.0
        # p = -1 integrates to a logarithm; everything else to a power.
        is_log = np.abs(q) < 1e-12
        with np.errstate(divide="ignore", invalid="ignore"):
            power = ya * (upper * (upper / a) ** p - a) / np.where(is_log, 1.0, q)
            logarithmic = ya * a * np.log(np.where(is_log, upper / a, 1.0))
        return np.where(is_log, logarithmic, power)

    segments = np.arange(xs.size - 1)
    cumulative = np.concatenate([[0.0], np.cumsum(_partial(segments, xs[1:]))])

    at = np.clip(new_edges, xs[0], xs[-1])
    which = np.clip(np.searchsorted(xs, at, side="right") - 1, 0, xs.size - 2)
    integral = cumulative[which] + _partial(which, at)
    return np.diff(integral) / np.diff(new_edges)


def resample_flux(energy_edges, flux, new_edges, method="loglog"):
    """Move a flux density from one binning to another.

    ``flux`` is a density (per GeV) defined on the bins of ``energy_edges``, as
    returned by :meth:`mint.MuDecaySimulator.get_flux_at_generic_location`, and
    the result is a density on the bins of ``new_edges``.

    ``method`` picks what happens between the input bin centers:

    ``"loglog"`` (default)
        Linear interpolation of :math:`\\log \\Phi` against :math:`\\log E`
        through the input bin centers, integrated in closed form and *averaged*
        over each output bin.  A neutrino spectrum is close to a power law in
        pieces, so this is smooth and accurate when refining a coarse grid --
        the usual case here, since a log-spaced MINT grid is much coarser than a
        GENIE grid at the energies that matter.  Averaging rather than sampling
        the interpolant at the bin center is what keeps the integral right where
        the spectrum turns over inside one output bin.

        Beyond the first and last input centers the interpolant is held flat, so
        the outer half-bins reproduce what the input histogram says there, and
        it is cut to zero outside the input range -- a falling spectrum is never
        extrapolated upward.  Those flat end pieces are also why the integral is
        conserved only to a fraction of a percent rather than exactly: over the
        outermost half-bin a constant is not the power law.  Bins with
        non-positive content are dropped from the interpolation, so MC zeros are
        bridged rather than punched through.

    ``"conserve"``
        Treat the input as exactly piecewise constant and redistribute it, which
        conserves the integral bin by bin.  Use it when coarsening, or whenever
        the integral matters more than the shape between centers.  Refining with
        it leaves visible steps at the old bin edges.

    Interpolating a *smooth* spectrum is safe; interpolating one that already
    carries oscillation wiggles is not, because the wiggles are narrower than
    the input bins.  Resample the unoscillated flux and apply the oscillation
    probabilities on the fine grid afterwards.
    """
    edges = np.asarray(energy_edges, dtype=float)
    values = np.asarray(flux, dtype=float)
    new_edges = np.asarray(new_edges, dtype=float)
    if values.shape != (edges.size - 1,):
        raise ValueError(
            f"flux has {values.size} entries but energy_edges defines {edges.size - 1} bins"
        )

    if method == "conserve":
        # Cumulative flux at the input edges, linearly interpolated in between:
        # differencing it on the new edges conserves the integral exactly.
        cumulative = np.concatenate([[0.0], np.cumsum(values * np.diff(edges))])
        at_new = np.interp(new_edges, edges, cumulative,
                           left=0.0, right=cumulative[-1])
        return np.diff(at_new) / np.diff(new_edges)

    if method != "loglog":
        raise ValueError(f"unknown method {method!r}; use 'loglog' or 'conserve'")

    if np.all(edges > 0):
        centers = np.sqrt(edges[1:] * edges[:-1])
    else:
        centers = 0.5 * (edges[1:] + edges[:-1])

    support = (values > 0) & (centers > 0)
    if support.sum() < 2:
        raise ValueError("need at least two positive flux bins to interpolate")
    filled = np.flatnonzero(support)
    # The input histogram carries content over the outer edges of its outermost
    # filled bins, not just between their centers.
    e_lo, e_hi = edges[filled[0]], edges[filled[-1] + 1]

    return _loglog_bin_averages(centers[support], values[support], e_lo, e_hi, new_edges)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _require_uproot():
    try:
        import uproot
    except ImportError as exc:   # pragma: no cover - depends on the environment
        raise ImportError(
            "writing GENIE flux histograms needs uproot: pip install 'mint-muc[genie]' "
            "(or use mint.genie_tools.write_flux_text, which needs only numpy)"
        ) from exc
    return uproot


def histogram_name(flavor):
    """Name of the ``TH1D`` holding one flavor's flux."""
    return f"flux_{flavor}"


def write_flux_root(path, energy_edges, fluxes, metadata=None):
    """Write one ``TH1D`` per flavor to ``path``, plus a metadata ``TObjString``.

    ``fluxes`` maps flavor names (keys of :data:`PDG_CODES`) to densities on the
    bins of ``energy_edges``.  All flavors share the binning, so their relative
    normalization is preserved -- which is what the multi-flavor drivers use.

    Returns the map from flavor to histogram name.
    """
    uproot = _require_uproot()
    edges = np.asarray(energy_edges, dtype=float)

    unknown = set(fluxes) - set(PDG_CODES)
    if unknown:
        raise ValueError(f"unknown neutrino flavors {sorted(unknown)}; expected {NU_FLAVORS}")

    names = {}
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    with uproot.recreate(path) as f:
        for flavor, flux in fluxes.items():
            values = np.asarray(flux, dtype=float)
            if values.shape != (edges.size - 1,):
                raise ValueError(
                    f"{flavor}: {values.size} flux entries against {edges.size - 1} bins"
                )
            if np.any(values < 0):
                raise ValueError(f"{flavor}: flux has negative bins")
            # float64 in, so uproot writes a TH1D -- GENIE C-casts the object to
            # TH1D* without checking, and a TH1F read that way is garbage.
            names[flavor] = histogram_name(flavor)
            f[names[flavor]] = (values, edges)
        if metadata is not None:
            f["mint_metadata"] = json.dumps(metadata, indent=2, default=str)
    return names


def write_flux_text(path_stem, energy_edges, fluxes):
    """Write GENIE "vector file" flux tables: two columns, ``energy  flux``.

    One file per flavor, ``{path_stem}_{flavor}.dat``.  This is the no-ROOT
    fallback: ``gevgen -f file.dat`` accepts it, but it splines the table and
    then accept-rejects it into 300 uniform bins over the ``-e`` range, so the
    ROOT histogram from :func:`write_flux_root` is the more faithful input.

    Returns the map from flavor to file path.
    """
    edges = np.asarray(energy_edges, dtype=float)
    centers = 0.5 * (edges[1:] + edges[:-1])
    directory = os.path.dirname(os.path.abspath(path_stem))
    if directory:
        os.makedirs(directory, exist_ok=True)

    written = {}
    for flavor, flux in fluxes.items():
        values = np.asarray(flux, dtype=float)
        if values.shape != centers.shape:
            raise ValueError(
                f"{flavor}: {values.size} flux entries against {centers.size} bins"
            )
        out = f"{path_stem}_{flavor}.dat"
        np.savetxt(
            out,
            np.column_stack([centers, values]),
            fmt="%.8e",
            header=f"E_nu [GeV]   dPhi/dE [{FLUX_UNITS}]   (flavor {flavor}, PDG {PDG_CODES[flavor]})",
        )
        written[flavor] = out
    return written


# ---------------------------------------------------------------------------
# Normalization and run commands
# ---------------------------------------------------------------------------

def normalization_summary(energy_edges, fluxes, event_rates=None, n_events=None):
    """Per-flavor bookkeeping to get a rate back out of a GENIE sample.

    GENIE keeps only the shape of the flux, so a generated sample has to be
    reweighted by hand.  For each flavor this returns

    ``flux_integral``
        :math:`\\int dE\\, d\\Phi/dE` in neutrinos / cm^2 / year.
    ``flux_fraction``
        that integral as a fraction of all flavors, i.e. how to split a fixed
        number of events between single-flavor ``gevgen`` runs.
    ``events_per_year``
        the interacting-event rate, if ``event_rates`` is given (densities in
        events / GeV / year on the same bins).
    ``weight_per_event``
        ``events_per_year / n_events``: what one generated event counts as,
        given ``n_events`` generated for that flavor.  ``n_events`` may be a
        single number or a per-flavor dict.
    """
    total_flux = sum(flux_integral(energy_edges, f) for f in fluxes.values())

    summary = {}
    for flavor, flux in fluxes.items():
        entry = {
            "pdg": PDG_CODES[flavor],
            "histogram": histogram_name(flavor),
            "flux_integral": flux_integral(energy_edges, flux),
        }
        entry["flux_fraction"] = entry["flux_integral"] / total_flux if total_flux > 0 else 0.0
        if event_rates is not None and flavor in event_rates:
            rate = flux_integral(energy_edges, event_rates[flavor])
            entry["events_per_year"] = rate
            n = n_events.get(flavor) if isinstance(n_events, dict) else n_events
            if n:
                entry["n_events"] = int(n)
                entry["weight_per_event"] = rate / int(n)
        summary[flavor] = entry
    return summary


def energy_range(energy_edges):
    """The ``-e emin,emax`` pair covering a flux histogram.

    Padded by one bin at the top because GENIE zeroes any bin whose upper edge
    compares greater than ``emax``, which floating-point edges can do to the
    last bin.
    """
    edges = np.asarray(energy_edges, dtype=float)
    width = float(edges[-1] - edges[-2])
    return float(edges[0]), float(edges[-1]) + width


def gevgen_argv(
    root_path,
    flavor,
    energy_edges,
    n_events,
    output,
    target=AR40,
    tune="G18_02a_00_000",
    cross_sections="$GENIE/data/xsec.xml",
    run_number=None,
    seed=None,
    event_generator_list="Default",
    message_thresholds=None,
    executable="gevgen",
):
    """The ``gevgen`` argument list for one flavor, ready for ``subprocess``."""
    e_lo, e_hi = energy_range(energy_edges)
    argv = [
        executable,
        "-n", str(int(n_events)),
        "-p", str(PDG_CODES[flavor]),
        "-t", str(int(target)),
        "-e", f"{e_lo:g},{e_hi:g}",
        "-f", f"{root_path},{histogram_name(flavor)}",
        "--tune", tune,
        "--cross-sections", str(cross_sections),
        "--event-generator-list", event_generator_list,
        "-o", str(output),
    ]
    if run_number is not None:
        argv += ["-r", str(int(run_number))]
    if seed is not None:
        argv += ["--seed", str(int(seed))]
    if message_thresholds is not None:
        argv += ["--message-thresholds", str(message_thresholds)]
    return argv


def gevgen_commands(
    root_path,
    fluxes,
    energy_edges,
    target=AR40,
    n_events=100_000,
    tune="G18_02a_00_000",
    cross_sections="$GENIE/data/xsec.xml",
    output_prefix=None,
    seed=None,
):
    """One ``gevgen`` command line per flavor.

    ``gevgen`` takes a single ``-p`` species per job, so a six-flavor export is
    six runs; :func:`gevgen_t2k_command` does all of them at once instead, and
    :func:`run_gevgen_all` runs the six from Python.
    """
    per_flavor = {}
    for flavor in fluxes:
        n = n_events.get(flavor) if isinstance(n_events, dict) else n_events
        prefix = output_prefix or "gntp"
        argv = gevgen_argv(
            root_path, flavor, energy_edges, n, f"{prefix}_{flavor}",
            target=target, tune=tune, cross_sections=cross_sections, seed=seed,
        )
        # drop the generator-list default to keep the printed line short
        i = argv.index("--event-generator-list")
        del argv[i:i + 2]
        per_flavor[flavor] = " ".join(argv)
    return per_flavor


def gevgen_t2k_command(
    root_path,
    fluxes,
    target=AR40,
    n_events=100_000,
    tune="G18_02a_00_000",
    cross_sections="$GENIE/data/xsec.xml",
    output_prefix=None,
    seed=None,
):
    """A single ``gevgen_t2k`` command generating every flavor in one job.

    Despite the name it is a generic histogram-flux driver: with
    ``-f file.root,12[hist],-12[hist],...`` it samples all the listed species,
    splitting events between them by the histograms' relative normalization --
    which is why they must share a binning and a common flux unit, as
    :func:`write_flux_root` writes them.

    It is only built if GENIE was configured with ``--enable-t2k``;
    :func:`gevgen_commands` needs nothing beyond the base install.
    """
    spec = ",".join(f"{PDG_CODES[flavor]}[{histogram_name(flavor)}]" for flavor in fluxes)
    cmd = [
        "gevgen_t2k",
        f"-f {root_path},{spec}",
        f"-g {target}",
        f"-n {int(n_events)}",
        f"--tune {tune}",
        f"--cross-sections {cross_sections}",
    ]
    if output_prefix:
        cmd.append(f"-o {output_prefix}")
    if seed is not None:
        cmd.append(f"--seed {int(seed)}")
    return " ".join(cmd)


# ---------------------------------------------------------------------------
# One-call export
# ---------------------------------------------------------------------------

def export_flux(
    path,
    energy_edges,
    fluxes,
    event_rates=None,
    new_edges=None,
    bin_width=0.1,
    e_max=None,
    method="loglog",
    target=AR40,
    n_events=100_000,
    tune="G18_02a_00_000",
    cross_sections="$GENIE/data/xsec.xml",
    metadata=None,
    write_text=False,
):
    """Resample MINT fluxes onto a GENIE grid and write the flux file.

    Parameters
    ----------
    path
        Output ROOT file.  A JSON sidecar with the same stem gets the metadata
        and the normalization table.
    energy_edges, fluxes
        The input binning and a ``{flavor: dPhi/dE}`` map on it, in
        neutrinos / cm^2 / GeV / year.
    event_rates
        Optional ``{flavor: dN/dE}`` interacting-event densities on the *same*
        input bins, used for the per-event weights in the summary.
    new_edges, bin_width, e_max
        The GENIE grid.  Give ``new_edges`` outright, or let it be built as
        :func:`uniform_energy_edges` from ``bin_width`` and ``e_max``
        (default: the top of the input range).
    method
        Resampling rule, see :func:`resample_flux`.

    Returns a dict with the output paths, the grid, the resampled fluxes, the
    normalization table, and ready-to-run ``gevgen`` command lines.
    """
    edges_in = np.asarray(energy_edges, dtype=float)
    if new_edges is None:
        new_edges = uniform_energy_edges(
            float(edges_in[-1]) if e_max is None else float(e_max), bin_width=bin_width
        )
    new_edges = np.asarray(new_edges, dtype=float)

    # Already on the target grid (the caller built the spectra there, e.g. to
    # evaluate oscillation probabilities at full resolution): nothing to do.
    identity = edges_in.shape == new_edges.shape and np.allclose(edges_in, new_edges)

    def _move(values):
        if identity:
            return np.asarray(values, dtype=float)
        return resample_flux(edges_in, values, new_edges, method=method)

    resampled = {flavor: _move(flux) for flavor, flux in fluxes.items()}
    resampled_rates = None
    if event_rates is not None:
        resampled_rates = {flavor: _move(rate) for flavor, rate in event_rates.items()}

    # Rates are quoted from the input grid, where they were computed; the
    # resampled ones only reach the summary through weight_per_event.
    norm = normalization_summary(
        new_edges, resampled,
        event_rates=resampled_rates, n_events=n_events,
    )
    if event_rates is not None:
        for flavor, entry in norm.items():
            if flavor in event_rates:
                entry["events_per_year_input_grid"] = flux_integral(
                    edges_in, event_rates[flavor]
                )

    full_metadata = {
        "generator": f"MINT {mint.__version__} (mint.genie_tools)",
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "flux_units": FLUX_UNITS,
        "energy_units": "GeV",
        "resampling": {
            "method": "none (input already on the output grid)" if identity else method,
            "input_bins": int(edges_in.size - 1),
            "input_range_GeV": [float(edges_in[0]), float(edges_in[-1])],
            "output_bins": int(new_edges.size - 1),
            "output_range_GeV": [float(new_edges[0]), float(new_edges[-1])],
            "bin_width_GeV": float(new_edges[1] - new_edges[0]),
        },
        "genie_target_pdg": int(target),
        "normalization": norm,
    }
    if metadata:
        full_metadata.update(metadata)

    root_path = os.fspath(path)
    write_flux_root(root_path, new_edges, resampled, metadata=full_metadata)

    stem = os.path.splitext(root_path)[0]
    json_path = f"{stem}.json"
    with open(json_path, "w") as fh:
        json.dump(full_metadata, fh, indent=2, default=str)

    text_paths = None
    if write_text:
        text_paths = write_flux_text(stem, new_edges, resampled)

    return {
        "root_path": root_path,
        "json_path": json_path,
        "text_paths": text_paths,
        "energy_edges": new_edges,
        "fluxes": resampled,
        "event_rates": resampled_rates,
        "normalization": norm,
        "metadata": full_metadata,
        "gevgen": gevgen_commands(
            root_path, resampled, new_edges, target=target, n_events=n_events,
            tune=tune, cross_sections=cross_sections,
        ),
        "gevgen_t2k": gevgen_t2k_command(
            root_path, resampled, target=target, n_events=n_events,
            tune=tune, cross_sections=cross_sections,
        ),
    }


# ---------------------------------------------------------------------------
# Running GENIE
# ---------------------------------------------------------------------------

class GenieInstall:
    """Where GENIE lives and which physics to run it with.

    Collects the handful of paths a ``gevgen`` job needs -- the install, the
    pre-computed cross-section splines, the tune they were made with -- and
    turns them into the environment and command lines.  ``spline_xml`` and
    ``tune`` must match: GENIE checks the tune recorded in the spline file and
    recomputes anything missing, which for argon takes hours.

    ``xsec_graphs`` is the ``xsec_graphs.root`` shipped next to the FNAL spline
    sets (``gspl2root`` output for every probe/target pair).  It is what
    :func:`genie_normalization` reads the total cross sections from.

    ``extra_lib_dirs`` are prepended to ``DYLD_LIBRARY_PATH`` /
    ``LD_LIBRARY_PATH`` -- wherever LHAPDF, Pythia8, and the libxml2 GENIE was
    linked against live on this machine.
    """

    def __init__(
        self,
        genie_dir,
        spline_xml,
        tune,
        xsec_graphs=None,
        extra_lib_dirs=(),
        messenger="Messenger_laconic.xml",
    ):
        self.genie_dir = os.path.abspath(os.path.expanduser(genie_dir))
        self.spline_xml = os.path.abspath(os.path.expanduser(spline_xml))
        self.tune = tune
        if xsec_graphs is None:
            candidate = os.path.join(os.path.dirname(self.spline_xml), "xsec_graphs.root")
            xsec_graphs = candidate if os.path.exists(candidate) else None
        self.xsec_graphs = (
            os.path.abspath(os.path.expanduser(xsec_graphs)) if xsec_graphs else None
        )
        self.extra_lib_dirs = [os.path.expanduser(d) for d in extra_lib_dirs]
        self.messenger = (
            os.path.join(self.genie_dir, "config", messenger)
            if messenger and not os.path.isabs(messenger) else messenger
        )

    def binary(self, name):
        return os.path.join(self.genie_dir, "bin", name)

    def check(self):
        """Raise with a useful message if anything a run needs is missing."""
        for what, path in (("gevgen", self.binary("gevgen")),
                           ("gntpc", self.binary("gntpc")),
                           ("spline file", self.spline_xml)):
            if not os.path.exists(path):
                raise FileNotFoundError(f"GENIE {what} not found at {path}")
        if self.messenger and not os.path.exists(self.messenger):
            raise FileNotFoundError(f"message-thresholds file not found: {self.messenger}")

    def env(self):
        """The process environment for a GENIE job."""
        env = os.environ.copy()
        env["GENIE"] = self.genie_dir
        env["GXMLPATH"] = os.path.dirname(self.spline_xml)
        env["PATH"] = os.path.join(self.genie_dir, "bin") + os.pathsep + env.get("PATH", "")
        libs = self.extra_lib_dirs + [os.path.join(self.genie_dir, "lib")]
        for var in ("DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"):
            env[var] = os.pathsep.join(libs + ([env[var]] if env.get(var) else []))
        return env


def _run_logged(argv, cwd, env, log_path, tail_lines):
    """Run a command, keeping only the last ``tail_lines`` of its output.

    A dev build of GENIE prints a full Pythia8 event listing for every DIS
    event (a hard-coded ``event.list()`` in Pythia8Hadro2019), which is about
    a kilobyte per event, so a 10^5-event job would leave a 100 MB log.  The
    ring buffer bounds that while keeping the part that matters when a job
    dies: the end.
    """
    tail = collections.deque(maxlen=tail_lines)
    dropped = 0
    t0 = time.time()
    proc = subprocess.Popen(
        argv, cwd=cwd, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    for line in proc.stdout:
        if len(tail) == tail.maxlen:
            dropped += 1
        tail.append(line)
    proc.wait()
    seconds = time.time() - t0
    with open(log_path, "w") as fh:
        fh.write("$ " + " ".join(argv) + "\n")
        fh.write(f"# exit {proc.returncode} after {seconds:.1f} s"
                 + (f"; first {dropped} lines dropped\n" if dropped else "\n"))
        fh.writelines(tail)
    return proc.returncode, seconds


def run_gevgen(
    install,
    flux_root,
    flavor,
    n_events,
    workdir,
    run_number=1,
    seed=None,
    target=AR40,
    convert=True,
    tail_lines=400,
):
    """Generate ``n_events`` interactions of one flavor from an exported flux.

    Runs ``gevgen`` in ``workdir/<flavor>/`` -- its own directory, because
    ``gevgen`` also writes ``./input-flux.root`` and a ``.status`` file, which
    parallel jobs must not share -- and then ``gntpc -f gst`` to get the flat
    summary tree :func:`read_gst` reads.

    Returns a dict with the output paths, exit codes and wall time.  It does
    not raise on a failed job; check ``returncode`` (and the log).
    """
    install.check()
    import uproot
    flux_root = os.path.abspath(flux_root)
    with uproot.open(flux_root) as f:
        edges = f[histogram_name(flavor)].axis().edges()

    jobdir = os.path.join(os.path.abspath(workdir), flavor)
    os.makedirs(jobdir, exist_ok=True)
    ghep = os.path.join(jobdir, f"{flavor}.ghep.root")
    gst = os.path.join(jobdir, f"{flavor}.gst.root")

    argv = gevgen_argv(
        flux_root, flavor, edges, n_events, ghep,
        target=target, tune=install.tune, cross_sections=install.spline_xml,
        run_number=run_number, seed=seed, message_thresholds=install.messenger,
        executable=install.binary("gevgen"),
    )
    rc, seconds = _run_logged(argv, jobdir, install.env(),
                              os.path.join(jobdir, "gevgen.log"), tail_lines)
    result = {
        "flavor": flavor, "n_events": int(n_events), "ghep": ghep, "gst": None,
        "log": os.path.join(jobdir, "gevgen.log"),
        "returncode": rc, "seconds": seconds, "gntpc_returncode": None,
    }
    if rc == 0 and convert:
        argv = [install.binary("gntpc"), "-i", ghep, "-f", "gst", "-o", gst]
        if install.messenger:
            argv += ["--message-thresholds", install.messenger]
        rc2, _ = _run_logged(argv, jobdir, install.env(),
                             os.path.join(jobdir, "gntpc.log"), tail_lines)
        result["gntpc_returncode"] = rc2
        if rc2 == 0:
            result["gst"] = gst
    return result


def run_gevgen_all(
    install,
    flux_root,
    n_events,
    workdir,
    flavors=NU_FLAVORS,
    seed=1,
    target=AR40,
    max_workers=None,
    **kwargs,
):
    """:func:`run_gevgen` for several flavors at once, one process each.

    ``n_events`` is a number or a per-flavor dict.  Jobs get distinct run
    numbers and seeds (``seed + i``).  Each loads the spline file on its own,
    so expect ~10 s of start-up per job on top of the generation time.
    """
    flavors = list(flavors)
    max_workers = max_workers or len(flavors)

    def _one(i_flavor):
        i, flavor = i_flavor
        n = n_events.get(flavor) if isinstance(n_events, dict) else n_events
        return run_gevgen(
            install, flux_root, flavor, n, workdir,
            run_number=i + 1, seed=None if seed is None else seed + i,
            target=target, **kwargs,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_one, enumerate(flavors)))
    return {r["flavor"]: r for r in results}


# The gst branches a rate study needs.  The full tree has ~130; ask for more
# by name through ``read_gst(..., branches=...)``.
GST_BRANCHES = (
    "iev", "neu", "tgt", "Ev", "El", "pl", "cthl", "wght", "XSec",
    "cc", "nc", "qel", "mec", "res", "dis", "coh",
    "nfp", "nfn", "nfpip", "nfpim", "nfpi0",
)


def read_gst(path, branches=GST_BRANCHES):
    """Read a ``gntpc -f gst`` file into a dict of numpy arrays.

    Energies are in GeV, ``XSec`` is the selected channel's cross section in
    10^-38 cm^2 (not the total -- see :func:`genie_normalization` for that),
    and ``wght`` is 1 for the unweighted events ``gevgen`` produces by default.
    """
    import uproot
    with uproot.open(path) as f:
        return f["gst"].arrays(list(branches), library="np")


def total_xsec(install_or_path, flavor, target=AR40):
    """GENIE's total (CC + NC) cross section vs energy for one flavor.

    Reads ``tot_cc`` and ``tot_nc`` from the ``gspl2root`` graph file and
    returns ``(E_GeV, sigma_cm2)`` **per nucleus**.
    """
    import uproot
    path = install_or_path.xsec_graphs if isinstance(install_or_path, GenieInstall) else install_or_path
    if not path:
        raise FileNotFoundError("no xsec_graphs.root: give GenieInstall(xsec_graphs=...) "
                                "or make one with gspl2root")
    dirname = f"{GRAPH_PROBE_NAMES[flavor]}_{TARGET_NAMES[target]}"
    with uproot.open(path) as f:
        d = f[dirname]
        e_cc, s_cc = d["tot_cc"].values()
        e_nc, s_nc = d["tot_nc"].values()
    energies = np.union1d(e_cc, e_nc)
    sigma = np.interp(energies, e_cc, s_cc, left=0.0) + np.interp(energies, e_nc, s_nc, left=0.0)
    return energies, sigma * 1e-38


def interaction_spectrum(energy_edges, flux, xsec_E, xsec, n_nuclei):
    """``dN/dE`` in events / GeV / year: ``N_nuclei * flux(E) * sigma(E)``.

    ``xsec`` is per nucleus in cm^2 on the grid ``xsec_E`` (as from
    :func:`total_xsec`), interpolated to the flux bin centers and zero below
    the first tabulated energy.
    """
    edges = np.asarray(energy_edges, dtype=float)
    centers = 0.5 * (edges[1:] + edges[:-1])
    sigma = np.interp(centers, xsec_E, xsec, left=0.0, right=xsec[-1])
    return n_nuclei * np.asarray(flux, dtype=float) * sigma


def genie_normalization(install_or_graphs, flux_root, json_path=None, n_generated=None, target=AR40):
    """Events per year from GENIE's cross sections, per flavor.

    GENIE samples the shape of ``flux x sigma`` but drops the scale, so the
    rate is worked out here from the same two ingredients: the exported flux
    histograms (``flux_root``) and GENIE's total cross section
    (:func:`total_xsec`).  The number of target nuclei comes from the
    ``target_nucleons`` the JSON sidecar recorded, divided by the mass number.

    For each flavor the result holds ``flux_integral`` (/cm^2/yr),
    ``sigma_avg_cm2`` (flux-averaged, per nucleus), ``events_per_year_genie``,
    the notebook's ``events_per_year_mint`` for comparison, the ``spectrum``
    (dN/dE on the flux grid) and, given ``n_generated`` (a number or per-flavor
    dict), ``weight_per_event`` -- what each generated event stands for.
    """
    import uproot
    flux_root = os.fspath(flux_root)
    if json_path is None:
        json_path = os.path.splitext(flux_root)[0] + ".json"
    with open(json_path) as fh:
        meta = json.load(fh)
    n_nuclei = meta["target_nucleons"] / TARGET_A[target]

    with uproot.open(flux_root) as f:
        hists = {flavor: f[histogram_name(flavor)].to_numpy()
                 for flavor in NU_FLAVORS if histogram_name(flavor) in f}

    out = {"n_nuclei": n_nuclei, "flavors": {}}
    for flavor, (flux, edges) in hists.items():
        e_x, s_x = total_xsec(install_or_graphs, flavor, target)
        spectrum = interaction_spectrum(edges, flux, e_x, s_x, n_nuclei)
        rate = flux_integral(edges, spectrum)
        phi = flux_integral(edges, flux)
        entry = {
            "pdg": PDG_CODES[flavor],
            "flux_integral": phi,
            "sigma_avg_cm2": rate / (n_nuclei * phi) if phi > 0 else 0.0,
            "events_per_year_genie": rate,
            "events_per_year_mint": meta["normalization"].get(flavor, {}).get("events_per_year"),
            "energy_edges": edges,
            "spectrum": spectrum,
        }
        n = n_generated.get(flavor) if isinstance(n_generated, dict) else n_generated
        if n:
            entry["n_generated"] = int(n)
            entry["weight_per_event"] = rate / int(n)
        out["flavors"][flavor] = entry
    return out


def print_normalization(norm):
    """Table of :func:`genie_normalization`: GENIE against the notebook."""
    print(f"{'flavor':>9} {'flux [/cm2/yr]':>15} {'<sigma> [cm2]':>13} "
          f"{'GENIE ev/yr':>12} {'MINT ev/yr':>11} {'ratio':>6} {'wt/event':>10}")
    for flavor in NU_FLAVORS:
        e = norm["flavors"].get(flavor)
        if e is None:
            continue
        mint = e["events_per_year_mint"]
        ratio = e["events_per_year_genie"] / mint if mint else float("nan")
        wt = e.get("weight_per_event", float("nan"))
        print(f"{flavor:>9} {e['flux_integral']:>15.4e} {e['sigma_avg_cm2']:>13.3e} "
              f"{e['events_per_year_genie']:>12.4e} {(mint or float('nan')):>11.4e} "
              f"{ratio:>6.3f} {wt:>10.3e}")


def print_summary(export):
    """Human-readable dump of what :func:`export_flux` wrote."""
    norm = export["normalization"]
    edges = export["energy_edges"]
    print(f"wrote {export['root_path']}")
    print(f"      {export['json_path']}")
    print(f"grid: {edges.size - 1} uniform bins, {edges[0]:g}-{edges[-1]:g} GeV "
          f"({edges[1] - edges[0]:g} GeV wide)\n")

    has_rates = any("events_per_year" in e for e in norm.values())
    header = f"{'flavor':>9} {'PDG':>4} {'flux [/cm2/yr]':>16} {'frac':>7}"
    if has_rates:
        header += f" {'events/yr':>12} {'wt/event':>11}"
    print(header)
    print("-" * len(header))
    for flavor in NU_FLAVORS:
        if flavor not in norm:
            continue
        e = norm[flavor]
        row = f"{flavor:>9} {e['pdg']:>4} {e['flux_integral']:>16.4e} {e['flux_fraction']:>7.4f}"
        if has_rates:
            row += f" {e.get('events_per_year', float('nan')):>12.4e}"
            row += f" {e.get('weight_per_event', float('nan')):>11.3e}"
        print(row)
