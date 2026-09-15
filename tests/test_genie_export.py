"""GENIE flux export.

The export has to survive a round trip through a ROOT file and come back with
the same shape and normalization, because a silent factor here becomes a silent
factor in every event rate GENIE reports downstream.
"""

import json

import numpy as np
import pytest

from mint import genie_tools as gt

uproot = pytest.importorskip("uproot")


# A falling power law, the shape a resampler is most likely to distort.
COARSE_EDGES = np.geomspace(0.1, 100.0, 40)
COARSE_CENTERS = np.sqrt(COARSE_EDGES[1:] * COARSE_EDGES[:-1])
COARSE_FLUX = 1e9 * COARSE_CENTERS**-2.0


# ---------------------------------------------------------------------------
# Binning and resampling
# ---------------------------------------------------------------------------

def test_uniform_edges_are_uniform_and_cover_the_range():
    edges = gt.uniform_energy_edges(70.0, bin_width=0.1)
    assert edges[0] == 0.0
    assert edges[-1] == pytest.approx(70.0)
    assert np.allclose(np.diff(edges), 0.1)
    # e_max that is not a whole number of bins rounds up, never truncates
    edges = gt.uniform_energy_edges(63.05, bin_width=0.5)
    assert edges[-1] >= 63.05


def test_conserve_resampling_preserves_the_integral():
    fine = gt.uniform_energy_edges(100.0, bin_width=0.25)
    out = gt.resample_flux(COARSE_EDGES, COARSE_FLUX, fine, method="conserve")
    # The input starts at 0.1 GeV, so only the part of the fine grid that
    # overlaps it can carry flux; the integral over the whole grid must match.
    assert gt.flux_integral(fine, out) == pytest.approx(
        gt.flux_integral(COARSE_EDGES, COARSE_FLUX), rel=1e-12
    )


def test_conserve_resampling_is_exact_bin_by_bin_on_a_subdivision():
    # Split every input bin in two: each half must carry half the content.
    fine = np.sort(np.concatenate([COARSE_EDGES, COARSE_CENTERS]))
    out = gt.resample_flux(COARSE_EDGES, COARSE_FLUX, fine, method="conserve")
    for i in range(COARSE_FLUX.size):
        assert out[2 * i] == pytest.approx(COARSE_FLUX[i], rel=1e-10)
        assert out[2 * i + 1] == pytest.approx(COARSE_FLUX[i], rel=1e-10)


def test_loglog_resampling_reproduces_a_power_law():
    # A power law is a straight line in log-log, so the interpolant is exact and
    # each output bin must come back with the power law's mean over that bin,
    # 1/(ab) for E^-2 -- not its value at the center, which for a wide bin at
    # low energy is over 10% high.
    fine = gt.uniform_energy_edges(100.0, bin_width=0.25)
    out = gt.resample_flux(COARSE_EDGES, COARSE_FLUX, fine, method="loglog")
    lo, hi = fine[:-1], fine[1:]
    inside = (lo > COARSE_CENTERS.min()) & (hi < COARSE_CENTERS.max())
    assert np.allclose(out[inside], 1e9 / (lo[inside] * hi[inside]), rtol=1e-4)


def test_loglog_resampling_keeps_the_integral():
    # The whole point of averaging instead of point-sampling: an exported flux
    # has to still integrate to the rate the notebook computed. It is not exact
    # -- the interpolant is flat over the outermost half-bin, where this steep
    # test spectrum is not -- but it has to be sub-percent, and insensitive to
    # how fine the output grid is.
    exact = gt.flux_integral(COARSE_EDGES, COARSE_FLUX)
    for bin_width in (0.25, 1.0, 5.0):
        grid = gt.uniform_energy_edges(100.0, bin_width=bin_width)
        out = gt.resample_flux(COARSE_EDGES, COARSE_FLUX, grid, method="loglog")
        assert gt.flux_integral(grid, out) == pytest.approx(exact, rel=0.01)


def test_resampling_does_not_extrapolate_past_the_support():
    fine = gt.uniform_energy_edges(200.0, bin_width=1.0)
    for method in ("loglog", "conserve"):
        out = gt.resample_flux(COARSE_EDGES, COARSE_FLUX, fine, method=method)
        centers = 0.5 * (fine[1:] + fine[:-1])
        assert np.all(out[centers > COARSE_EDGES[-1]] == 0.0)


def test_loglog_bridges_interior_mc_zeros():
    flux = COARSE_FLUX.copy()
    flux[10] = 0.0            # an empty MC bin, not a real hole in the spectrum
    fine = gt.uniform_energy_edges(100.0, bin_width=0.25)
    out = gt.resample_flux(COARSE_EDGES, flux, fine, method="loglog")
    centers = 0.5 * (fine[1:] + fine[:-1])
    near = np.argmin(np.abs(centers - COARSE_CENTERS[10]))
    assert out[near] > 0.0


def test_resample_rejects_mismatched_input():
    with pytest.raises(ValueError):
        gt.resample_flux(COARSE_EDGES, COARSE_FLUX[:-1], COARSE_EDGES)
    with pytest.raises(ValueError):
        gt.resample_flux(COARSE_EDGES, COARSE_FLUX, COARSE_EDGES, method="spline")


# ---------------------------------------------------------------------------
# The ROOT file GENIE reads
# ---------------------------------------------------------------------------

def _demo_fluxes(edges):
    centers = 0.5 * (edges[1:] + edges[:-1])
    peak = np.exp(-0.5 * ((centers - 8.0) / 3.0) ** 2)
    return {"numubar": 1e10 * peak, "nue": 3e9 * peak, "nutaubar": 5e8 * peak}


def test_root_round_trip_keeps_bins_and_contents(tmp_path):
    edges = gt.uniform_energy_edges(30.0, bin_width=0.1)
    fluxes = _demo_fluxes(edges)
    path = tmp_path / "flux.root"

    names = gt.write_flux_root(path, edges, fluxes, metadata={"baseline_km": 1300})
    assert names == {f: f"flux_{f}" for f in fluxes}

    with uproot.open(path) as f:
        for flavor, flux in fluxes.items():
            hist = f[f"flux_{flavor}"]
            # GENIE C-casts the object to TH1D* without checking the type, so a
            # TH1F here would be read as garbage.
            assert hist.classname == "TH1D"
            assert np.allclose(hist.axis().edges(), edges)
            assert np.allclose(hist.values(), flux)
        assert json.loads(f["mint_metadata"])["baseline_km"] == 1300


def test_written_axis_is_uniform_so_genie_needs_no_width_correction(tmp_path):
    # ROOT's TH1::GetRandom treats bin content as a per-bin probability and
    # never divides by the width, so a variable-width axis would misrepresent a
    # density. Uniform bins make the two agree up to a constant.
    edges = gt.uniform_energy_edges(30.0, bin_width=0.1)
    path = tmp_path / "flux.root"
    gt.write_flux_root(path, edges, _demo_fluxes(edges))
    with uproot.open(path) as f:
        widths = np.diff(f["flux_numubar"].axis().edges())
    assert np.allclose(widths, widths[0])


def test_write_rejects_bad_input(tmp_path):
    edges = gt.uniform_energy_edges(10.0, bin_width=0.5)
    good = np.ones(edges.size - 1)
    with pytest.raises(ValueError, match="unknown neutrino flavors"):
        gt.write_flux_root(tmp_path / "a.root", edges, {"numubaz": good})
    with pytest.raises(ValueError, match="negative"):
        gt.write_flux_root(tmp_path / "b.root", edges, {"nue": -good})
    with pytest.raises(ValueError, match="against"):
        gt.write_flux_root(tmp_path / "c.root", edges, {"nue": good[:-1]})


def test_text_tables_hold_the_same_numbers(tmp_path):
    edges = gt.uniform_energy_edges(10.0, bin_width=0.5)
    fluxes = _demo_fluxes(edges)
    written = gt.write_flux_text(tmp_path / "flux", edges, fluxes)
    for flavor, path in written.items():
        table = np.loadtxt(path)
        assert np.allclose(table[:, 0], 0.5 * (edges[1:] + edges[:-1]))
        assert np.allclose(table[:, 1], fluxes[flavor], rtol=1e-7)


# ---------------------------------------------------------------------------
# Normalization bookkeeping
# ---------------------------------------------------------------------------

def test_normalization_summary_splits_the_flux_and_weights_events():
    edges = gt.uniform_energy_edges(30.0, bin_width=0.1)
    fluxes = _demo_fluxes(edges)
    rates = {flavor: 1e-5 * flux for flavor, flux in fluxes.items()}

    summary = gt.normalization_summary(edges, fluxes, event_rates=rates, n_events=1000)

    assert sum(e["flux_fraction"] for e in summary.values()) == pytest.approx(1.0)
    for flavor, entry in summary.items():
        assert entry["pdg"] == gt.PDG_CODES[flavor]
        assert entry["flux_integral"] == pytest.approx(
            gt.flux_integral(edges, fluxes[flavor])
        )
        # a GENIE sample of n_events for this flavor stands for a year of running
        assert entry["weight_per_event"] * 1000 == pytest.approx(
            entry["events_per_year"]
        )


def test_normalization_accepts_per_flavor_statistics():
    edges = gt.uniform_energy_edges(30.0, bin_width=0.1)
    fluxes = _demo_fluxes(edges)
    rates = {flavor: 1e-5 * flux for flavor, flux in fluxes.items()}
    n = {"numubar": 500, "nue": 200, "nutaubar": 100}
    summary = gt.normalization_summary(edges, fluxes, event_rates=rates, n_events=n)
    for flavor, entry in summary.items():
        assert entry["n_events"] == n[flavor]
        assert entry["weight_per_event"] == pytest.approx(
            entry["events_per_year"] / n[flavor]
        )


# ---------------------------------------------------------------------------
# Command lines
# ---------------------------------------------------------------------------

def test_gevgen_energy_range_covers_the_whole_histogram():
    # GENIE zeroes any bin whose upper edge compares greater than emax, which
    # float edges can do to the last bin, so the range is padded by one bin.
    edges = gt.uniform_energy_edges(70.0, bin_width=0.1)
    cmds = gt.gevgen_commands("/data/flux.root", {"numubar": np.ones(edges.size - 1)}, edges)
    cmd = cmds["numubar"]
    lo, hi = (float(x) for x in cmd.split("-e ")[1].split()[0].split(","))
    assert lo <= edges[0]
    assert hi > edges[-1]


def test_gevgen_commands_are_one_flavor_each():
    edges = gt.uniform_energy_edges(10.0, bin_width=0.5)
    fluxes = _demo_fluxes(edges)
    cmds = gt.gevgen_commands("/data/flux.root", fluxes, edges, n_events=4242)
    assert set(cmds) == set(fluxes)
    for flavor, cmd in cmds.items():
        assert f"-p {gt.PDG_CODES[flavor]}" in cmd
        assert f"/data/flux.root,flux_{flavor}" in cmd
        assert f"-t {gt.AR40}" in cmd
        assert "-n 4242" in cmd


def test_gevgen_t2k_command_lists_every_species():
    edges = gt.uniform_energy_edges(10.0, bin_width=0.5)
    fluxes = _demo_fluxes(edges)
    cmd = gt.gevgen_t2k_command("/data/flux.root", fluxes)
    for flavor in fluxes:
        assert f"{gt.PDG_CODES[flavor]}[flux_{flavor}]" in cmd
    assert f"-g {gt.AR40}" in cmd


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def test_export_flux_writes_everything_and_conserves_the_rate(tmp_path):
    fluxes = {"numubar": COARSE_FLUX, "nue": 0.3 * COARSE_FLUX}
    rates = {flavor: 1e-8 * flux for flavor, flux in fluxes.items()}

    out = gt.export_flux(
        tmp_path / "rla2_dune_flux.root",
        COARSE_EDGES,
        fluxes,
        event_rates=rates,
        bin_width=0.25,
        e_max=100.0,
        n_events=5000,
        metadata={"machine": "RLA2", "baseline_km": 1300},
        write_text=True,
    )

    assert (tmp_path / "rla2_dune_flux.root").exists()
    assert (tmp_path / "rla2_dune_flux.json").exists()
    assert len(out["text_paths"]) == 2

    meta = json.loads((tmp_path / "rla2_dune_flux.json").read_text())
    assert meta["machine"] == "RLA2"
    assert meta["resampling"]["output_bins"] == 400

    # Resampling has to leave the rate the notebook computed intact, so the
    # exported flux still means events per year.
    for flavor in fluxes:
        entry = out["normalization"][flavor]
        assert entry["events_per_year"] == pytest.approx(
            entry["events_per_year_input_grid"], rel=0.01
        )

    with uproot.open(out["root_path"]) as f:
        assert set(f.keys(cycle=False)) == {"flux_numubar", "flux_nue", "mint_metadata"}


def test_export_flux_leaves_a_matching_grid_untouched(tmp_path):
    # The notebook builds the oscillated spectra directly on the GENIE grid, so
    # exporting must not re-interpolate them.
    edges = gt.uniform_energy_edges(30.0, bin_width=0.1)
    fluxes = _demo_fluxes(edges)
    out = gt.export_flux(tmp_path / "flux.root", edges, fluxes, new_edges=edges)
    for flavor, flux in fluxes.items():
        assert np.array_equal(out["fluxes"][flavor], flux)
    assert out["metadata"]["resampling"]["method"].startswith("none")
