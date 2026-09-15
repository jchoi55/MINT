"""Driving GENIE from Python.

GENIE itself is not available on CI, so the runner is exercised against fake
``gevgen``/``gntpc`` scripts: what matters here is the environment, the
directory layout, the bounded log, and the arithmetic that turns a shape-only
sample back into a rate.
"""

import os
import stat
import textwrap

import numpy as np
import pytest

from mint import genie_tools as gt

uproot = pytest.importorskip("uproot")


@pytest.fixture
def fake_genie(tmp_path):
    """A GENIE tree whose gevgen prints a lot and writes its -o file."""
    genie = tmp_path / "Generator"
    (genie / "bin").mkdir(parents=True)
    (genie / "config").mkdir()
    (genie / "config" / "Messenger_laconic.xml").write_text("<x/>")
    gevgen = genie / "bin" / "gevgen"
    gevgen.write_text(textwrap.dedent("""\
        #!/bin/sh
        # record the arguments and the environment the job saw
        echo "$@" > argv.txt
        echo "$GENIE|$GXMLPATH|$DYLD_LIBRARY_PATH" > env.txt
        i=0; while [ $i -lt 2000 ]; do echo "chatter line $i"; i=$((i+1)); done
        # -o is the last pair
        out=""; prev=""; for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done
        echo "ghep" > "$out"
        echo "input-flux" > input-flux.root
        echo "done"
        """))
    gntpc = genie / "bin" / "gntpc"
    gntpc.write_text(textwrap.dedent("""\
        #!/bin/sh
        out=""; prev=""; for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done
        echo "gst" > "$out"
        """))
    for f in (gevgen, gntpc):
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    splines = tmp_path / "xsec" / "gxspl.xml"
    splines.parent.mkdir()
    splines.write_text("<genie_xsec_spline_list/>")
    return gt.GenieInstall(genie, splines, "G18_10a_02_11a",
                           extra_lib_dirs=[tmp_path / "libs"])


@pytest.fixture
def flux_file(tmp_path):
    edges = gt.uniform_energy_edges(30.0, bin_width=0.1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    peak = np.exp(-0.5 * ((centers - 8.0) / 3.0) ** 2)
    fluxes = {"numubar": 1e10 * peak, "nue": 3e9 * peak}
    path = tmp_path / "flux.root"
    gt.export_flux(path, edges, fluxes, new_edges=edges,
                   metadata={"target_nucleons": 4.0e36})
    return path, edges, fluxes


# ---------------------------------------------------------------------------
# Install / environment
# ---------------------------------------------------------------------------

def test_install_builds_the_environment_and_finds_binaries(fake_genie):
    fake_genie.check()
    env = fake_genie.env()
    assert env["GENIE"] == fake_genie.genie_dir
    assert env["GXMLPATH"] == os.path.dirname(fake_genie.spline_xml)
    assert env["PATH"].startswith(os.path.join(fake_genie.genie_dir, "bin"))
    # user library dirs come first, GENIE's own lib after them
    libs = env["DYLD_LIBRARY_PATH"].split(os.pathsep)
    assert libs[0].endswith("libs")
    assert libs[1] == os.path.join(fake_genie.genie_dir, "lib")


def test_install_check_names_what_is_missing(tmp_path):
    install = gt.GenieInstall(tmp_path / "nowhere", tmp_path / "no.xml", "G18_10a_02_11a")
    with pytest.raises(FileNotFoundError, match="gevgen"):
        install.check()


def test_tune_and_splines_are_passed_through(fake_genie, flux_file, tmp_path):
    path, edges, _ = flux_file
    gt.run_gevgen(fake_genie, path, "numubar", 50, tmp_path / "runs")
    argv = (tmp_path / "runs" / "numubar" / "argv.txt").read_text().split()
    assert argv[argv.index("--tune") + 1] == "G18_10a_02_11a"
    assert argv[argv.index("--cross-sections") + 1] == fake_genie.spline_xml
    assert argv[argv.index("-p") + 1] == "-14"
    assert argv[argv.index("-n") + 1] == "50"
    lo, hi = (float(x) for x in argv[argv.index("-e") + 1].split(","))
    assert lo == 0.0 and hi > edges[-1]
    assert (tmp_path / "runs" / "numubar" / "env.txt").read_text().startswith(fake_genie.genie_dir)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def test_each_flavor_gets_its_own_directory_and_a_bounded_log(fake_genie, flux_file, tmp_path):
    path, _, _ = flux_file
    results = gt.run_gevgen_all(fake_genie, path, 10, tmp_path / "runs",
                                flavors=("numubar", "nue"), tail_lines=100)
    assert set(results) == {"numubar", "nue"}
    for flavor, r in results.items():
        assert r["returncode"] == 0 and r["gntpc_returncode"] == 0
        assert os.path.dirname(r["ghep"]).endswith(flavor)
        assert os.path.exists(r["gst"])
        # gevgen's stray input-flux.root landed in the job's own directory
        assert (tmp_path / "runs" / flavor / "input-flux.root").exists()
        log = open(r["log"]).read().splitlines()
        assert len(log) <= 102                       # header + tail only
        assert "lines dropped" in log[1]
        assert log[-1] == "done"                     # the end is what survives


def test_parallel_jobs_get_distinct_runs_and_seeds(fake_genie, flux_file, tmp_path):
    path, _, _ = flux_file
    gt.run_gevgen_all(fake_genie, path, 10, tmp_path / "runs",
                      flavors=("numubar", "nue"), seed=100)
    seen = {}
    for flavor in ("numubar", "nue"):
        argv = (tmp_path / "runs" / flavor / "argv.txt").read_text().split()
        seen[flavor] = (argv[argv.index("-r") + 1], argv[argv.index("--seed") + 1])
    assert seen["numubar"] != seen["nue"]
    assert {s for _, s in seen.values()} == {"100", "101"}


def test_failed_job_is_reported_not_raised(fake_genie, flux_file, tmp_path):
    path, _, _ = flux_file
    bad = fake_genie.binary("gevgen")
    with open(bad, "w") as fh:
        fh.write("#!/bin/sh\necho boom\nexit 3\n")
    r = gt.run_gevgen(fake_genie, path, "nue", 10, tmp_path / "runs")
    assert r["returncode"] == 3
    assert r["gst"] is None and r["gntpc_returncode"] is None
    assert "boom" in open(r["log"]).read()


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------

def test_interaction_spectrum_is_flux_times_sigma_times_targets():
    edges = gt.uniform_energy_edges(10.0, bin_width=1.0)
    flux = np.ones(10)
    xsec_E = np.array([0.0, 100.0])
    xsec = 1e-38 * np.array([0.0, 100.0])          # sigma = 1e-38 E
    dNdE = gt.interaction_spectrum(edges, flux, xsec_E, xsec, n_nuclei=1e36)
    centers = 0.5 * (edges[1:] + edges[:-1])
    assert np.allclose(dNdE, 1e36 * 1.0 * 1e-38 * centers)


def test_sigma_below_the_table_is_zero_not_extrapolated():
    # bin centers 0.25, 0.75, 1.25, 1.75 GeV against a table starting at 1 GeV
    edges = gt.uniform_energy_edges(2.0, bin_width=0.5)
    dNdE = gt.interaction_spectrum(edges, np.ones(4), np.array([1.0, 2.0]),
                                   np.array([1e-38, 2e-38]), n_nuclei=1.0)
    assert dNdE[0] == 0.0 and dNdE[1] == 0.0
    assert dNdE[2] > 0.0 and dNdE[3] > dNdE[2]


def test_energy_range_pads_the_top_bin():
    edges = gt.uniform_energy_edges(70.0, bin_width=0.1)
    lo, hi = gt.energy_range(edges)
    assert lo == 0.0
    assert hi == pytest.approx(70.1)
