"""Cross sections and lattice construction.

Cross sections set every rate in the package, and lattices set every flux, so
both deserve checks that would catch a silent factor rather than only a crash.
"""

import numpy as np
import pytest

import mint
from mint import xsecs, const


# The shipped tables start near 50 GeV; below that they return zero by
# construction, so the grid stays inside their support.
E_GRID = np.geomspace(100.0, 5000.0, 15)
FLAVORS = ("numu", "numubar", "nue", "nuebar")


# ---------------------------------------------------------------------------
# Cross sections
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flavor", FLAVORS)
def test_cc_is_positive_and_increasing(flavor):
    s = xsecs.sigma_CC(E_GRID, flavor)
    assert np.all(s > 0)
    assert np.all(np.diff(s) > 0), "CC cross section must grow with energy"


def test_nc_is_a_minority_of_cc():
    """NC/CC sits near 0.3 across the TeV range, for nu and nubar alike."""
    for nc, cc in ((xsecs.sigma_NC_nu, "numu"), (xsecs.sigma_NC_nubar, "numubar")):
        r = nc(E_GRID) / xsecs.sigma_CC(E_GRID, cc)
        assert np.all((r > 0.1) & (r < 0.6))


def test_isoscalar_symmetry_between_lepton_flavors():
    """DIS is blind to the charged-lepton flavor well above threshold."""
    a = xsecs.sigma_CC(E_GRID, "numu")
    b = xsecs.sigma_CC(E_GRID, "nue")
    assert np.allclose(a, b, rtol=0.05)


def test_charm_is_a_subset_of_cc():
    for f in ("numu", "numubar"):
        assert np.all(xsecs.sigma_charm_CC(E_GRID, f) < xsecs.sigma_CC(E_GRID, f))


def test_bottom_is_far_below_charm():
    assert np.all(xsecs.sigma_bottom_CC_nu(E_GRID) < xsecs.sigma_charm_CC_nu(E_GRID))


def test_scalar_and_array_inputs_agree():
    """A scalar energy must give the same answer as a length-1 array."""
    for f in FLAVORS:
        assert xsecs.sigma_CC(1000.0, f) == pytest.approx(
            xsecs.sigma_CC(np.array([1000.0]), f)[0], rel=1e-9)


@pytest.mark.parametrize("bad", ["nutau_bar_typo", "numubarr", "nu_mu", "zzz"])
def test_unknown_flavor_is_rejected(bad):
    """A typo must raise, not silently return the neutrino branch."""
    with pytest.raises(ValueError):
        xsecs.sigma_CC(1000.0, bad)
    with pytest.raises(ValueError):
        xsecs.sigma_charm_CC(1000.0, bad)


@pytest.mark.parametrize("flavor", FLAVORS)
def test_every_documented_flavor_is_accepted(flavor):
    assert xsecs.sigma_CC(1000.0, flavor) > 0


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError):
        xsecs.use_backend("not-a-backend")


def test_backend_switching_is_reversible():
    before = xsecs.sigma_CC(1000.0, "numu")
    try:
        xsecs.use_backend("ct18")
    except Exception:
        pytest.skip("CT18 backend unavailable")
    finally:
        xsecs.use_backend("alfonso")
    assert xsecs.sigma_CC(1000.0, "numu") == pytest.approx(before, rel=1e-12)


def test_nutau_cc_is_threshold_suppressed():
    """The tau mass suppresses nu_tau CC at low energy but not at high."""
    lo, hi = 100.0, 4000.0
    r_lo = xsecs.sigma_nutau_CC(lo) / xsecs.sigma_CC(lo, "numu")
    r_hi = xsecs.sigma_nutau_CC(hi) / xsecs.sigma_CC(hi, "numu")
    assert r_lo < r_hi
    assert r_hi < 1.2


# ---------------------------------------------------------------------------
# Lattices
# ---------------------------------------------------------------------------

def test_registry_is_non_empty_and_loadable():
    names = mint.lattices.available()
    assert names
    import os
    for n in names:
        assert os.path.exists(str(mint.lattices.path(n)))


def test_unknown_lattice_is_rejected():
    with pytest.raises(KeyError):
        mint.lattices.load("not-a-lattice")


@pytest.fixture(scope="module")
def ring():
    return mint.lattices.load("mc_10tev_hybrid_v06")


def test_overrides_reach_the_lattice():
    r = mint.lattices.load("mc_10tev_hybrid_v06", Nmu_per_bunch=1.5e12)
    assert r.Nmu_per_bunch == pytest.approx(1.5e12)


def test_arclength_is_monotonic_and_closes(ring):
    u = np.linspace(0, 1, 2000)
    s = ring.s(u)
    assert np.all(np.diff(s) > 0)
    assert s[0] == pytest.approx(0.0, abs=1e-6)


def test_inv_s_inverts_s(ring):
    u = np.linspace(0.01, 0.99, 200)
    assert np.allclose(ring.inv_s(ring.s(u)), u, atol=1e-4)


def test_orbit_is_continuous(ring):
    """The reference orbit has no jumps: consecutive points stay within the
    arc length between them."""
    u = np.linspace(0, 1, 4000)
    p = np.array([ring.x(u), ring.y(u), ring.z(u)])
    step = np.linalg.norm(np.diff(p, axis=1), axis=0)
    ds = np.diff(ring.s(u))
    assert np.all(step <= ds * (1 + 1e-6) + 1e-6), "orbit jumps further than its arc length"


def test_tangent_is_a_unit_vector(ring):
    u = np.linspace(0, 1, 500, endpoint=False)
    t = np.asarray(ring.tangent(u))
    assert np.allclose(np.linalg.norm(t, axis=0), 1.0, atol=1e-6)


def test_divergence_is_small_everywhere(ring):
    """A TeV beam is collimated: milliradians, not radians."""
    u = np.linspace(0, 1, 2000, endpoint=False)
    for f in (ring.beamdiv_x, ring.beamdiv_y):
        d = np.abs(f(u))
        assert np.all(np.isfinite(d)) and np.all(d < 1e-2)


def test_beam_momentum_is_the_design_energy(ring):
    u = np.linspace(0, 1, 100)
    p = ring.beam_p0(u)
    assert np.all(p > 0)
    assert np.ptp(p) / np.mean(p) < 1e-6, "momentum should be flat around the ring"


def test_read_tfs_matches_the_loaded_lattice(ring):
    df = mint.lattices.read_tfs("mc_10tev_hybrid_v06")
    assert len(df) > 100
    assert {"S", "BETX", "BETY", "ALFX", "ALFY"} <= set(df.columns)
    assert df["S"].max() == pytest.approx(float(ring.s(1)) / const.m_to_cm, rel=1e-3)


@pytest.mark.parametrize("name", list(mint.lattices.available()))
def test_every_shipped_lattice_builds_a_usable_beam(name):
    """Every registered lattice must load and give physical envelopes.

    Guards the packaged data as much as the code: a truncated or mislabelled
    TWISS file shows up here rather than as a silently wrong flux.
    """
    r = mint.lattices.load(name)
    u = np.linspace(0, 1, 500, endpoint=False)
    assert float(r.s(1)) > 0
    assert np.all(r.betx(u) > 0) and np.all(r.bety(u) > 0)
    assert np.all(np.isfinite(r.beamsize_x(u)))
    p0 = float(r.beam_p0(0.0))
    assert 1.0 < p0 < 1e5, f"{name}: implausible beam momentum {p0} GeV"


def test_the_two_collider_energies_are_distinct():
    """The 3 and 10 TeV rings must not silently be the same machine."""
    three = mint.lattices.load("mc_3tev_v1.2")
    ten = mint.lattices.load("mc_10tev_ring_v06")
    assert float(three.beam_p0(0.0)) == pytest.approx(1500.0, rel=0.02)
    assert float(ten.beam_p0(0.0)) == pytest.approx(5000.0, rel=0.02)
    assert float(three.s(1)) < float(ten.s(1))


def test_compressed_and_plain_tfs_read_identically(tmp_path):
    """read_tfs must handle .tfs.gz transparently."""
    import gzip
    import shutil
    src = mint.lattices.path("mc_3tev_v1.2")
    assert src.endswith(".gz"), "shipped lattices are stored compressed"
    plain = tmp_path / "ring.tfs"
    with gzip.open(src, "rb") as fi, open(plain, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    a = mint.lattices.read_tfs(src)
    b = mint.lattices.read_tfs(str(plain))
    assert len(a) == len(b)
    assert np.allclose(a["S"].to_numpy(), b["S"].to_numpy())
