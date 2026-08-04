"""Invariants that MINT results depend on.

These are cheap, deterministic checks of the properties the physics studies
lean on: that the beam normalization closes, that the optics are internally
consistent, that the geometry is self-consistent, and that the cross-section
backends agree with each other. Run them with::

    pytest tests/
"""

import numpy as np
import pytest

import mint
from mint import const


# ---------------------------------------------------------------------------
# Fixtures -- one lattice and one small beam sample shared by the whole module.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ring():
    return mint.lattices.load("mc_10tev_hybrid_v06")


@pytest.fixture(scope="module")
def beam(ring):
    """A small mu+ beam sample: (ring, sim, injections_per_year)."""
    return mint.beams.standard_beam("numubar", n_evals=5e4, ring=ring)


# ---------------------------------------------------------------------------
# Beam normalization
# ---------------------------------------------------------------------------

def test_muon_survival_is_included(beam):
    """Total decay weight must be N_mu x (1 - exp(-T/gamma tau)) x lattice share.

    Muons are sampled uniformly along the store, so the survival factor has to
    come in through the weights. If it silently went missing every rate in the
    package would be ~17% high.
    """
    ring, sim, _ = beam
    C, C_machine = float(ring.s(1)), ring.total_circumference
    turns = const.c_LIGHT / (ring.finj * C_machine)
    gamma = ring.beam_p0(0.0) / const.m_mu
    n_lambda = turns * C_machine / (gamma * const.c_LIGHT * const.tau0_mu)
    expected = ring.Nmu_per_bunch * (1 - np.exp(-n_lambda)) * C / C_machine
    # 1% covers the MC statistics of this small sample; a missing
    # survival factor would be a 17% error.
    assert sim.weights.sum() == pytest.approx(expected, rel=1e-2)


def test_muons_per_year_matches_machine_parameters(ring):
    ipy = mint.beams.injections_per_year(ring)
    assert ring.Nmu_per_bunch * ipy == pytest.approx(1.0e20, rel=0.05)


# ---------------------------------------------------------------------------
# Optics
# ---------------------------------------------------------------------------

def test_courant_snyder_envelopes(ring):
    """sigma^2 = eps beta, plus the dispersive term in the horizontal plane.

    The vertical plane of a flat ring is pure betatron, so it must hold to
    interpolation accuracy. The horizontal plane additionally carries
    (D sigma_delta)^2, making eps beta a lower bound.

    Both are checked away from the low-beta IP. There beta swings by orders of
    magnitude between adjacent Twiss rows, and sigma and beta -- separate
    interpolants built from the same grid -- stop agreeing pointwise. That
    limitation is real and documented; this test is about the other 99% of the
    ring, where the envelopes must be exactly consistent.
    """
    u = np.linspace(0, 1, 5000, endpoint=False)
    eps = ring.emittance_RMS
    betx, bety = ring.betx(u), ring.bety(u)
    away_from_ip = (betx > 1.0) & (bety > 1.0)          # beta in metres
    assert away_from_ip.mean() > 0.95, "IP mask should exclude only a sliver"

    sig_x = ring.beamsize_x(u) / const.m_to_cm
    assert np.all(sig_x[away_from_ip] ** 2
                  >= eps * betx[away_from_ip] * (1 - 2e-3))

    sig_y = ring.beamsize_y(u) / const.m_to_cm
    assert np.allclose(sig_y[away_from_ip] ** 2,
                       eps * bety[away_from_ip], rtol=2e-3)


def test_beta_is_positive_everywhere(ring):
    u = np.linspace(0, 1, 5000, endpoint=False)
    assert np.all(ring.betx(u) > 0) and np.all(ring.bety(u) > 0)


def test_correlation_matches_alpha(ring):
    """<u u'> = -eps alpha, the stored covariance."""
    u = np.linspace(0, 1, 500, endpoint=False)
    assert np.allclose(ring.cov_x_xp(u), -ring.emittance_RMS * ring.alfx(u), rtol=1e-6)


# ---------------------------------------------------------------------------
# Detector geometry
# ---------------------------------------------------------------------------

def test_default_detector_is_the_benchmark():
    assert mint.detectors.benchmark.name == mint.detectors.DEFAULT


def test_signal_column_is_the_quoted_value():
    """43 g/cm^2 in the vertex tracker + argon -- the number the rates use."""
    det = mint.detectors.benchmark
    g_per_cm2 = det.nucleon_column() * const.m_proton_in_g
    assert g_per_cm2 == pytest.approx(43.0, abs=1.0)


def test_aperture_grows_downstream():
    det = mint.detectors.benchmark
    assert det.radius_back > det.radius > 0


def test_column_is_additive_over_kinds():
    """The signal column may not exceed the whole-detector column."""
    det = mint.detectors.benchmark
    assert det.nucleon_column(kinds=None) >= det.nucleon_column()


@pytest.mark.parametrize("name", mint.detectors.available())
def test_every_registered_detector_builds(name):
    det = mint.detectors.load(name)
    assert det.nucleon_column() > 0


# ---------------------------------------------------------------------------
# Cross sections
# ---------------------------------------------------------------------------

def test_cc_cross_section_is_quasi_linear():
    """sigma_CC/E is slowly varying across the TeV range we use."""
    E = np.array([100.0, 1000.0, 5000.0])
    r = mint.xsecs.sigma_CC(E, "numu") / E
    assert np.all(r > 0)
    assert r.max() / r.min() < 3.0


def test_neutrino_exceeds_antineutrino_cc():
    E = np.geomspace(50, 5000, 12)
    assert np.all(mint.xsecs.sigma_CC(E, "numu") > mint.xsecs.sigma_CC(E, "numubar"))


def test_charm_is_a_sane_fraction_of_cc():
    E = np.geomspace(100, 5000, 10)
    for fl in ("numu", "numubar"):
        frac = mint.xsecs.sigma_charm_CC(E, fl) / mint.xsecs.sigma_CC(E, fl)
        assert np.all((frac > 0.01) & (frac < 0.30))


def test_backends_agree_within_modelling_spread():
    """The tabulated and CT18 backends should not disagree wildly."""
    E = np.geomspace(100, 3000, 6)
    try:
        mint.xsecs.use_backend("ct18")
        ct18 = mint.xsecs.sigma_CC(E, "numu")
    except Exception:
        pytest.skip("CT18 backend unavailable (needs the `parton` extra)")
    finally:
        mint.xsecs.use_backend("alfonso")
    tabulated = mint.xsecs.sigma_CC(E, "numu")
    assert np.all(np.abs(ct18 / tabulated - 1) < 0.5)


# ---------------------------------------------------------------------------
# Event generation
# ---------------------------------------------------------------------------

def test_reweighting_preserves_the_sample(beam):
    """A reweighted copy is the same phase-space points, different weights."""
    ring, sim, _ = beam
    other = sim.reweighted_copy(nuflavor="nue")
    assert len(other.weights.flatten()) == len(sim.weights.flatten())
    assert not np.allclose(other.weights_decay, sim.weights_decay)


def test_flux_is_a_subset_of_the_emitted_sample(beam):
    ring, sim, ipy = beam
    det = mint.detectors.benchmark
    E, w = det.face_flux(sim, exposure=ipy)
    assert 0 < w.sum() < sim.weights.sum() * ipy


def test_neutrino_energies_respect_the_beam_endpoint(beam):
    ring, sim, _ = beam
    E = np.asarray(sim.pnu["E"])
    assert E.min() >= 0
    assert E.max() < 1.05 * ring.beam_p0(0.0)
