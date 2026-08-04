"""One-call setup for standard MINT physics studies.

The physics examples all start from the same place: a lattice, a muon-decay
simulation per neutrino flavor (with exact reweighting between flavors of the
same beam), and the annual exposure normalization. This module provides that
in one call::

    import mint

    # single beam / flavor
    ring, sim, ipy = mint.examples.standard_beam("numubar")

    # the full two-beam setup used by the examples: mu+ beam (clockwise:
    # numubar + nue) and mu- beam (counter-clockwise: numu + nuebar)
    ring, sims, ipy = mint.examples.both_beams()
    sims["numubar"]  # mu+ beam, points at the downstream (+z) detector
    sims["numu"]     # mu- beam, points at the upstream (-z) detector

``ipy`` is the number of injections per year; ``sim.weights * ipy`` are
annual rates. Need more statistics? Pass ``n_evals`` (the vegas sample size)
-- everything is generated on the fly, one vegas evaluation per call, with
all flavors of both beams obtained from it by exact reweighting.
"""

import numpy as np

from mint import const
from mint import lattices
from mint.MuC import MuDecaySimulator

# CP mirror: which flavors belong to which beam / circulation direction
MU_PLUS_FLAVORS = ("numubar", "nue")        # mu+ -> e+ nu_e numubar, clockwise
MU_MINUS_FLAVORS = ("numu", "nuebar")       # mu- -> e- nuebar nu_mu, counter-clockwise

DEFAULT_LATTICE = "mc_10tev_hybrid_v06"


def injections_per_year(ring):
    """Annual exposure normalization: injections per year of operation."""
    return ring.finj * ring.bunch_multiplicity * ring.duty_factor * const.s_in_year


def standard_beam(nuflavor="numubar", direction="clockwise",
                  lattice=DEFAULT_LATTICE, n_evals=5e5, beam_dynamics=True,
                  muon_polarization=0.0, cycles="full_injection", ring=None):
    """One beam, one flavor. Returns (ring, sim, injections_per_year)."""
    if ring is None:
        ring = lattices.load(lattice)
    sim = MuDecaySimulator(muon_polarization=muon_polarization, lattice=ring,
                           nuflavor=nuflavor, cycles=cycles, n_evals=n_evals,
                           beam_dynamics=beam_dynamics)
    sim.decay_muons()
    sim.place_muons_on_lattice(lattice=ring, direction=direction)
    return ring, sim, injections_per_year(ring)


def both_beams(flavors=MU_PLUS_FLAVORS + MU_MINUS_FLAVORS,
               lattice=DEFAULT_LATTICE, n_evals=5e5, beam_dynamics=True,
               muon_polarization=0.0, cycles="full_injection"):
    """The standard two-beam configuration with identical detectors.

    One vegas evaluation; every requested flavor is an exact reweighting of
    it, placed clockwise (mu+ beam flavors) or counter-clockwise (mu- beam
    flavors). Returns (ring, {flavor: sim}, injections_per_year).
    """
    base_flavor = flavors[0]
    base_dir = "clockwise" if base_flavor in MU_PLUS_FLAVORS else "counter-clockwise"
    ring, base, ipy = standard_beam(base_flavor, direction=base_dir,
                                    lattice=lattice, n_evals=n_evals,
                                    beam_dynamics=beam_dynamics,
                                    muon_polarization=muon_polarization,
                                    cycles=cycles)
    sims = {base_flavor: base}
    for fl in flavors[1:]:
        s = base.reweighted_copy(nuflavor=fl)
        d = "clockwise" if fl in MU_PLUS_FLAVORS else "counter-clockwise"
        s.place_muons_on_lattice(lattice=ring, direction=d)
        sims[fl] = s
    return ring, sims, ipy
