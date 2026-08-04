"""MINT -- the Muon Induced Neutrino Tool.

Neutrino fluxes at muon facilities: muon colliders, neutrino factories, and
the accelerator stages that feed them. MINT decays muons along a realistic
accelerator lattice and propagates the neutrinos to a detector, keeping the
beam optics, muon polarization, and NLO decay corrections.

A first look::

    import mint

    ring, sim, ipy = mint.examples.standard_beam("numubar")
    det = mint.detectors.benchmark
    E, w = det.face_flux(sim, exposure=ipy)

The pieces you are most likely to touch:

===================  =======================================================
``mint.lattices``    load a shipped ring, or build one from your MAD-X TFS
``mint.detectors``   the benchmark forward detector, or your own geometry
``mint.examples``    one-call setups for the standard beam configurations
``mint.MuC``         :class:`MuDecaySimulator`, the event generator itself
``mint.xsecs``       neutrino cross sections (switchable backends)
``mint.plot_tools``  figure styling used by the example notebooks
===================  =======================================================
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mint-muc")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "unknown"

from mint import const
from mint import mudecay_tools
from mint import detector_tools
from mint import lattice_tools
from mint import lattices
from mint import xsecs
from mint import MuC
from mint import detectors
from mint import beamline
from mint import examples
from mint import plot_tools

# The two classes most users construct directly.
from mint.lattice_tools import Lattice
from mint.MuC import MuDecaySimulator

__all__ = [
    "__version__",
    "const",
    "mudecay_tools",
    "detector_tools",
    "lattice_tools",
    "lattices",
    "xsecs",
    "MuC",
    "detectors",
    "beamline",
    "examples",
    "plot_tools",
    "Lattice",
    "MuDecaySimulator",
]
