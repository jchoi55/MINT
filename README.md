# Muon Induced Neutrino Tool (MINT)

<img src="assets/mint-logo.svg" alt="MINT logo" width="120" />

Neutrino fluxes at muon facilities — muon colliders, neutrino factories, and
muon accelerator stages. MINT decays muons along realistic accelerator lattices
(built from MAD-X TWISS/TFS tables or simple parametric geometries), including
polarization, NLO radiative corrections to the decay, and beam optics
(envelope, divergence, momentum spread), and propagates the neutrinos to
arbitrary detector locations.

## Installation

```bash
pip install .            # from the repository root
pip install -e .[dev]    # editable install for development
```

## Quickstart

```python
import numpy as np
import mint

# 1. Load a collider-ring lattice shipped with MINT
print(mint.lattices.available())
ring = mint.lattices.load("mc_10tev_hybrid_v06")

# ... or build one from your own MAD-X TWISS/TFS file
# ring = mint.lattices.from_tfs("my_ring.tfs", emittance_RMS=5.25e-10, Nmu_per_bunch=2e12)

# 2. Decay muons along the ring (mu+ -> e+ nu_e nu_mu-bar)
sim = mint.MuDecaySimulator(
    muon_polarization=0.0,
    lattice=ring,
    nuflavor="numubar",
    n_evals=1e5,
    beam_dynamics=True,
)
sim.decay_muons()
sim.place_muons_on_lattice(lattice=ring, direction="clockwise")

# 3. Neutrino flux through a detector face 1 km downstream of the IP
E, flux = sim.get_flux_at_generic_location(
    det_location=[0, 0, 1e5],  # cm
    det_radius=2e2,            # cm
    ebins=np.linspace(0, 5e3, 31),
)
```

**Run vegas once, reuse forever.** The event generation only samples the
rest-frame decay phase space; flavor, polarization, and NLO are exact
matrix-element reweights of the same sample:

```python
sim_nue = sim.reweighted_copy(nuflavor="nue")      # no new vegas run
sim_nue.place_muons_on_lattice(lattice=ring, direction="clockwise")

sim.save_events("mudecays.npz")                    # persist the sample ...
sim2 = mint.MuDecaySimulator.load_events("mudecays.npz", lattice=ring)  # ... reuse later
```

**Interaction vertices in a detector.** Ray-trace the placed neutrinos through a
detector geometry and generate weighted interaction vertices (exponential
attenuation along each chord, upstream-volume shielding included):

```python
from mint import detector_tools as dt

detector = dt.uniform_hydrogen_cylinder(distance_cm=1e5)   # 1 g/cm^3 H, 10 m x 2 m diameter
vertices = detector.generate_interactions(sim, exposure=bunches_per_year)
vertices[["x", "y", "z", "E", "w"]]                        # weighted vertex sample (DataFrame)
```

Detectors are lists of material volumes — compose `dt.CylinderVolume`s with any
`dt.Material` to build segmented geometries.

**Partial lattices.** If a TFS file covers only part of a machine (e.g. an
interaction region of a larger ring), pass the full machine length — decays are
placed on the covered section while muons age and decay over the whole ring:

```python
ring = mint.lattices.load("mc_10tev_hybrid_v06", total_circumference=10e5)  # cm
```

See `main_collider_ring_studies.ipynb` for the standard validation plots
(geometry, beam optics, forward flux, event rates).

## Repository layout

| Path | Contents |
| --- | --- |
| `mint/` | The Python package |
| `mint/MuC.py` | `MuDecaySimulator` — muon decays along a lattice, flux at detectors |
| `mint/lattices.py` | **User entry point for lattices**: registry of shipped optics, `load()` / `from_tfs()` |
| `mint/lattice_tools.py` | `Lattice` class and parametric geometries (racetrack, RLA, straight, ...) |
| `mint/beam_optics.py` | Twiss smoothing: TFS table → beam envelopes/divergences/dispersion |
| `mint/mudecay_tools.py` | Polarized (N)LO muon-decay matrix elements and vegas generator |
| `mint/xsecs.py` | Neutrino cross sections (DIS, ES, tridents, resonant channels) |
| `mint/detector_tools.py` | Materials, detector geometries (`CylinderVolume`, `Detector`), and neutrino interaction-vertex generation |
| `mint/const.py`, `mint/collider_tools.py`, `mint/plot_tools.py` | Constants, collider parameter sets, plotting style |
| `mint/lattice_data/` | MAD-X TWISS files shipped with the package |
| `main_collider_ring_studies.ipynb` | Collider-ring validation notebook |
| `main_RLA_studies.ipynb` | Low-energy accelerator (RLA) studies |
| `physics-examples/` | Physics case studies built on top of MINT (HNL sensitivity, luminosity). These may require extra packages (e.g. [DarkNews](https://github.com/LBL-Neutrino-Physics/DarkNews-generator)), which are **not** dependencies of MINT |
| `beam-optics/` | Large development TFS files and derived pickles (not shipped with the package) |
| `dev_examples/` | Development / extended-study notebooks |

## Building distributions

```bash
python -m build
```

Artifacts are placed in `dist/`. The packaged data (cross-section tables and
the reference lattices in `mint/lattice_data/`) ships inside the wheel, so
installed users can run flux simulations without cloning the repository.
