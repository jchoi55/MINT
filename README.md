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

**Interaction vertices in a detector.** Ray-trace the placed neutrinos through
a detector and generate weighted interaction vertices, with exponential
attenuation along each chord and upstream shielding included:

```python
det = mint.detectors.benchmark                     # the benchmark forward detector
rates = det.signal_interactions(sim, nuflavor="numubar", exposure=ipy)
print(rates["total"])                              # interactions/year in the signal volume
```

A detector is a stack of coaxial material volumes, so you can build your own
geometry by composing `mint.detector_tools` volumes with any `Material`. See
`dev_examples/benchmark_detector.ipynb` for a worked construction.

**Partial lattices.** If a TFS file covers only part of a machine — an
interaction region of a larger ring, say — pass the full machine length. Decays
are placed on the covered section while the muons age and decay over the whole
ring:

```python
ring = mint.lattices.load("mc_10tev_hybrid_v06", total_circumference=10e5)  # cm
```

## Repository layout

| Path | Contents |
| --- | --- |
| `mint/` | The Python package |
| `mint/MuC.py` | `MuDecaySimulator` — muon decays along a lattice, and the flux they produce |
| `mint/lattices.py` | Entry point for lattices: the registry of shipped optics, `load()` and `from_tfs()` |
| `mint/lattice_tools.py` | The `Lattice` class, Twiss smoothing, and parametric geometries |
| `mint/detectors.py` | The `Detector` class and the `benchmark` instance |
| `mint/detector_tools.py` | Materials and volumes to build detectors from |
| `mint/beamline.py` | Shielding and material budget between the IP and the detector |
| `mint/mudecay_tools.py` | Polarized (N)LO muon-decay matrix elements and the vegas generator |
| `mint/xsecs.py` | Neutrino cross sections (DIS, elastic, tridents, resonances) |
| `mint/lattice_data/` | MAD-X TWISS files shipped with the package |
| `dev_examples/` | How the simulation works — beam optics, detector, rates, accelerator chain |
| `physics-examples/` | The physics studies behind the paper |
| `tests/` | The invariants the results depend on (`pytest tests/`) |

## Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

These check the properties the physics leans on: that the beam normalization
closes including muon survival in the store, that the Courant–Snyder envelopes
are self-consistent, that the detector column densities are what the rates
assume, and that the cross-section backends agree.

## Citation

If you use MINT, please cite the accompanying paper. See `physics-examples/` for
the studies it reports.

## Building distributions

```bash
python -m build
```

Artifacts land in `dist/`. The packaged data — cross-section tables and the
reference lattices in `mint/lattice_data/` — ships inside the wheel, so an
installed user can run flux simulations without cloning the repository.
