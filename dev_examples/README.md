# How MINT works

Four notebooks covering the machinery: the beam, the detector, the rates, and
the accelerator stages that feed them. Read these if you want to understand or
extend the simulation. For the physics results in the paper, see
[`../physics-examples/`](../physics-examples/).

Run them from this directory, so that the relative paths to `data/` and
`plots/` resolve:

```bash
cd dev_examples
jupyter lab
```

| Notebook | What it covers |
| --- | --- |
| [`beam_optics_plots.ipynb`](beam_optics_plots.ipynb) | Turning a MAD-X TWISS table into a beam: closed orbit, envelope and divergence through the low-beta IP, and the forward flux that results. Sections 5–10 are the checks worth running on any lattice you bring yourself, ending with an independent Σ-matrix model that carries the ⟨uu′⟩ correlation explicitly. |
| [`benchmark_detector.ipynb`](benchmark_detector.ipynb) | What `mint.detectors.benchmark` builds — aperture, module layout, composition, material budget — and the beamline between it and the IP. Also scans where the detector is best placed. The construction in section 2 is the template for describing your own detector. |
| [`collider_ring.ipynb`](collider_ring.ipynb) | Event rates at the detector: radial distributions, energy spectra for all four flavors, context against existing neutrino beams, and how the rate falls off between 250 m and 50 km. |
| [`low_energy_components.ipynb`](low_energy_components.ipynb) | Fluxes from the rest of the accelerator chain — pre-accelerator, both recirculating linacs, and the collider ring — on a common footing. |

`detector_diagnostics.py` holds the helpers that measure a detector against a
beam: how much of the flux the aperture contains, and what divergence the
observed spot implies. These live here rather than in `mint` because they
characterise a *pairing* of beam and detector rather than either one alone.

`data/` holds the digitized fluxes of existing and planned neutrino experiments,
used for the comparison in `collider_ring.ipynb`.
