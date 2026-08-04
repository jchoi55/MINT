# Physics studies

The studies behind the MINT forward-neutrino paper. Nothing here is part of the
`mint` package — these notebooks import MINT, and a few external tools, as
libraries. If you want to understand the simulation itself rather than its
results, start with [`../dev_examples/`](../dev_examples/).

Run them from this directory:

```bash
cd physics-examples
jupyter lab
```

Every notebook opens the same way:

```python
import mint

ring, sims, ipy = mint.examples.both_beams()   # one vegas sample, all four flavors
det = mint.detectors.benchmark                 # the benchmark forward detector
E, w = det.face_flux(sims["numubar"], exposure=ipy)
```

`mint.examples.both_beams` generates the muon-decay simulation once and derives
every flavor from it by exact reweighting, placing the μ⁺ beam clockwise and the
μ⁻ beam counter-clockwise. `mint.detectors.benchmark` is the detector described
in `../dev_examples/benchmark_detector.ipynb`: a conical aperture opening from
1.3 m to 2.4 m at 5 km from the interaction point, whose **signal volume** — the
vertex tracker plus the argon TPC, 3.2 t and 43 g/cm² — is what all quoted rates
use. If you want more statistics, raise `n_evals`; nothing is precomputed.

## The notebooks

| # | Notebook | What it computes |
| --- | --- | --- |
| 0 | [`0_xsec_summary`](0_xsec_summary.ipynb) | Every SM channel the other notebooks use: σ/E for DIS, elastic scattering, inverse lepton decays, resonant ν̄ₑe⁻ formation, and tridents, with the per-channel event rates they imply. |
| 1 | [`1_muon_beam_flux`](1_muon_beam_flux.ipynb) | The tertiary muon beam made by ν CC interactions in the upstream rock: transport with nuPyProp, multiple scattering, inverse muon decay, and face maps. About 6×10¹¹ μ⁺ and 7×10¹¹ μ⁻ per year reach the face at ⟨E⟩ ≈ 1.2 TeV, roughly 2 per bunch crossing. |
| 2 | [`2_nutau_secondary_flux`](2_nutau_secondary_flux.ipynb) | Secondary ν_τ from charm chains in the rock, inverse tau decay, resonant ν̄ₑe⁻ → D_s*⁻, and ℓτ tridents. Roughly 0.2 ν_τ CC events per year across both detectors — about 10⁻¹⁰ of the beam CC rate. |
| 3 | [`3_wrong_sign_flux`](3_wrong_sign_flux.ipynb) | Wrong-sign neutrinos regenerated in the rock, from prompt inverse-muon-decay partners and charm semileptonic decays. Contaminates the sign-tagged CC rate at the few×10⁻⁹ level. |
| 4 | [`4_hnl_sensitivity`](4_hnl_sensitivity.ipynb) | Heavy neutral leptons coupled by mixing: DarkNews upscattering, channel-resolved decay widths, and displaced vertices in both the rock and the detector. Reach in \|U_μ4\|² and \|U_e4\|². |
| 5 | [`5_hnl_dipole_sensitivity`](5_hnl_dipole_sensitivity.ipynb) | HNLs coupled by an electromagnetic dipole: coherent, proton-elastic and DIS upscattering, with a ν–e scattering analysis and correlated systematics. |

## Supporting modules

| Module | Purpose |
| --- | --- |
| `muon_beam.py` | Muon transport through rock (nuPyProp energy loss + our own Highland scattering). |
| `nutau_secondary.py` | Charm production and decay, inverse tau decay, resonances, tridents, and the wrong-sign chains. |
| `hnl_tools.py`, `hnl_widths.py` | HNL portals, decay widths, and the detector geometry for displaced decays. |
| `dipole_limits.py` | Existing limits on the dipole portal, for the exclusion plots. |
| `validate_dipole_xsecs.py` | Standalone spinor-level validation of the dipole cross sections against DarkNews. Run it directly: `python validate_dipole_xsecs.py`. |

## Extra dependencies

The core `mint` package deliberately does not depend on these. Install what the
notebook you want needs:

```bash
pip install -e ".[examples]"     # DarkNews, parton, nupyprop
```

Notebooks 4 and 5 need [DarkNews](https://github.com/mhostert/DarkNews-generator);
notebook 1 needs [nuPyProp](https://github.com/NuSpaceSim/nupyprop); the trident
channels in notebook 2 additionally need a local install of
[NEPTUNE](https://github.com/mhostert/NEPTUNE).

The first run of notebooks 4 and 5 tabulates DarkNews cross sections and caches
them next to the notebook; subsequent runs reuse the cache.
