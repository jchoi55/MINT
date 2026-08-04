# MINT physics examples

Applications of the MINT beam simulation to physics studies at a 10 TeV muon
collider forward-neutrino facility. Nothing in this folder is part of the
`mint` package — these examples import MINT (and, where needed, external BSM
tools) as libraries. Every notebook starts from the same two-line setup:

```python
import mint

ring, sims, ipy = mint.examples.both_beams()     # one vegas sample, all flavors
det = mint.detectors.load("hpgar_5km")           # the standard forward detector
E, w = det.face_flux(sims["numubar"], exposure=ipy)
```

`mint.examples` generates the muon-decay simulation (exact per-flavor
reweighting of a single vegas sample, μ⁺ beam clockwise / μ⁻ beam
counter-clockwise); `mint.detectors` bundles the detector geometry (10 m ×
4 m HPGAr TPC + 3×5 cm W production slabs at 5 km, 10 m air gap, rock
shielding) with the ray-tracing helpers. Need more statistics? Raise
`n_evals` — everything is generated on the fly.

Each notebook follows the same progression: **validation plots → main
result → variations**.

## The notebooks

| # | notebook | content |
| --- | --- | --- |
| 0 | `0_xsec_summary` | All SM scattering channels used across the studies: σ/E panels (DIS, ES, inverse lepton decays, ν̄_e-e meson resonances incl. D*/D_s*, tridents) and per-channel event rates in the standard detector. |
| 1 | `1_muon_beam_flux` | The tertiary muon "beam" from ν CC in the rock: nupyprop propagation (validated against Mott/CSDA/Highland), inverse muon decay, 2D face maps, spectra per bunch crossing (~3.5–4 μ/crossing at ⟨E⟩ ≈ 1.3 TeV). |
| 2 | `2_nutau_secondary_flux` | Secondary ν_τ sources: charm chains in the rock (dimuon-standard kinematics, polarized τ decays), inverse tau decay, resonant ν̄_e e⁻ → D_s*⁻, and ℓτ tridents (NEPTUNE cross sections + exact four-vector reconstruction, all four beam flavors); 1–50 km baseline scan. Grand total ≈ 2.3 ν_τ CC events/yr. |
| 3 | `3_wrong_sign_flux` | Wrong-sign neutrinos from rock regeneration: prompt-IMD partner neutrinos (μ⁻-beam flavors only) + charm semileptonic chains (D⁰/D±/D_s → Xℓν, shared dimuon-standard machinery); contamination of the sign-tagged CC rates at the 10⁻⁹–10⁻⁸ level. |
| 4 | `4_hnl_sensitivity` | Mixing-portal HNLs: DarkNews-exact upscattering, channel-resolved decay widths, rock + in-detector displaced vertices; \|U_μ4\|² and \|U_e4\|² reach (10 yr, both beams); inverse HNL decay ν̄_e e⁻ → ℓ⁻N̄ incl. the τ-coupling estimate. |
| 5 | `5_hnl_dipole_sensitivity` | Dipole-portal HNLs: coherent/p-elastic/DIS upscattering (DarkNews-exact, validated), visibility-classified samples, ν-e scattering analysis with correlated-systematics χ². |
| 6 | `6_muc_lfv_alp` | LFV ALP from the tertiary muon beam: μW → τaW (exact static-source 2→3, Mott-validated), anarchic-coupling sensitivity on the (m_a, 1/f_a) plane. |

Supporting modules: `muon_beam.py` (nupyprop muon transport),
`nutau_secondary.py` (charm/ITD/resonance/trident + wrong-sign chains; the trident source
needs a local [NEPTUNE](https://github.com/mhostert/NEPTUNE) install), `hnl_tools.py` +
`hnl_widths.py` (portals, widths, geometry),
`dipole_limits.py`, `muc_alp_tools.py` (LFV ALP production);
`validate_dipole_xsecs.py` is a standalone validation suite for the dipole
cross sections (spinor-level checks against DarkNews).

`alp/`, `3_lfv_main_plots.ipynb` and `digitized/` are the LFV-ALP library and
plotting assets from the companion paper (Fox et al.), used by notebook 6 for
conventions and existing limits.

Data directories: `dipole_limits_data/` (nu-dipole digitizations). The DarkNews cross sections
are disk-cached in `.dn_dipole_xsec_cache.npz` — delete it to force
recomputation.

Exposure conventions: notebooks 4–6 use 10 years with both beams and
identical detectors; notebooks 0–3 quote per-year rates.

Cross-section backends: `mint.xsecs.use_backend("ct18")` switches all DIS
cross sections to MINT's own LO calculation with CT18NNLO PDFs
(`mint.lo_dis` — PDF-consistent with the differential samplers, real NC,
cached interpolators); `use_backend("alfonso")` restores the shipped NNLO
tables (default). The two agree at the ~10% level (see the cross-check cell
in notebook 0).

## Setup

```bash
pip install mint-muc            # or `pip install -e .` from the repo root
pip install DarkNews parton     # HNL notebooks (exact upscattering cross sections)
pip install nupyprop            # muon transport (notebooks 1 and 6)
```

The existing-limits overlays additionally use
[`HNLimits`](https://github.com/mhostert/HNLimits) (notebook 4) and the local
`alp` library (notebook 6).
