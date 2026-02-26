# Muon Induced Neutrino Tool (MINT)

<img src="assets/mint-logo.svg" alt="MINT logo" width="120" />

Generating neutrino fluxes at muon facilities, including neutrino factories and muon colliders.


## Installation

### From source (for now)

```bash
pip install .
```

For development:

```bash
pip install -e .[dev]
```
and then you can just edit files in this repo directly.

## Packaging

This Python package is built via `pyproject.toml` (PEP 517/518/621).

Build source and wheel distributions with:

```bash
python -m build
```

The generated artifacts are placed in `dist/`.


## Explanations

* `mint/MuC.py` contains all the main classes for generating muon decay events along a specific geometry.

* The optics of the collider rings are generated in the notebook `create_beam_optics_files.ipynb`.

* Low-energy accelerator studies can be found in `LA_studies.ipynb`

* Main collider ring studies can be found in `collider_studies.ipynb`