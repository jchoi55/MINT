"""User-facing entry point for creating :class:`mint.lattice_tools.Lattice` objects.

MINT ships a small library of reference accelerator optics (MAD-X TWISS/TFS
tables) as package data in ``mint/lattice_data/``. This module turns those --
or any TFS file of your own -- into a ready-to-use ``Lattice`` in one call:

    import mint

    # a lattice shipped with MINT (see mint.lattices.available())
    ring = mint.lattices.load("mc_10tev_hybrid_v06")

    # or your own MAD-X TWISS/TFS file
    ring = mint.lattices.from_tfs("my_ring.tfs", emittance_RMS=5.25e-10)

Both return a ``Lattice`` with the smoothed geometry, beam envelopes and
divergences, and the operational beam parameters (``Nmu_per_bunch``, ``finj``,
``bunch_multiplicity``, ``duty_factor``) already attached. All keyword
arguments of :func:`mint.lattice_tools.create_smoothed_lattice` can be overridden.
"""

from importlib.resources import files

import numpy as np

from mint import lattice_tools


# Geometric RMS emittance used in the 10 TeV MuC studies: normalized emittance
# of 25 um-rad divided by the relativistic gamma of a 5 TeV muon (~0.105 GeV mass).
_EMITTANCE_RMS_10TEV = 25e-6 / (5e3 / 0.105)  # [m rad]
# 3 TeV design: same 25 um normalized emittance, lower relativistic gamma.
_EMITTANCE_RMS_3TEV = 25e-6 / (1.5e3 / 0.105)  # [m rad]

#: Registry of lattices shipped with MINT. Each entry maps a short name to the
#: packaged TFS/TWISS file, the default smoothing settings, and the default
#: operational beam parameters used in the collider-ring studies.
REGISTRY = {
    "mc_10tev_hybrid_v06": {
        "file": "twiss_ir1ccma_hybrid_v06.outx",
        "description": (
            "10 TeV muon collider interaction region, hybrid IR design v06 "
            "(MAD-X twiss: ir1ccma_hybrid_v06; covers ~1.5 km of the 10 km ring)"
        ),
        "emittance_RMS": _EMITTANCE_RMS_10TEV,
        "Nmu_per_bunch": 2e12,
        "finj": 5.0,
        "bunch_multiplicity": 1.0,
        "duty_factor": 1 / np.pi,  # ~1e7 s of operation per year
        # full machine circumference [cm]: the twiss covers only the IR, muons
        # circulate (and decay) over the whole 10 km ring
        "total_circumference": 10e5,
        # longitudinal beam parameters (IMCC 10 TeV design):
        # relative momentum spread dp/p and RMS bunch length [cm] (1.5 mm ~ 5 ps)
        "beamdiv_z": 1.0e-3,
        "beamsize_z": 0.15,
    },
    "mc_10tev_IR_v09": {
        "file": "twiss_IR_v09.outx",
        "description": (
            "10 TeV muon collider interaction region only, v09 "
            "(sublattice of the 10 km ring)"
        ),
        "emittance_RMS": _EMITTANCE_RMS_10TEV,
        "Nmu_per_bunch": 2e12,
        "finj": 5.0,
        "bunch_multiplicity": 1.0,
        "duty_factor": 1 / np.pi,
        "total_circumference": 10e5,
        "beamdiv_z": 1.0e-3,
        "beamsize_z": 0.15,
    },
    "mc_10tev_ring_v06": {
        "file": "ring_v06.tfs.gz",
        "description": (
            "10 TeV muon collider, the FULL 8.7 km ring, design v06 "
            "(arcs included, so no total_circumference override is needed)"
        ),
        "emittance_RMS": _EMITTANCE_RMS_10TEV,
        "Nmu_per_bunch": 2e12,
        "finj": 5.0,
        "bunch_multiplicity": 1.0,
        "duty_factor": 1 / np.pi,
        "beamdiv_z": 1.0e-3,
        "beamsize_z": 0.15,
    },
    "mc_3tev_v1.2": {
        "file": "mc3tev_v1.2.tfs.gz",
        "description": (
            "3 TeV muon collider, the full 4.3 km ring, design v1.2 "
            "(1.5 TeV per beam)"
        ),
        "emittance_RMS": _EMITTANCE_RMS_3TEV,
        "Nmu_per_bunch": 2.2e12,
        "finj": 5.0,
        "bunch_multiplicity": 1.0,
        "duty_factor": 1 / np.pi,
        "beamdiv_z": 1.0e-3,
        "beamsize_z": 0.5,
    },
}

#: Operational beam parameters attached to every lattice (with their defaults).
BEAM_PARAMETERS = {
    "Nmu_per_bunch": 1.0,
    "finj": 1.0,
    "bunch_multiplicity": 1.0,
    "duty_factor": 1.0,
}


def available():
    """Return {name: description} of the lattices shipped with MINT."""
    return {name: entry["description"] for name, entry in REGISTRY.items()}


def path(name):
    """Absolute path of a packaged lattice file (raises KeyError if unknown)."""
    return str(files("mint.lattice_data").joinpath(REGISTRY[name]["file"]))


def read_tfs(name_or_path):
    """Read a MAD-X TFS/TWISS table into a DataFrame (with survey geometry).

    Accepts either a registry name (see :func:`available`) or a filesystem path.
    """
    if name_or_path in REGISTRY:
        name_or_path = path(name_or_path)
    return lattice_tools.get_lattice_dataframe_from_tfs(name_or_path)


def from_tfs(
    tfs_file,
    emittance_RMS,
    n_elements=120_000,
    rotated=True,
    midpoint=True,
    if_sublattice=False,
    name=None,
    short_name=None,
    Nmu_per_bunch=None,
    finj=None,
    bunch_multiplicity=None,
    duty_factor=None,
    total_circumference=None,
    beamdiv_z=None,
    beamsize_z=None,
    **smoothing_kwargs,
):
    """Create a ``Lattice`` from a MAD-X TFS/TWISS file in one call.

    Parameters
    ----------
    tfs_file : str
        Path to the TFS/TWISS table (or a registry name, see :func:`load`).
    emittance_RMS : float
        Geometric RMS emittance [m rad] used for beam envelopes/divergences.
    n_elements, rotated, midpoint, if_sublattice :
        Passed to :func:`mint.lattice_tools.create_smoothed_lattice`.
    name, short_name : str, optional
        Labels attached to the lattice (default: the file stem).
    Nmu_per_bunch, finj, bunch_multiplicity, duty_factor : float, optional
        Operational beam parameters (default 1, i.e. per-muon normalization).
    total_circumference : float, optional
        Full machine circumference [cm]. Set this when the TFS file covers only
        a portion of the machine (e.g. the interaction region of a larger ring):
        muon decays are placed on the covered section only, while the muons
        propagate -- and age/decay -- over the full circumference. Default None
        (the lattice is the whole machine).
    beamdiv_z : float or callable, optional
        Relative momentum spread dp/p of the beam (RMS). Default None (0).
        A scalar value is also used as the ``sigma_delta`` of the optics
        smoothing (unless overridden), so the dispersive contributions
        (D_x dp/p in size, D'_x dp/p in divergence) enter the horizontal
        beam envelopes.
    beamsize_z : float or callable, optional
        RMS bunch length [cm]; smears the decay times relative to the bunch
        crossing. Default None (0).
    **smoothing_kwargs :
        Any extra keyword accepted by ``create_smoothed_lattice`` (e.g.
        ``include_dispersion``, ``sigma_delta``).

    Returns
    -------
    mint.lattice_tools.Lattice
    """
    df = read_tfs(tfs_file)
    # A scalar dp/p doubles as the dispersion sigma_delta of the optics
    # (sigma_x^2 = eps*beta + (D_x dp/p)^2 and likewise for the divergence).
    if "sigma_delta" not in smoothing_kwargs and isinstance(beamdiv_z, (int, float)):
        smoothing_kwargs["sigma_delta"] = float(beamdiv_z)
    lattice_dict = lattice_tools.create_smoothed_lattice(
        df,
        emittance_RMS=emittance_RMS,
        n_elements=n_elements,
        rotated=rotated,
        midpoint=midpoint,
        if_sublattice=if_sublattice,
        **smoothing_kwargs,
    )
    # Longitudinal beam parameters: insert before construction so the Lattice
    # class wraps scalars into constant callables.
    if beamdiv_z is not None:
        lattice_dict["beamdiv_z"] = beamdiv_z
    if beamsize_z is not None:
        lattice_dict["beamsize_z"] = beamsize_z
    lattice = lattice_tools.Lattice(**lattice_dict)

    stem = str(tfs_file).split("/")[-1].rsplit(".", 1)[0]
    lattice.name = name if name is not None else stem
    lattice.short_name = short_name if short_name is not None else lattice.name

    overrides = {
        "Nmu_per_bunch": Nmu_per_bunch,
        "finj": finj,
        "bunch_multiplicity": bunch_multiplicity,
        "duty_factor": duty_factor,
    }
    for key, default in BEAM_PARAMETERS.items():
        value = overrides[key]
        setattr(lattice, key, default if value is None else value)

    if total_circumference is not None:
        lattice.total_circumference = total_circumference

    return lattice


def load(name, **overrides):
    """Load a lattice shipped with MINT by name, e.g. ``"mc_10tev_hybrid_v06"``.

    Any keyword of :func:`from_tfs` can be overridden, e.g.
    ``load("mc_10tev_hybrid_v06", Nmu_per_bunch=1.8e12, n_elements=60_000)``.
    """
    if name not in REGISTRY:
        known = ", ".join(REGISTRY)
        raise KeyError(f"Unknown lattice '{name}'. Shipped lattices: {known}")
    entry = REGISTRY[name]

    kwargs = {
        "emittance_RMS": entry["emittance_RMS"],
        "name": name,
        "short_name": name,
        "total_circumference": entry.get("total_circumference"),
        "beamdiv_z": entry.get("beamdiv_z"),
        "beamsize_z": entry.get("beamsize_z"),
    }
    for key in BEAM_PARAMETERS:
        kwargs[key] = entry.get(key)
    kwargs.update(overrides)

    return from_tfs(path(name), **kwargs)
