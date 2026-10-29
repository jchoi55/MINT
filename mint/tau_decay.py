"""Decay the taus GENIE leaves undecayed, with TAUOLA++.

GENIE keeps the tau of a nu_tau CC event as the final-state primary lepton
(``fspl`` = +-15, momentum in ``El``, ``pxl``, ``pyl``, ``pzl``) and, by
default, does not decay it (``DecayParticleWithCode=15`` is false in
CommonDecay.xml), so its decay products are not in the gst file.  This module
hands those taus to ``tauola/tau_decay``, a small driver around TAUOLA++'s tau
gun (``Tauola::decayOne``), with the polarization GENIE assigned: fully
left-handed tau-, fully right-handed tau+ (the massless-lepton limit).  The
driver's header explains TAUOLA's sign convention for that vector.

The daughters come back in the lab frame, one row per stable particle,
neutrinos included.  TAUOLA's own radiative photons are kept.  Results are
written next to the gst file (``<flavor>.taus.txt``, ``<flavor>.tau_daughters.txt``
and ``<flavor>.tauola.log``) and reused on the next call.

Building the driver needs a TAUOLA++ install (``configure --prefix=$TAUOLA_DIR``,
HepMC not required) and gfortran; see ``tauola/Makefile``.
"""
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER_DIR = os.path.join(HERE, "tauola")
DRIVER = os.path.join(DRIVER_DIR, "tau_decay")

PDG_TAU = 15
DAUGHTER_COLUMNS = ("iev", "pdg", "px", "py", "pz", "E")


def build(tauola_dir=None):
    """Compile the driver (``make`` in mint/tauola).  ``tauola_dir`` overrides
    the TAUOLA++ prefix, default ``~/opt/tauola``."""
    env = os.environ.copy()
    if tauola_dir:
        env["TAUOLA_DIR"] = os.path.abspath(os.path.expanduser(tauola_dir))
    subprocess.run(["make", "-C", DRIVER_DIR], check=True, env=env)
    return DRIVER


def export_taus(gst_path, out_txt):
    """Write the undecayed taus of a gst file in the driver's input format,
    "iev pdg px py pz E helicity", and return how many there were.

    helicity is GENIE's polarization projected on the tau direction: -1 for a
    tau-, +1 for a tau+ (``SetPrimaryLeptonPolarization``).
    """
    import uproot
    with uproot.open(gst_path) as f:
        a = f["gst"].arrays(["iev", "fspl", "El", "pxl", "pyl", "pzl"], library="np")
    sel = np.abs(a["fspl"]) == PDG_TAU
    pdg = a["fspl"][sel]
    helicity = np.where(pdg > 0, -1.0, 1.0)
    rows = np.column_stack([a["iev"][sel], pdg, a["pxl"][sel], a["pyl"][sel],
                            a["pzl"][sel], a["El"][sel], helicity])
    np.savetxt(out_txt, rows, fmt="%d %d %.9e %.9e %.9e %.9e %.1f",
               header=" ".join(DAUGHTER_COLUMNS) + " helicity")
    return int(sel.sum())


def read_daughters(path):
    """The driver's output as a dict of flat arrays (see DAUGHTER_COLUMNS);
    ``iev`` is the gst event number each daughter belongs to."""
    d = np.loadtxt(path, ndmin=2)
    if d.size == 0:
        d = np.empty((0, len(DAUGHTER_COLUMNS)))
    out = {name: d[:, i] for i, name in enumerate(DAUGHTER_COLUMNS)}
    out["iev"] = out["iev"].astype(int)
    out["pdg"] = out["pdg"].astype(int)
    return out


def output_paths(gst_path, out_dir=None):
    out_dir = out_dir or os.path.dirname(os.path.abspath(gst_path))
    stem = os.path.join(out_dir, os.path.basename(gst_path).replace(".gst.root", ""))
    return {"taus": stem + ".taus.txt", "daughters": stem + ".tau_daughters.txt",
            "log": stem + ".tauola.log"}


def decay_taus(gst_path, out_dir=None, seed=4357, force=False):
    """Decay every tau of a gst file and return its daughters (read_daughters).

    The driver runs once per file; the result on disk is reused unless
    ``force`` or the gst file is newer than it (a regenerated sample has new
    taus, and stale daughters would be matched to the wrong events by number).
    ``seed`` is TAUOLA's RANMAR seed.
    """
    paths = output_paths(gst_path, out_dir)
    stale = (not os.path.exists(paths["daughters"])
             or os.path.getmtime(gst_path) > os.path.getmtime(paths["daughters"]))
    if force or stale:
        if not os.path.exists(DRIVER):
            build()
        n = export_taus(gst_path, paths["taus"])
        with open(paths["log"], "w") as fh:
            subprocess.run([DRIVER, paths["taus"], paths["daughters"], str(seed)],
                           check=True, stdout=fh, stderr=subprocess.STDOUT)
        with open(paths["log"]) as fh:
            summary = [l for l in fh if "taus decayed" in l]
        print(f"{os.path.basename(gst_path)}: {n} taus -> "
              f"{summary[-1].strip() if summary else 'no summary in log'}")
    return read_daughters(paths["daughters"])


def decay_summary(daughters):
    """Branching fractions of the decayed sample: fraction with a muon, with an
    electron, hadronic (neither)."""
    iev = daughters["iev"]
    pdg = np.abs(daughters["pdg"])
    events = np.unique(iev)
    has_mu = np.isin(events, iev[pdg == 13])
    has_e = np.isin(events, iev[pdg == 11])
    n = max(events.size, 1)
    return {"n_tau": int(events.size), "mu": float(has_mu.sum() / n),
            "e": float(has_e.sum() / n), "hadronic": float((~has_mu & ~has_e).sum() / n)}
