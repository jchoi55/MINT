#!/usr/bin/env python3
"""Propagate TeV muons through standard rock with nupyprop loss tables.

This is a flat-slab toy transport for collider-forward distances, not the
Earth-skimming neutrino geometry used by nupyprop's command-line runner.
Energy losses use nupyprop's standard-rock muon alpha, beta, cross-section,
and inelasticity-CDF tables. Angular broadening is an optional Highland
multiple-scattering approximation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import tempfile
from typing import Sequence

import numpy as np


PROCESS_NAMES = ("bremsstrahlung", "pair_production", "photonuclear")
MCS_CONSTANT_GEV = 13.6e-3


@dataclass(frozen=True)
class MuonRockTables:
    energy_grid_gev: np.ndarray
    log_energy_grid: np.ndarray
    density_g_cm3: float
    muon_mass_gev: float
    alpha_gev_cm2_g: np.ndarray
    beta_cut_cm2_g: np.ndarray
    beta_total_cm2_g: np.ndarray
    hard_xc_cm2_g: np.ndarray
    quantile_y: np.ndarray
    pn_model: str


@dataclass
class SimulationResult:
    distances_km: np.ndarray
    energy_gev: np.ndarray
    theta_mrad: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    summary_rows: list[dict[str, float]]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Propagate 5 TeV-scale muons through standard rock and write "
            "survival, energy, and angular distributions."
        )
    )
    parser.add_argument("--energy-gev", type=float, default=5_000.0)
    parser.add_argument("--initial-energy-sigma-gev", type=float, default=0.0)
    parser.add_argument("--n-muons", type=int, default=10_000)
    parser.add_argument("--distances-km", type=float, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--step-m", type=float, default=1.0)
    parser.add_argument("--mode", choices=("stochastic", "continuous"), default="stochastic")
    parser.add_argument("--pn-model", default="allm", choices=("allm", "bb"))
    parser.add_argument("--min-energy-gev", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--initial-angle-sigma-urad", type=float, default=0.0)
    parser.add_argument("--no-mcs", action="store_true", help="Disable multiple scattering.")
    parser.add_argument(
        "--radiation-length-g-cm2",
        type=float,
        default=25.0,
        help="Standard-rock radiation length used for Highland scattering.",
    )
    parser.add_argument(
        "--rock-density-g-cm3",
        type=float,
        default=None,
        help="Override nupyprop's standard-rock density.",
    )
    parser.add_argument(
        "--quantile-grid",
        type=int,
        default=4096,
        help="Number of quantile points used to invert nupyprop inelasticity CDFs.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("plots/muon_rock_propagation"))
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--save-per-muon-csv", action="store_true")
    return parser.parse_args(argv)


def load_nupyprop_tables(pn_model: str, quantile_grid: int) -> MuonRockTables:
    try:
        import nupyprop.constants as const
        import nupyprop.data as data
    except ImportError as exc:
        raise SystemExit(
            "nupyprop is required for this script but is not importable.\n"
            "Practical alternatives before proceeding are: PROPOSAL for "
            "stochastic muon transport, MMC/MUSIC for muon range studies, or "
            "Geant4 for full detector/rock geometry with scattering."
        ) from exc

    energy_grid = np.asarray(const.E_lep, dtype=float)
    alpha = np.asarray(data.get_alpha("muon", "rock"), dtype=float)
    beta_cut = np.asarray(data.get_beta("muon", pn_model, "rock", "cut"), dtype=float)
    beta_total = np.asarray(data.get_beta("muon", pn_model, "rock", "total"), dtype=float)
    hard_xc = np.asarray(data.get_xc("muon", pn_model, "rock"), dtype=float)
    ixc = np.asarray(data.get_ixc("muon", pn_model, "rock"), dtype=float)

    expected = (31, energy_grid.size, len(PROCESS_NAMES))
    if ixc.shape != expected:
        raise RuntimeError(f"Unexpected nupyprop CDF shape {ixc.shape}; expected {expected}.")

    y_grid = np.insert(10 ** (-np.linspace(0.1, 3.0, num=30)), 0, 1.0)
    quantile_y = invert_inelasticity_cdfs(ixc, y_grid, quantile_grid)

    return MuonRockTables(
        energy_grid_gev=energy_grid,
        log_energy_grid=np.log10(energy_grid),
        density_g_cm3=float(const.rho_rock),
        muon_mass_gev=float(const.m_mu),
        alpha_gev_cm2_g=alpha,
        beta_cut_cm2_g=beta_cut,
        beta_total_cm2_g=beta_total,
        hard_xc_cm2_g=hard_xc,
        quantile_y=quantile_y,
        pn_model=pn_model,
    )


def invert_inelasticity_cdfs(ixc: np.ndarray, y_grid: np.ndarray, n_quantiles: int) -> np.ndarray:
    if n_quantiles < 128:
        raise ValueError("--quantile-grid should be at least 128.")

    q_grid = np.linspace(0.0, 1.0, n_quantiles)
    n_energy = ixc.shape[1]
    n_process = ixc.shape[2]
    quantile_y = np.zeros((n_process, n_energy, n_quantiles), dtype=float)

    for process in range(n_process):
        for energy_index in range(n_energy):
            cdf = np.asarray(ixc[:, energy_index, process], dtype=float)
            cdf = np.nan_to_num(cdf, nan=0.0, posinf=1.0, neginf=0.0)
            cdf = np.maximum.accumulate(np.clip(cdf, 0.0, 1.0))
            if cdf[-1] <= 0.0:
                continue

            cdf = cdf / cdf[-1]
            unique_cdf, unique_index = np.unique(cdf, return_index=True)
            unique_y = y_grid[unique_index]
            if unique_cdf[0] > 0.0:
                unique_cdf = np.insert(unique_cdf, 0, 0.0)
                unique_y = np.insert(unique_y, 0, 1.0)
            if unique_cdf[-1] < 1.0:
                unique_cdf = np.append(unique_cdf, 1.0)
                unique_y = np.append(unique_y, y_grid[-1])

            quantile_y[process, energy_index] = np.interp(q_grid, unique_cdf, unique_y)

    return quantile_y


def interp_energy(values: np.ndarray, energy_gev: np.ndarray, tables: MuonRockTables) -> np.ndarray:
    energy = np.clip(energy_gev, tables.energy_grid_gev[0], tables.energy_grid_gev[-1])
    log_energy = np.log10(energy)

    if values.ndim == 1:
        return np.interp(log_energy, tables.log_energy_grid, values)

    out = np.empty((energy.size, values.shape[1]), dtype=float)
    for column in range(values.shape[1]):
        out[:, column] = np.interp(log_energy, tables.log_energy_grid, values[:, column])
    return out


def apply_continuous_loss(
    energy_gev: np.ndarray,
    column_depth_g_cm2: float,
    alpha_gev_cm2_g: np.ndarray,
    beta_cm2_g: np.ndarray,
) -> np.ndarray:
    beta = np.maximum(beta_cm2_g, 0.0)
    alpha = np.maximum(alpha_gev_cm2_g, 0.0)
    out = np.empty_like(energy_gev)
    small_beta = beta * column_depth_g_cm2 < 1e-8

    out[small_beta] = energy_gev[small_beta] - alpha[small_beta] * column_depth_g_cm2
    active = ~small_beta
    if np.any(active):
        ratio = alpha[active] / beta[active]
        out[active] = (energy_gev[active] + ratio) * np.exp(-beta[active] * column_depth_g_cm2) - ratio

    return np.maximum(out, 0.0)


def sample_inelasticity(
    energy_gev: np.ndarray,
    process: np.ndarray,
    rng: np.random.Generator,
    tables: MuonRockTables,
) -> np.ndarray:
    n = energy_gev.size
    if n == 0:
        return np.empty(0, dtype=float)

    energy = np.clip(energy_gev, tables.energy_grid_gev[0], tables.energy_grid_gev[-1])
    log_energy = np.log10(energy)
    high = np.searchsorted(tables.log_energy_grid, log_energy, side="right")
    high = np.clip(high, 1, tables.log_energy_grid.size - 1)
    low = high - 1
    energy_weight = (
        (log_energy - tables.log_energy_grid[low])
        / (tables.log_energy_grid[high] - tables.log_energy_grid[low])
    )

    n_quantiles = tables.quantile_y.shape[2]
    q_float = rng.random(n) * (n_quantiles - 1)
    q_low = np.floor(q_float).astype(int)
    q_high = np.minimum(q_low + 1, n_quantiles - 1)
    q_weight = q_float - q_low

    y = np.zeros(n, dtype=float)
    for process_id in range(len(PROCESS_NAMES)):
        mask = process == process_id
        if not np.any(mask):
            continue

        lo = low[mask]
        hi = high[mask]
        q0 = q_low[mask]
        q1 = q_high[mask]
        qw = q_weight[mask]
        ew = energy_weight[mask]
        quantiles = tables.quantile_y[process_id]

        y_lo = quantiles[lo, q0] + qw * (quantiles[lo, q1] - quantiles[lo, q0])
        y_hi = quantiles[hi, q0] + qw * (quantiles[hi, q1] - quantiles[hi, q0])
        y[mask] = y_lo + ew * (y_hi - y_lo)

    return np.clip(y, 0.0, 0.999999)


def propagate_one_step(
    energy_gev: np.ndarray,
    theta_x: np.ndarray,
    theta_y: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    step_m: float,
    mode: str,
    min_energy_gev: float,
    include_mcs: bool,
    radiation_length_g_cm2: float,
    rng: np.random.Generator,
    tables: MuonRockTables,
) -> None:
    alive_index = np.flatnonzero(energy_gev > min_energy_gev)
    if alive_index.size == 0:
        return

    x_m[alive_index] += theta_x[alive_index] * step_m
    y_m[alive_index] += theta_y[alive_index] * step_m

    column_depth = tables.density_g_cm3 * step_m * 100.0
    e_alive = energy_gev[alive_index]
    alpha = interp_energy(tables.alpha_gev_cm2_g, e_alive, tables)

    beta_table = tables.beta_total_cm2_g if mode == "continuous" else tables.beta_cut_cm2_g
    beta = np.sum(interp_energy(beta_table, e_alive, tables), axis=1)
    energy_gev[alive_index] = apply_continuous_loss(e_alive, column_depth, alpha, beta)

    if mode == "stochastic":
        apply_hard_losses(energy_gev, alive_index, column_depth, min_energy_gev, rng, tables)

    energy_gev[energy_gev < min_energy_gev] = 0.0
    if include_mcs:
        apply_multiple_scattering(
            energy_gev,
            theta_x,
            theta_y,
            x_m,
            y_m,
            step_m,
            column_depth,
            min_energy_gev,
            radiation_length_g_cm2,
            rng,
            tables,
        )


def apply_hard_losses(
    energy_gev: np.ndarray,
    candidate_index: np.ndarray,
    column_depth_g_cm2: float,
    min_energy_gev: float,
    rng: np.random.Generator,
    tables: MuonRockTables,
) -> None:
    active = candidate_index[energy_gev[candidate_index] > min_energy_gev]
    if active.size == 0:
        return

    sigma = np.maximum(interp_energy(tables.hard_xc_cm2_g, energy_gev[active], tables), 0.0)
    total_lambda = np.sum(sigma, axis=1) * column_depth_g_cm2
    n_interactions = rng.poisson(total_lambda)
    max_interactions = int(np.max(n_interactions)) if n_interactions.size else 0

    for interaction_number in range(max_interactions):
        interacting = active[(n_interactions > interaction_number) & (energy_gev[active] > min_energy_gev)]
        if interacting.size == 0:
            continue

        sigma_now = np.maximum(interp_energy(tables.hard_xc_cm2_g, energy_gev[interacting], tables), 0.0)
        sigma_total = np.sum(sigma_now, axis=1)
        valid = sigma_total > 0.0
        if not np.any(valid):
            continue

        interacting = interacting[valid]
        sigma_now = sigma_now[valid]
        sigma_total = sigma_total[valid]
        draw = rng.random(interacting.size) * sigma_total
        process = np.zeros(interacting.size, dtype=int)
        process[draw >= sigma_now[:, 0]] = 1
        process[draw >= sigma_now[:, 0] + sigma_now[:, 1]] = 2

        y_loss = sample_inelasticity(energy_gev[interacting], process, rng, tables)
        energy_gev[interacting] *= 1.0 - y_loss


def apply_multiple_scattering(
    energy_gev: np.ndarray,
    theta_x: np.ndarray,
    theta_y: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    step_m: float,
    column_depth_g_cm2: float,
    min_energy_gev: float,
    radiation_length_g_cm2: float,
    rng: np.random.Generator,
    tables: MuonRockTables,
) -> None:
    alive = np.flatnonzero(energy_gev > min_energy_gev)
    if alive.size == 0:
        return

    thickness_x0 = column_depth_g_cm2 / radiation_length_g_cm2
    if thickness_x0 <= 0.0:
        return

    energy = energy_gev[alive]
    momentum = np.sqrt(np.maximum(energy * energy - tables.muon_mass_gev**2, 1e-24))
    beta_rel = np.clip(momentum / energy, 1e-12, 1.0)
    highland = 1.0 + 0.038 * np.log(thickness_x0)
    highland = max(highland, 0.0)
    theta0 = MCS_CONSTANT_GEV / (beta_rel * momentum) * np.sqrt(thickness_x0) * highland

    kick_x = rng.normal(0.0, theta0)
    kick_y = rng.normal(0.0, theta0)
    theta_x[alive] += kick_x
    theta_y[alive] += kick_y
    x_m[alive] += 0.5 * kick_x * step_m
    y_m[alive] += 0.5 * kick_y * step_m


def simulate(args: argparse.Namespace, tables: MuonRockTables) -> SimulationResult:
    distances_km = np.unique(np.asarray(args.distances_km, dtype=float))
    distances_km.sort()
    if distances_km.size == 0 or np.any(distances_km <= 0.0):
        raise ValueError("--distances-km must contain positive distances.")
    if args.energy_gev <= args.min_energy_gev:
        raise ValueError("--energy-gev must be larger than --min-energy-gev.")
    if args.n_muons <= 0:
        raise ValueError("--n-muons must be positive.")
    if args.step_m <= 0.0:
        raise ValueError("--step-m must be positive.")
    if args.radiation_length_g_cm2 <= 0.0:
        raise ValueError("--radiation-length-g-cm2 must be positive.")

    if args.rock_density_g_cm3 is not None:
        tables = MuonRockTables(
            energy_grid_gev=tables.energy_grid_gev,
            log_energy_grid=tables.log_energy_grid,
            density_g_cm3=args.rock_density_g_cm3,
            muon_mass_gev=tables.muon_mass_gev,
            alpha_gev_cm2_g=tables.alpha_gev_cm2_g,
            beta_cut_cm2_g=tables.beta_cut_cm2_g,
            beta_total_cm2_g=tables.beta_total_cm2_g,
            hard_xc_cm2_g=tables.hard_xc_cm2_g,
            quantile_y=tables.quantile_y,
            pn_model=tables.pn_model,
        )

    rng = np.random.default_rng(args.seed)
    energy_gev = np.full(args.n_muons, args.energy_gev, dtype=float)
    if args.initial_energy_sigma_gev > 0.0:
        energy_gev = rng.normal(args.energy_gev, args.initial_energy_sigma_gev, args.n_muons)
        energy_gev = np.clip(energy_gev, args.min_energy_gev, None)

    initial_angle_sigma_rad = args.initial_angle_sigma_urad * 1e-6
    theta_x = rng.normal(0.0, initial_angle_sigma_rad, args.n_muons)
    theta_y = rng.normal(0.0, initial_angle_sigma_rad, args.n_muons)
    x_m = np.zeros(args.n_muons, dtype=float)
    y_m = np.zeros(args.n_muons, dtype=float)

    n_record = distances_km.size
    energy_record = np.zeros((n_record, args.n_muons), dtype=float)
    theta_record = np.zeros_like(energy_record)
    x_record = np.zeros_like(energy_record)
    y_record = np.zeros_like(energy_record)

    distance_m = 0.0
    include_mcs = not args.no_mcs
    for record_index, target_km in enumerate(distances_km):
        target_m = target_km * 1000.0
        while distance_m < target_m - 1e-12:
            step_m = min(args.step_m, target_m - distance_m)
            propagate_one_step(
                energy_gev,
                theta_x,
                theta_y,
                x_m,
                y_m,
                step_m,
                args.mode,
                args.min_energy_gev,
                include_mcs,
                args.radiation_length_g_cm2,
                rng,
                tables,
            )
            distance_m += step_m

        energy_record[record_index] = energy_gev
        theta_record[record_index] = 1e3 * np.hypot(theta_x, theta_y)
        x_record[record_index] = x_m
        y_record[record_index] = y_m

    summary_rows = summarize(distances_km, energy_record, theta_record, x_record, y_record, args)
    return SimulationResult(distances_km, energy_record, theta_record, x_record, y_record, summary_rows)


def percentile_or_nan(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, percentile))


def summarize(
    distances_km: np.ndarray,
    energy_gev: np.ndarray,
    theta_mrad: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for index, distance in enumerate(distances_km):
        survivor = energy_gev[index] > args.min_energy_gev
        e = energy_gev[index, survivor]
        theta = theta_mrad[index, survivor]
        radius = np.hypot(x_m[index, survivor], y_m[index, survivor])
        rows.append(
            {
                "distance_km": float(distance),
                "survival_fraction": float(np.mean(survivor)),
                "survivors": int(np.count_nonzero(survivor)),
                "energy_mean_gev": float(np.mean(e)) if e.size else float("nan"),
                "energy_p05_gev": percentile_or_nan(e, 5),
                "energy_p16_gev": percentile_or_nan(e, 16),
                "energy_p50_gev": percentile_or_nan(e, 50),
                "energy_p84_gev": percentile_or_nan(e, 84),
                "energy_p95_gev": percentile_or_nan(e, 95),
                "theta_p50_mrad": percentile_or_nan(theta, 50),
                "theta_p68_mrad": percentile_or_nan(theta, 68),
                "theta_p95_mrad": percentile_or_nan(theta, 95),
                "radius_p50_m": percentile_or_nan(radius, 50),
                "radius_p95_m": percentile_or_nan(radius, 95),
            }
        )
    return rows


def write_outputs(args: argparse.Namespace, tables: MuonRockTables, result: SimulationResult) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result.summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result.summary_rows)

    npz_path = args.output_dir / "muon_samples.npz"
    np.savez_compressed(
        npz_path,
        distances_km=result.distances_km,
        energy_gev=result.energy_gev,
        theta_mrad=result.theta_mrad,
        x_m=result.x_m,
        y_m=result.y_m,
        process_names=np.asarray(PROCESS_NAMES),
        initial_energy_gev=args.energy_gev,
        min_energy_gev=args.min_energy_gev,
        mode=args.mode,
        pn_model=tables.pn_model,
        density_g_cm3=tables.density_g_cm3,
    )

    if args.save_per_muon_csv:
        samples_path = args.output_dir / "per_muon_samples.csv"
        with samples_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["distance_km", "muon_id", "energy_gev", "theta_mrad", "x_m", "y_m"])
            for distance_index, distance in enumerate(result.distances_km):
                for muon_id in range(result.energy_gev.shape[1]):
                    writer.writerow(
                        [
                            f"{distance:.8g}",
                            muon_id,
                            f"{result.energy_gev[distance_index, muon_id]:.8g}",
                            f"{result.theta_mrad[distance_index, muon_id]:.8g}",
                            f"{result.x_m[distance_index, muon_id]:.8g}",
                            f"{result.y_m[distance_index, muon_id]:.8g}",
                        ]
                    )

    if not args.no_plots:
        write_plots(args.output_dir, result, args)

    print(f"Wrote {summary_path}")
    print(f"Wrote {npz_path}")
    print_summary(result.summary_rows)


def write_plots(output_dir: Path, result: SimulationResult, args: argparse.Namespace) -> None:
    matplotlib_cache = Path(tempfile.gettempdir()) / "mint_matplotlib_cache"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    import matplotlib.pyplot as plt

    write_energy_plot(output_dir, result, args, plt)
    write_survival_plot(output_dir, result, plt)
    write_angle_plot(output_dir, result, args, plt)
    write_radius_plot(output_dir, result, args, plt)


def write_energy_plot(output_dir: Path, result: SimulationResult, args: argparse.Namespace, plt) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    upper = max(args.energy_gev * 1.05, args.min_energy_gev * 10.0)
    bins = np.geomspace(args.min_energy_gev, upper, 90)
    for index, distance in enumerate(result.distances_km):
        survived = result.energy_gev[index] > args.min_energy_gev
        values = result.energy_gev[index, survived]
        label = f"{distance:g} km, surv={np.mean(survived):.3f}"
        if values.size:
            ax.hist(values, bins=bins, histtype="step", density=True, linewidth=1.5, label=label)
        else:
            ax.plot([], [], label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Muon energy after rock [GeV]")
    ax.set_ylabel("Density among surviving muons")
    ax.set_title(f"{args.energy_gev:g} GeV muons in standard rock")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "energy_distribution.png", dpi=200)
    plt.close(fig)


def write_survival_plot(output_dir: Path, result: SimulationResult, plt) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    survival = [row["survival_fraction"] for row in result.summary_rows]
    ax.plot(result.distances_km, survival, marker="o")
    ax.set_xlabel("Rock distance [km]")
    ax.set_ylabel("Survival fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "survival_fraction.png", dpi=200)
    plt.close(fig)


def write_angle_plot(output_dir: Path, result: SimulationResult, args: argparse.Namespace, plt) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    all_theta = []
    for index in range(result.distances_km.size):
        survived = result.energy_gev[index] > args.min_energy_gev
        if np.any(survived):
            all_theta.append(result.theta_mrad[index, survived])
    if all_theta:
        theta_max = float(np.percentile(np.concatenate(all_theta), 99.5))
        theta_max = max(theta_max, 1e-6)
    else:
        theta_max = 1.0
    bins = np.linspace(0.0, theta_max, 90)

    for index, distance in enumerate(result.distances_km):
        survived = result.energy_gev[index] > args.min_energy_gev
        values = result.theta_mrad[index, survived]
        label = f"{distance:g} km"
        if values.size:
            ax.hist(values, bins=bins, histtype="step", density=True, linewidth=1.5, label=label)
        else:
            ax.plot([], [], label=label)
    ax.set_xlabel("Space angle relative to initial direction [mrad]")
    ax.set_ylabel("Density among surviving muons")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "angular_distribution.png", dpi=200)
    plt.close(fig)


def write_radius_plot(output_dir: Path, result: SimulationResult, args: argparse.Namespace, plt) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    all_radius = []
    radius = np.hypot(result.x_m, result.y_m)
    for index in range(result.distances_km.size):
        survived = result.energy_gev[index] > args.min_energy_gev
        if np.any(survived):
            all_radius.append(radius[index, survived])
    if all_radius:
        radius_max = float(np.percentile(np.concatenate(all_radius), 99.5))
        radius_max = max(radius_max, 1e-6)
    else:
        radius_max = 1.0
    bins = np.linspace(0.0, radius_max, 90)

    for index, distance in enumerate(result.distances_km):
        survived = result.energy_gev[index] > args.min_energy_gev
        values = radius[index, survived]
        label = f"{distance:g} km"
        if values.size:
            ax.hist(values, bins=bins, histtype="step", density=True, linewidth=1.5, label=label)
        else:
            ax.plot([], [], label=label)
    ax.set_xlabel("Transverse displacement [m]")
    ax.set_ylabel("Density among surviving muons")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "transverse_displacement.png", dpi=200)
    plt.close(fig)


def print_summary(rows: list[dict[str, float]]) -> None:
    print("\nSummary for surviving muons:")
    print(
        " distance_km  survival  E50_GeV  E16_GeV  E84_GeV  "
        "theta95_mrad  radius95_m"
    )
    for row in rows:
        print(
            f" {row['distance_km']:11.3g}"
            f"  {row['survival_fraction']:8.4f}"
            f"  {row['energy_p50_gev']:8.3g}"
            f"  {row['energy_p16_gev']:8.3g}"
            f"  {row['energy_p84_gev']:8.3g}"
            f"  {row['theta_p95_mrad']:12.3g}"
            f"  {row['radius_p95_m']:10.3g}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    tables = load_nupyprop_tables(args.pn_model, args.quantile_grid)
    result = simulate(args, tables)
    write_outputs(args, tables, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
