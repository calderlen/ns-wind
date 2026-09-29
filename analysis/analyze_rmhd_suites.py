"""Compare completed nonrotating and rotating RMHD parameter suites.

The analysis reads PLUTO ``dbl`` snapshots directly.  It writes plots and a
JSON provenance/metric manifest, but intentionally does not write CSV files.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

from rmhd_diagnostics import (
    SURFACE_RADIUS_KM, derive_rmhd_profiles, interpolate_profile,
    outward_crossing_radii,
)
from wind_common import (
    REPO_DIR, read_catalog, read_definitions, read_grid, read_parameters,
    read_snapshot,
)


DEFAULT_NONROTATING = REPO_DIR / "runs" / "rmhd_experiments"
DEFAULT_ROTATING = REPO_DIR / "runs" / "rmhd_rot_experiments"
DEFAULT_OUTPUT = REPO_DIR / "runs" / "rmhd_analysis"
SAMPLE_RADII = (20.0, 100.0, 1000.0)
PROFILE_KEYS = (
    "velocity_c", "velocity_phi_c", "sound_speed_c", "alfven_speed_c",
    "mdot_g_s", "kinetic_power_erg_s", "poynting_power_erg_s",
    "angular_momentum_fluid_erg", "angular_momentum_electromagnetic_erg",
    "angular_momentum_total_erg", "magnetization", "field_phi_over_r",
)
MATCHED_NAMES = (
    "resolution_2x", "M_NS_1p2", "M_NS_1p8", "M_NS_2p0",
    "RHO_IN_1e10", "RHO_IN_1e11", "B_SURF_1e13", "B_SURF_1e15",
)
ROTATION_NAMES = {
    2.0: "P_ROT_MS_2",
    4.0: "P_ROT_MS_4",
    8.0: "resolution_2x",
    16.0: "P_ROT_MS_16",
}
COLORS = ("#2166ac", "#d6604d", "#1b9e77", "#984ea3", "#e6ab02")


plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "mathtext.fontset": "dejavuserif",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linewidth": 0.6,
    "savefig.facecolor": "white",
})


@dataclass
class RunSummary:
    suite: str
    name: str
    run_dir: Path
    state: str
    description: str
    parameters: dict[str, float]
    radius_km: np.ndarray | None = None
    times_ms: np.ndarray | None = None
    final: dict[str, np.ndarray] | None = None
    late_median: dict[str, np.ndarray] | None = None
    late_p16: dict[str, np.ndarray] | None = None
    late_p84: dict[str, np.ndarray] | None = None
    histories: dict[str, dict[str, np.ndarray]] | None = None
    crossings: dict[str, np.ndarray] | None = None
    final_crossings: dict[str, list[float]] | None = None
    metrics: dict[str, float | bool | str] | None = None
    snapshot_count: int = 0
    late_snapshot_count: int = 0
    error: str | None = None

    @property
    def complete(self) -> bool:
        return self.state == "complete" and self.final is not None

    @property
    def label(self) -> str:
        return f"{self.suite}: {self.name}"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def discover_runs(suite_dir: Path, suite: str) -> list[RunSummary]:
    runs = []
    for experiment_path in sorted(suite_dir.glob("*/experiment.json")):
        run_dir = experiment_path.parent
        experiment = _read_json(experiment_path)
        status = _read_json(run_dir / "run_status.json")
        metadata = experiment.get("experiment", {})
        state = status.get("state", "prepared")
        if (run_dir / "run_complete.json").exists():
            state = "complete"
        runs.append(RunSummary(
            suite=suite,
            name=run_dir.name,
            run_dir=run_dir,
            state=state,
            description=metadata.get("description", ""),
            parameters={},
        ))
    return runs


def _robust_relative_span(values: np.ndarray) -> float:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan")
    low, median, high = np.percentile(values, (16.0, 50.0, 84.0))
    scale = max(abs(float(median)), float(np.max(np.abs(values))) * 1.0e-12)
    return float((high - low) / scale) if scale else 0.0


def _radial_relative_span(values: np.ndarray) -> float:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan")
    low, median, high = np.percentile(values, (5.0, 50.0, 95.0))
    scale = max(abs(float(median)), float(np.max(np.abs(values))) * 1.0e-12)
    return float((high - low) / scale) if scale else 0.0


def analyze_run(run: RunSummary, late_fraction: float) -> RunSummary:
    if run.state != "complete":
        run.error = f"run state is {run.state}"
        catalog_path = run.run_dir / "dbl.out"
        if catalog_path.exists():
            run.snapshot_count = len(read_catalog(catalog_path))
        return run

    physics, units = read_definitions(run.run_dir / "definitions.h")
    if physics != "RMHD":
        raise ValueError(f"{run.run_dir} is {physics}, expected RMHD")
    run.parameters = read_parameters(run.run_dir / "pluto.ini")
    radius_code = read_grid(run.run_dir / "grid.out")
    radius_km = radius_code * units["UNIT_LENGTH"] / 1.0e5
    catalog = read_catalog(run.run_dir / "dbl.out")
    numbers = sorted(catalog)
    if not numbers:
        raise ValueError(f"No completed snapshots in {run.run_dir}")
    times_code = np.asarray([catalog[number]["time"] for number in numbers])
    if np.any(np.diff(times_code) <= 0.0):
        raise ValueError(f"Non-increasing snapshot times in {run.run_dir}")
    times_ms = times_code * units["UNIT_LENGTH"] / units["UNIT_VELOCITY"] * 1.0e3
    late_start = max(0, len(numbers) - max(2, int(np.ceil(late_fraction * len(numbers)))))
    late_profiles = {key: [] for key in PROFILE_KEYS}
    sample_labels = [f"{radius:g} km" for radius in SAMPLE_RADII] + ["outer"]
    histories = {
        key: {label: np.empty(len(numbers)) for label in sample_labels}
        for key in PROFILE_KEYS
    }
    crossing_histories = {
        "sonic": np.full(len(numbers), np.nan),
        "alfven": np.full(len(numbers), np.nan),
        "sonic_count": np.zeros(len(numbers), dtype=int),
        "alfven_count": np.zeros(len(numbers), dtype=int),
    }
    final = None
    final_crossings = {"sonic": [], "alfven": []}
    active = radius_km > SURFACE_RADIUS_KM
    if np.count_nonzero(active) < 2:
        raise ValueError(f"No active radial domain outside {SURFACE_RADIUS_KM:g} km")

    for index, number in enumerate(numbers):
        snapshot = read_snapshot(run.run_dir, number, radius_code, catalog)
        profile = derive_rmhd_profiles(snapshot, radius_code, units, run.parameters)
        for key in PROFILE_KEYS:
            for requested, label in zip(SAMPLE_RADII, sample_labels[:-1]):
                histories[key][label][index] = interpolate_profile(
                    radius_km[active], profile[key][active], requested
                )
            histories[key]["outer"][index] = profile[key][active][-1]
            if index >= late_start:
                late_profiles[key].append(np.asarray(profile[key]))
        roots = outward_crossing_radii(radius_km, profile, active)
        for kind in ("sonic", "alfven"):
            crossing_histories[f"{kind}_count"][index] = len(roots[kind])
            if roots[kind]:
                crossing_histories[kind][index] = roots[kind][0]
        if index == len(numbers) - 1:
            final = {key: np.asarray(profile[key]).copy() for key in PROFILE_KEYS}
            final_crossings = roots

    late_arrays = {key: np.asarray(values) for key, values in late_profiles.items()}
    late_median = {key: np.nanmedian(values, axis=0) for key, values in late_arrays.items()}
    late_p16 = {key: np.nanpercentile(values, 16.0, axis=0) for key, values in late_arrays.items()}
    late_p84 = {key: np.nanpercentile(values, 84.0, axis=0) for key, values in late_arrays.items()}

    radial = active & (radius_km >= 20.0)
    late_slice = slice(late_start, len(numbers))
    metrics: dict[str, float | bool | str] = {
        "late_fraction": late_fraction,
        "temporal_vr_outer_relative_span": _robust_relative_span(
            histories["velocity_c"]["outer"][late_slice]
        ),
        "temporal_mdot_outer_relative_span": _robust_relative_span(
            histories["mdot_g_s"]["outer"][late_slice]
        ),
        "radial_mdot_relative_span": _radial_relative_span(
            late_median["mdot_g_s"][radial]
        ),
    }
    rotating = "P_ROT_MS" in run.parameters
    if rotating:
        metrics.update(
            temporal_jdot_outer_relative_span=_robust_relative_span(
                histories["angular_momentum_total_erg"]["outer"][late_slice]
            ),
            radial_jdot_relative_span=_radial_relative_span(
                late_median["angular_momentum_total_erg"][radial]
            ),
        )
    else:
        metrics.update(
            temporal_jdot_outer_relative_span=float("nan"),
            radial_jdot_relative_span=float("nan"),
        )
    checks = [
        metrics["temporal_vr_outer_relative_span"] <= 0.05,
        metrics["temporal_mdot_outer_relative_span"] <= 0.05,
        metrics["radial_mdot_relative_span"] <= 0.10,
    ]
    if rotating:
        checks.extend([
            metrics["temporal_jdot_outer_relative_span"] <= 0.10,
            metrics["radial_jdot_relative_span"] <= 0.10,
        ])
    metrics["steady"] = bool(all(checks))
    metrics["steady_label"] = "steady" if metrics["steady"] else "not steady"

    run.radius_km = radius_km
    run.times_ms = times_ms
    run.final = final
    run.late_median = late_median
    run.late_p16 = late_p16
    run.late_p84 = late_p84
    run.histories = histories
    run.crossings = crossing_histories
    run.final_crossings = final_crossings
    run.metrics = metrics
    run.snapshot_count = len(numbers)
    run.late_snapshot_count = len(numbers) - late_start
    return run


def _axis_style(axis, *, xlog=True, symlog=False, log=False):
    if xlog:
        axis.set_xscale("log")
    if symlog:
        values = np.concatenate([
            np.ravel(line.get_ydata()) for line in axis.lines
            if np.size(line.get_ydata())
        ])
        values = values[np.isfinite(values)]
        maximum = float(np.max(np.abs(values))) if values.size else 0.0
        nonzero = np.abs(values[values != 0.0])
        if maximum == 0.0 or not nonzero.size:
            linthresh = 1.0
            axis.set_ylim(-1.0, 1.0)
        else:
            linthresh = max(maximum * 1.0e-8, float(np.percentile(nonzero, 1.0)) * 0.1)
            low = min(float(values.min()) * 1.15, -0.5 * linthresh)
            high = max(float(values.max()) * 1.15, 0.5 * linthresh)
            axis.set_ylim(low, high)
        axis.set_yscale("symlog", linthresh=linthresh)
    elif log:
        axis.set_yscale("log")
    axis.tick_params(which="both", direction="in", top=True, right=True)
    axis.axvline(SURFACE_RADIUS_KM, color="0.5", lw=0.8, ls=":")


def _plot_profile(axis, run: RunSummary, key: str, color: str, label: str):
    radius = run.radius_km
    axis.fill_between(
        radius, run.late_p16[key], run.late_p84[key], color=color, alpha=0.11,
        linewidth=0,
    )
    axis.plot(radius, run.late_median[key], color=color, lw=1.7, label=label)
    axis.plot(radius, run.final[key], color=color, lw=1.0, ls="--", alpha=0.9)


def _save_single(fig, output_dir: Path, stem: str) -> Path:
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pdf


def plot_rotation_profiles(runs: list[RunSummary], output_dir: Path) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(11.4, 11.0), sharex=True)
    panels = (
        ("velocity_c", r"$v_r/c$", False, False),
        ("velocity_phi_c", r"$v_\phi/c$", True, False),
        ("sound_speed_c", r"$c_s/c$", False, False),
        ("alfven_speed_c", r"$v_A/c$", False, False),
        ("magnetization", r"$\sigma=b^2/(\rho h)$", False, True),
        ("field_phi_over_r", r"$B_\phi/B_r$", True, False),
    )
    for axis, (key, ylabel, symlog, log) in zip(axes.ravel(), panels):
        for color, run in zip(COLORS, runs):
            _plot_profile(axis, run, key, color, f"{run.parameters['P_ROT_MS']:g} ms")
        _axis_style(axis, symlog=symlog, log=log)
        axis.set_ylabel(ylabel)
    for axis in axes[-1]:
        axis.set_xlabel(r"radius [km]")
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=9)
    fig.legend(
        handles=[
            Line2D([], [], color="0.2", lw=1.7, label="late median"),
            Line2D([], [], color="0.2", lw=1.0, ls="--", label="final"),
        ],
        loc="upper center", bbox_to_anchor=(0.70, 0.995), frameon=False, ncol=2,
    )
    fig.suptitle("Rotating RMHD period scan: late-time profiles", y=1.02)
    fig.tight_layout()
    return _save_single(fig, output_dir, "rotation_period_profiles")


def plot_rotation_fluxes(runs: list[RunSummary], output_dir: Path) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(11.4, 11.0), sharex=True)
    panels = (
        ("mdot_g_s", r"$\dot M$ [g s$^{-1}$]"),
        ("kinetic_power_erg_s", r"$\dot E_{\rm kin}$ [erg s$^{-1}$]"),
        ("poynting_power_erg_s", r"$\dot E_{\rm EM}$ [erg s$^{-1}$]"),
        ("angular_momentum_fluid_erg", r"$\dot J_{\rm fluid}$ [erg]"),
        ("angular_momentum_electromagnetic_erg", r"$\dot J_{\rm EM}$ [erg]"),
        ("angular_momentum_total_erg", r"$\dot J_{\rm total}$ [erg]"),
    )
    for axis, (key, ylabel) in zip(axes.ravel(), panels):
        for color, run in zip(COLORS, runs):
            _plot_profile(axis, run, key, color, f"{run.parameters['P_ROT_MS']:g} ms")
        _axis_style(axis, symlog=True)
        axis.set_ylabel(ylabel)
    for axis in axes[-1]:
        axis.set_xlabel("radius [km]")
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=9)
    fig.suptitle(
        "Rotating RMHD period scan: mass, energy, and angular-momentum fluxes\n"
        "Energy panels show kinetic and Poynting terms separately; their sum is not a conserved total energy flux",
        y=1.03, fontsize=12,
    )
    fig.tight_layout()
    return _save_single(fig, output_dir, "rotation_period_fluxes")


def _pair_title(nonrot: RunSummary, rot: RunSummary) -> str:
    description = nonrot.description.rstrip(".")
    return f"{nonrot.name}: {description} (rotating member P = {rot.parameters['P_ROT_MS']:g} ms)"


def _save_multipage(
    output_dir: Path,
    stem: str,
    page_builder,
    pairs: list[tuple[RunSummary, RunSummary]],
) -> Path:
    pdf_path = output_dir / f"{stem}.pdf"
    with PdfPages(pdf_path) as pdf:
        for page, pair in enumerate(pairs, start=1):
            fig = page_builder(*pair)
            pdf.savefig(fig, bbox_inches="tight")
            fig.savefig(
                output_dir / f"{stem}_page_{page:02d}.png",
                dpi=200, bbox_inches="tight",
            )
            plt.close(fig)
    return pdf_path


def _matched_page(nonrot: RunSummary, rot: RunSummary, panels, title: str):
    fig, axes = plt.subplots(3, 2, figsize=(11.4, 10.4), sharex=True)
    for axis, (key, ylabel, symlog, log) in zip(axes.ravel(), panels):
        _plot_profile(axis, nonrot, key, COLORS[0], "nonrotating")
        _plot_profile(axis, rot, key, COLORS[1], f"rotating ({rot.parameters['P_ROT_MS']:g} ms)")
        _axis_style(axis, symlog=symlog, log=log)
        axis.set_ylabel(ylabel)
    for axis in axes[-1]:
        axis.set_xlabel("radius [km]")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(title, y=1.01, fontsize=12)
    fig.tight_layout()
    return fig


def plot_matched_profiles(pairs, output_dir: Path) -> Path:
    panels = (
        ("velocity_c", r"$v_r/c$", False, False),
        ("velocity_phi_c", r"$v_\phi/c$", True, False),
        ("sound_speed_c", r"$c_s/c$", False, False),
        ("alfven_speed_c", r"$v_A/c$", False, False),
        ("magnetization", r"$\sigma$", False, True),
        ("field_phi_over_r", r"$B_\phi/B_r$", True, False),
    )
    return _save_multipage(
        output_dir, "matched_nonrotating_rotating_profiles",
        lambda n, r: _matched_page(n, r, panels, _pair_title(n, r)), pairs,
    )


def plot_matched_fluxes(pairs, output_dir: Path) -> Path:
    panels = tuple((key, ylabel, True, False) for key, ylabel in (
        ("mdot_g_s", r"$\dot M$ [g s$^{-1}$]"),
        ("kinetic_power_erg_s", r"$\dot E_{\rm kin}$ [erg s$^{-1}$]"),
        ("poynting_power_erg_s", r"$\dot E_{\rm EM}$ [erg s$^{-1}$]"),
        ("angular_momentum_fluid_erg", r"$\dot J_{\rm fluid}$ [erg]"),
        ("angular_momentum_electromagnetic_erg", r"$\dot J_{\rm EM}$ [erg]"),
        ("angular_momentum_total_erg", r"$\dot J_{\rm total}$ [erg]"),
    ))
    return _save_multipage(
        output_dir, "matched_nonrotating_rotating_fluxes",
        lambda n, r: _matched_page(
            n, r, panels, _pair_title(n, r)
            + "\nEnergy terms are diagnostics, not a complete conserved energy flux",
        ), pairs,
    )


def plot_late_histories(runs: list[RunSummary], output_dir: Path) -> Path:
    fig, axes = plt.subplots(4, 2, figsize=(11.8, 13.0), sharex=True)
    panels = (
        ("velocity_c", "outer", r"outer $v_r/c$", False),
        ("velocity_phi_c", "outer", r"outer $v_\phi/c$", True),
        ("mdot_g_s", "outer", r"outer $\dot M$ [g s$^{-1}$]", True),
        ("poynting_power_erg_s", "outer", r"outer $\dot E_{\rm EM}$ [erg s$^{-1}$]", True),
        ("angular_momentum_total_erg", "outer", r"outer $\dot J_{\rm total}$ [erg]", True),
        ("magnetization", "outer", r"outer $\sigma$", False),
    )
    for axis, (key, sample, ylabel, symlog) in zip(axes.ravel()[:6], panels):
        for color, run in zip(COLORS, runs):
            late_start = len(run.times_ms) - run.late_snapshot_count
            axis.plot(
                run.times_ms[late_start:], run.histories[key][sample][late_start:],
                color=color, lw=1.5, label=f"{run.parameters['P_ROT_MS']:g} ms",
            )
        axis.set_ylabel(ylabel)
        axis.tick_params(which="both", direction="in", top=True, right=True)
        if symlog:
            axis.set_yscale("symlog", linthresh=1.0e-4)
        elif key == "magnetization":
            axis.set_yscale("log")
    for column, kind in enumerate(("sonic", "alfven")):
        axis = axes[3, column]
        for color, run in zip(COLORS, runs):
            late_start = len(run.times_ms) - run.late_snapshot_count
            axis.plot(
                run.times_ms[late_start:], run.crossings[kind][late_start:],
                color=color, lw=1.5, label=f"{run.parameters['P_ROT_MS']:g} ms",
            )
        axis.set_ylabel(f"first outward {kind} crossing [km]")
        axis.set_yscale("log")
        axis.tick_params(which="both", direction="in", top=True, right=True)
    for axis in axes[-1]:
        axis.set_xlabel("simulation time [ms]")
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=9)
    fig.suptitle("Rotation scan: last 20% of saved snapshots", y=1.01)
    fig.tight_layout()
    return _save_single(fig, output_dir, "late_time_histories")


def plot_steadiness(runs: list[RunSummary], output_dir: Path) -> Path:
    metric_specs = (
        ("temporal_vr_outer_relative_span", 0.05, r"temporal $v_r$"),
        ("temporal_mdot_outer_relative_span", 0.05, r"temporal $\dot M$"),
        ("radial_mdot_relative_span", 0.10, r"radial $\dot M$"),
        ("temporal_jdot_outer_relative_span", 0.10, r"temporal $\dot J$"),
        ("radial_jdot_relative_span", 0.10, r"radial $\dot J$"),
    )
    matrix = np.full((len(runs), len(metric_specs)), np.nan)
    labels = []
    for row, run in enumerate(runs):
        labels.append(run.label)
        if run.metrics:
            for column, (key, threshold, _) in enumerate(metric_specs):
                value = run.metrics.get(key, np.nan)
                matrix[row, column] = value / threshold if np.isfinite(value) else np.nan
    fig, axis = plt.subplots(figsize=(10.6, max(7.2, 0.38 * len(runs) + 2.5)))
    masked = np.ma.masked_invalid(matrix)
    image = axis.imshow(masked, cmap="RdYlGn_r", vmin=0.0, vmax=2.0, aspect="auto")
    image.cmap.set_bad("#d9d9d9")
    axis.set_xticks(range(len(metric_specs)), [item[2] for item in metric_specs], rotation=28, ha="right")
    axis.set_yticks(range(len(runs)), labels)
    for row, run in enumerate(runs):
        for column in range(len(metric_specs)):
            value = matrix[row, column]
            if not np.isfinite(value):
                text = "n/a"
            elif abs(value) >= 1.0e4:
                text = f"{value:.1e}x"
            else:
                text = f"{value:.2f}x"
            axis.text(column, row, text, ha="center", va="center", fontsize=7.5)
        state = "failed" if not run.complete else run.metrics["steady_label"]
        axis.text(len(metric_specs) - 0.42, row, f"  {state}", ha="left", va="center", fontsize=8,
                  color="#9b2226" if state != "steady" else "#006d2c")
    colorbar = fig.colorbar(image, ax=axis, pad=0.17, fraction=0.035)
    colorbar.set_label("metric / adopted threshold (<= 1 passes)")
    axis.set_title(
        "RMHD suite steadiness audit\n"
        "late-time temporal stability plus radial flux consistency; gray = not applicable/failed"
    )
    fig.tight_layout()
    return _save_single(fig, output_dir, "steadiness_matrix")


def plot_crossings(runs: list[RunSummary], output_dir: Path) -> Path:
    completed = [run for run in runs if run.complete]
    labels = [run.label for run in completed]
    y = np.arange(len(completed))
    fig, axes = plt.subplots(1, 2, figsize=(13.0, max(7.0, 0.36 * len(completed) + 2.2)), sharey=True)
    for axis, kind, color in zip(axes, ("sonic", "alfven"), ("#d97706", "#6d28d9")):
        medians, lows, highs, finals = [], [], [], []
        for run in completed:
            late = run.crossings[kind][-run.late_snapshot_count:]
            finite = late[np.isfinite(late)]
            if finite.size:
                low, median, high = np.percentile(finite, (16, 50, 84))
            else:
                low = median = high = np.nan
            medians.append(median)
            lows.append(low)
            highs.append(high)
            finals.append(run.final_crossings[kind][0] if run.final_crossings[kind] else np.nan)
        medians, lows, highs, finals = map(np.asarray, (medians, lows, highs, finals))
        axis.errorbar(
            medians, y, xerr=np.vstack((medians - lows, highs - medians)),
            fmt="o", color=color, ecolor=color, capsize=2.5, label="late median (16-84%)",
        )
        axis.scatter(finals, y, marker="|", s=120, color="black", label="final")
        axis.set_xscale("log")
        axis.set_xlabel(f"first outward {kind} speed crossing [km]")
        axis.grid(True, which="both", alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Sonic and Alfvén speed crossings across completed RMHD runs\n"
        "These are v_r=c_s and v_r=v_A locations, not full RMHD characteristic critical surfaces",
        fontsize=12,
    )
    fig.tight_layout()
    return _save_single(fig, output_dir, "crossing_radii")


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_manifest(path: Path, runs: list[RunSummary], outputs: Iterable[Path], late_fraction: float):
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "PLUTO dbl snapshots read directly; no CSV intermediates",
        "method": {
            "late_time_fraction": late_fraction,
            "late_profile": "median with 16th-84th percentile envelope",
            "final_profile": "last completed snapshot",
            "samples_km": list(SAMPLE_RADII) + ["outermost active cell"],
            "interpolation": "linear interpolation of already-derived quantities",
            "steadiness": {
                "temporal_vr_outer_relative_span_max": 0.05,
                "temporal_mdot_outer_relative_span_max": 0.05,
                "radial_mdot_relative_span_max": 0.10,
                "temporal_jdot_outer_relative_span_max_rotating": 0.10,
                "radial_jdot_relative_span_max_rotating": 0.10,
                "note": "Operational diagnostic thresholds, not a proof of an exact steady solution",
            },
            "energy_caveat": "Kinetic and Poynting powers are plotted separately; their sum omits enthalpy and gravity and is not treated as a conserved total energy flux",
            "crossing_caveat": "Crossings are v_r=c_s and v_r=v_A speed crossings, not full RMHD characteristic critical surfaces",
        },
        "outputs": [str(output.resolve()) for output in outputs],
        "runs": [],
    }
    for run in runs:
        payload["runs"].append({
            "suite": run.suite,
            "name": run.name,
            "run_dir": str(run.run_dir.resolve()),
            "state": run.state,
            "description": run.description,
            "parameters": run.parameters,
            "snapshot_count": run.snapshot_count,
            "late_snapshot_count": run.late_snapshot_count,
            "metrics": run.metrics,
            "final_crossings_km": run.final_crossings,
            "error": run.error,
        })
    path.write_text(json.dumps(_json_safe(payload), indent=2) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--nonrotating-dir", type=Path, default=DEFAULT_NONROTATING)
    parser.add_argument("--rotating-dir", type=Path, default=DEFAULT_ROTATING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--late-fraction", type=float, default=0.20)
    args = parser.parse_args(argv)
    if not 0.0 < args.late_fraction <= 1.0:
        parser.error("--late-fraction must be in (0, 1]")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = (
        discover_runs(args.nonrotating_dir.resolve(), "nonrotating")
        + discover_runs(args.rotating_dir.resolve(), "rotating")
    )
    for index, run in enumerate(runs, start=1):
        print(f"[{index:02d}/{len(runs):02d}] {run.label}: {run.state}", flush=True)
        try:
            analyze_run(run, args.late_fraction)
        except Exception as error:
            run.state = "analysis_failed"
            run.error = str(error)
            print(f"  analysis failed: {error}", flush=True)

    lookup = {(run.suite, run.name): run for run in runs}
    rotation_runs = [lookup[("rotating", ROTATION_NAMES[period])] for period in ROTATION_NAMES]
    if not all(run.complete for run in rotation_runs):
        missing = [run.label for run in rotation_runs if not run.complete]
        raise RuntimeError(f"Required rotation runs are unavailable: {missing}")
    pairs = [
        (lookup[("nonrotating", name)], lookup[("rotating", name)])
        for name in MATCHED_NAMES
    ]
    if not all(left.complete and right.complete for left, right in pairs):
        raise RuntimeError("One or more matched nonrotating/rotating runs are unavailable")

    outputs = [
        plot_rotation_profiles(rotation_runs, args.output_dir),
        plot_rotation_fluxes(rotation_runs, args.output_dir),
        plot_matched_profiles(pairs, args.output_dir),
        plot_matched_fluxes(pairs, args.output_dir),
        plot_late_histories(rotation_runs, args.output_dir),
        plot_steadiness(runs, args.output_dir),
        plot_crossings(runs, args.output_dir),
    ]
    manifest = args.output_dir / "analysis_manifest.json"
    write_manifest(manifest, runs, outputs, args.late_fraction)
    print("\nSteadiness summary")
    for run in runs:
        state = run.metrics["steady_label"] if run.metrics else run.state
        print(f"  {run.label:<42} {state}")
    print(f"\nSaved {len(outputs)} PDFs, PNG copies/pages, and {manifest}")


if __name__ == "__main__":
    main()
