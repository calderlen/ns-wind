"""Animate or export saved 1D relativistic-magnetohydrodynamic wind profiles."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.lines import Line2D
import numpy as np

from wind_common import (
    C_CGS, REPO_DIR, first_outward_zero_crossing, read_definitions, read_catalog,
    read_grid, read_parameters, read_snapshot,
)
from rmhd_diagnostics import derive_rmhd_profiles
from plot_helpers import (
    padded_linear_limits, padded_log_limits, signed_log_limits,
    scientific_latex, sample_histories,
)


DEFAULT_RUN_DIR = REPO_DIR / "problems" / "rmhd_mpi"
DEFAULT_OUTPUT_NAME = "wind_evolution_rmhd.gif"
SURFACE_RADIUS_KM = 12.0
M_SUN_G = 1.9885e33
KELVIN_PER_MEV = 1.16045e10
COMPONENT_COLORS = {
    "velocity_cm_s": "#0072b2",
    "velocity_phi_cm_s": "#009e73",
    "sound_speed_cm_s": "#d97706",
    "alfven_speed_cm_s": "#7c3aed",
    "escape_velocity_cm_s": "#c62828",
    "field_r_gauss": "#0072b2",
    "field_phi_gauss": "#7c3aed",
    "fluid_bernoulli_c2": "#0072b2",
    "electromagnetic_bernoulli_c2": "#7c3aed",
    "bernoulli_c2": "#009e73",
    "kinetic_power_erg_s": "#0072b2",
    "thermal_enthalpy_power_erg_s": "#d97706",
    "poynting_power_erg_s": "#7c3aed",
    "gravitational_power_erg_s": "#c62828",
    "total_power_erg_s": "#009e73",
    "angular_momentum_fluid_erg": "#0072b2",
    "angular_momentum_electromagnetic_erg": "#7c3aed",
    "angular_momentum_total_erg": "#009e73",
    "specific_angular_momentum_fluid_cm2_s": "#0072b2",
    "specific_angular_momentum_electromagnetic_cm2_s": "#7c3aed",
    "specific_angular_momentum_total_cm2_s": "#009e73",
}

plt.rcParams.update(
    {
        "font.family": "CMU Bright",
        "mathtext.fontset": "custom",
        "mathtext.rm": "CMU Bright",
        "mathtext.it": "CMU Bright:style=oblique",
        "mathtext.bf": "CMU Bright:weight=semibold",
        "mathtext.sf": "CMU Bright",
        "mathtext.fallback": "cm",
    }
)


def _time_limits(times: np.ndarray) -> tuple[float, float]:
    if times.size > 1 and times[-1] > times[0]:
        return float(times[0]), float(times[-1])
    center = float(times[0])
    margin = max(abs(center) * 0.05, 1.0)
    return center - margin, center + margin


def animate_rmhd_profiles(
    argv: list[str] | None = None,
    *,
    default_run_dir: Path = DEFAULT_RUN_DIR,
    default_output_name: str = DEFAULT_OUTPUT_NAME,
) -> Path:
    """Build an RMHD preview, final-profile PDF, or GIF and return its path."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=default_run_dir)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <run-dir>/plots/<animation-name>.gif; --pdf uses .pdf)",
    )
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Animate every Nth snapshot while always including the latest one.",
    )
    export = parser.add_mutually_exclusive_group()
    export.add_argument(
        "--preview",
        action="store_true",
        help="Save the latest-frame layout as a PNG instead of encoding a GIF.",
    )
    export.add_argument(
        "--pdf",
        action="store_true",
        help="Save final radial profiles and time histories as a vector PDF.",
    )
    args = parser.parse_args(argv)

    if args.fps <= 0.0:
        parser.error("--fps must be positive")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    if args.stride <= 0:
        parser.error("--stride must be positive")

    run_dir = args.run_dir.resolve()
    output = (args.output or run_dir / "plots" / default_output_name).resolve()
    physics, units = read_definitions(run_dir / "definitions.h")
    if physics != "RMHD":
        raise ValueError(f"Expected an RMHD run; found {physics}")
    parameters = read_parameters(run_dir / "pluto.ini")
    radius_km = read_grid(run_dir / "grid.out")
    catalog = read_catalog(run_dir / "dbl.out")
    all_numbers = sorted(catalog)
    if not all_numbers:
        raise ValueError(f"No completed snapshots listed in {run_dir / 'dbl.out'}")

    snapshot_numbers = all_numbers[::args.stride]
    if snapshot_numbers[-1] != all_numbers[-1]:
        snapshot_numbers.append(all_numbers[-1])

    profiles: list[dict[str, np.ndarray]] = []
    times_s: list[float] = []
    time_unit = units["UNIT_LENGTH"] / units["UNIT_VELOCITY"]
    for number in snapshot_numbers:
        snapshot = read_snapshot(run_dir, number, radius_km, catalog)
        profile = derive_rmhd_profiles(snapshot, radius_km, units, parameters)
        # Display conversions only; the numerical diagnostics retain their units.
        profile["mdot_msun_s"] = profile["mdot_g_s"] / M_SUN_G
        profile["temperature_mev"] = profile["temperature"] / KELVIN_PER_MEV
        for component in (
            "velocity", "velocity_phi", "sound_speed", "alfven_speed", "escape_velocity",
        ):
            profile[f"{component}_cm_s"] = profile[f"{component}_c"] * C_CGS
        profiles.append(profile)
        times_s.append(float(snapshot["time"]) * time_unit)

    initial = profiles[0]
    times = np.asarray(times_s)
    active = radius_km > SURFACE_RADIUS_KM
    if not np.any(active):
        raise ValueError("The radial grid has no active cells outside 12 km")
    surface_index = int(np.flatnonzero(active)[0])

    density_all = [profile["density"] for profile in profiles]
    pressure_all = [profile["pressure"] for profile in profiles]
    temperature_all = [profile["temperature_mev"] for profile in profiles]
    gamma_all = [profile["lorentz_gamma"] for profile in profiles]
    plasma_beta_all = [profile["plasma_beta"] for profile in profiles]
    specific_angular_momentum_keys = (
        "specific_angular_momentum_fluid_cm2_s",
        "specific_angular_momentum_electromagnetic_cm2_s",
        "specific_angular_momentum_total_cm2_s",
    )
    specific_angular_momentum_all = [
        profile[key] for profile in profiles for key in specific_angular_momentum_keys
    ]
    velocity_all = [profile["velocity_cm_s"] for profile in profiles]
    velocity_phi_all = [profile["velocity_phi_cm_s"] for profile in profiles]
    sound_all = [profile["sound_speed_cm_s"] for profile in profiles]
    alfven_all = [profile["alfven_speed_cm_s"] for profile in profiles]
    field_all = [
        profile[key]
        for profile in profiles
        for key in ("field_r_gauss", "field_phi_gauss")
    ]
    bernoulli_all = [
        profile[key]
        for profile in profiles
        for key in (
            "fluid_bernoulli_c2", "electromagnetic_bernoulli_c2", "bernoulli_c2",
        )
    ]
    mdot_all = [profile["mdot_msun_s"] for profile in profiles]
    power_all = [
        profile[key]
        for profile in profiles
        for key in (
            "kinetic_power_erg_s", "thermal_enthalpy_power_erg_s",
            "poynting_power_erg_s", "gravitational_power_erg_s", "total_power_erg_s",
        )
    ]
    magnetization_all = [profile["magnetization"] for profile in profiles]
    angular_momentum_all = [
        profile[key]
        for profile in profiles
        for key in (
            "angular_momentum_fluid_erg",
            "angular_momentum_electromagnetic_erg",
            "angular_momentum_total_erg",
        )
    ]

    surface_vr = np.asarray(
        [profile["velocity_cm_s"][surface_index] for profile in profiles]
    )
    outer_vr = np.asarray([profile["velocity_cm_s"][-1] for profile in profiles])
    surface_vphi = np.asarray(
        [profile["velocity_phi_cm_s"][surface_index] for profile in profiles]
    )
    outer_vphi = np.asarray(
        [profile["velocity_phi_cm_s"][-1] for profile in profiles]
    )
    surface_bratio = np.asarray(
        [profile["field_phi_over_r"][surface_index] for profile in profiles]
    )
    outer_bratio = np.asarray(
        [profile["field_phi_over_r"][-1] for profile in profiles]
    )
    surface_sigma = np.asarray(
        [profile["magnetization"][surface_index] for profile in profiles]
    )
    outer_sigma = np.asarray(
        [profile["magnetization"][-1] for profile in profiles]
    )
    mdot_diagnostics = sample_histories(profiles, radius_km, "mdot_msun_s")
    power_diagnostics = sample_histories(profiles, radius_km, "total_power_erg_s")
    angular_momentum_diagnostics = sample_histories(
        profiles, radius_km, "angular_momentum_total_erg"
    )
    gamma_diagnostics = sample_histories(profiles, radius_km, "lorentz_gamma")
    specific_angular_momentum_diagnostics = sample_histories(
        profiles, radius_km, "specific_angular_momentum_total_cm2_s"
    )

    fig = plt.figure(figsize=(13.4, 19.8))
    grid = fig.add_gridspec(
        7, 3, hspace=0.18, wspace=0.36, width_ratios=(1.0, 1.0, 1.08)
    )
    axes = np.empty((5, 3), dtype=object)
    for row in range(5):
        for col in range(3):
            sharex = axes[0, col] if row else None
            if row == 0 and col == 1:
                speed_grid = grid[row, col].subgridspec(
                    3, 1, height_ratios=(0.13, 1.0, 0.20), hspace=0.06
                )
                speed_legend_axis = fig.add_subplot(
                    speed_grid[0], label="rmhd-speed-legend"
                )
                axes[row, col] = fig.add_subplot(
                    speed_grid[1], label="rmhd-speed-profile"
                )
                speed_status_axis = fig.add_subplot(
                    speed_grid[2], label="rmhd-speed-status"
                )
                speed_legend_axis.set_axis_off()
                speed_status_axis.set_axis_off()
            else:
                axes[row, col] = fig.add_subplot(grid[row, col], sharex=sharex)
    gamma_axis = fig.add_subplot(grid[5, 0], sharex=axes[0, 0])
    plasma_beta_axis = fig.add_subplot(grid[5, 1], sharex=axes[0, 1])
    gamma_history_axis = fig.add_subplot(grid[5, 2], sharex=axes[0, 2])
    specific_angular_momentum_axis = fig.add_subplot(grid[6, :2], sharex=axes[0, 0])
    specific_angular_momentum_history_axis = fig.add_subplot(grid[6, 2], sharex=axes[0, 2])
    all_axes = [
        *axes.ravel(), gamma_axis, plasma_beta_axis, gamma_history_axis,
        specific_angular_momentum_axis, specific_angular_momentum_history_axis,
    ]
    history_axes = [
        *axes[:, 2], gamma_history_axis, specific_angular_momentum_history_axis,
    ]
    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.055, top=0.95)

    initial_color = "0.62"
    current_color = "#1b6ca8"
    initial_style = {"color": initial_color, "lw": 1.3, "ls": "--"}
    current_style = {"color": current_color, "lw": 1.6}

    radial_lines = {}
    history_lines = []

    def plot_pair(axis, key):
        axis.plot(radius_km, initial[key], **initial_style)
        radial_lines[key], = axis.plot(radius_km, initial[key], **current_style)

    def plot_components(axis, components):
        for key, linestyle, width in components:
            radial_lines[key], = axis.plot(
                radius_km, initial[key], color=COMPONENT_COLORS[key],
                lw=width, ls=linestyle,
            )

    def plot_histories(axis, series, *, width=1.4):
        for history, color, linestyle, label in series:
            line, = axis.plot(
                times[:1], history[:1], color=color,
                lw=width, ls=linestyle, label=label,
            )
            history_lines.append((line, history))

    plot_pair(axes[0, 0], "density")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylim(*padded_log_limits(density_all))
    axes[0, 0].set_ylabel(r"$\rho\;[\mathrm{g\,cm^{-3}}]$")

    plot_pair(axes[1, 0], "pressure")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylim(*padded_log_limits(pressure_all))
    axes[1, 0].set_ylabel(r"$P\;[\mathrm{dyn\,cm^{-2}}]$")

    plot_pair(axes[2, 0], "temperature_mev")
    axes[2, 0].set_yscale("log")
    axes[2, 0].set_ylim(*padded_log_limits(temperature_all))
    axes[2, 0].set_ylabel(r"$k_B T\;[\mathrm{MeV}]$")

    field_low, field_high, field_linthresh = signed_log_limits(field_all)
    plot_components(axes[3, 0], (
        ("field_r_gauss", "-", 1.6),
        ("field_phi_gauss", "--", 1.6),
    ))
    axes[3, 0].axhline(0.0, color="black", lw=0.8)
    axes[3, 0].set_yscale("symlog", linthresh=field_linthresh)
    axes[3, 0].set_ylim(field_low, field_high)
    axes[3, 0].set_ylabel(r"$B\;[\mathrm{G}]$")
    axes[3, 0].legend(
        handles=[
            radial_lines["field_r_gauss"],
            radial_lines["field_phi_gauss"],
        ],
        labels=[r"$B_r$", r"$B_\phi$"],
        loc="best",
        fontsize=8,
    )

    plot_pair(axes[4, 0], "magnetization")
    axes[4, 0].set_yscale("log")
    axes[4, 0].set_ylim(*padded_log_limits(magnetization_all))
    axes[4, 0].set_ylabel(r"$\sigma$")

    speed_axis = axes[0, 1]
    plot_components(speed_axis, (
        ("velocity_cm_s", "-", 1.85),
        ("velocity_phi_cm_s", "--", 1.45),
        ("sound_speed_cm_s", ":", 1.45),
        ("alfven_speed_cm_s", (0, (3, 1, 1, 1)), 1.45),
    ))
    escape_line, = speed_axis.plot(
        radius_km, initial["escape_velocity_cm_s"],
        color=COMPONENT_COLORS["escape_velocity_cm_s"], lw=1.3, ls="-.", alpha=0.85,
    )
    speed_axis.axhline(0.0, color="black", lw=0.8)
    sonic_line, = speed_axis.plot(
        [], [], color="#d97706", lw=0.9, ls=":", alpha=0.6, visible=False
    )
    sonic_marker, = speed_axis.plot(
        [], [], marker="o", ms=5.0, mec="white", mew=0.6,
        color="#d97706", ls="none", visible=False,
    )
    speed_axis.set_ylim(
        *padded_linear_limits(
            velocity_all
            + velocity_phi_all
            + sound_all
            + alfven_all
            + [initial["escape_velocity_cm_s"]],
            include_zero=True,
        )
    )
    speed_axis.set_ylabel(r"$v\;[\mathrm{cm\,s^{-1}}]$")
    speed_legend_axis.legend(
        handles=[
            radial_lines["velocity_cm_s"],
            radial_lines["velocity_phi_cm_s"],
            radial_lines["sound_speed_cm_s"],
            radial_lines["alfven_speed_cm_s"],
            escape_line,
        ],
        labels=[r"$v_r$", r"$v_\phi$", r"$c_s$", r"$v_A$", r"$v_{\rm esc}$"],
        loc="center",
        fontsize=8,
        ncol=5, frameon=False, handlelength=1.7, handletextpad=0.4,
        columnspacing=0.8, borderaxespad=0.0,
    )
    sonic_text = speed_status_axis.text(
        0.0, 0.97, "", transform=speed_status_axis.transAxes,
        ha="left", va="top", fontsize=7.5, color="#b45309",
    )
    alfven_radius_line, = speed_axis.plot(
        [], [], color="#7c3aed", lw=0.9, ls=":", alpha=0.6, visible=False
    )
    alfven_radius_marker, = speed_axis.plot(
        [], [], marker="s", ms=4.8, mec="white", mew=0.6,
        color="#7c3aed", ls="none", visible=False,
    )
    alfven_radius_text = speed_status_axis.text(
        0.0, 0.44, "", transform=speed_status_axis.transAxes,
        ha="left", va="top", fontsize=7.5, color="#6d28d9",
    )

    plot_components(axes[1, 1], (
        ("fluid_bernoulli_c2", "--", 1.45),
        ("electromagnetic_bernoulli_c2", ":", 1.55),
        ("bernoulli_c2", "-", 2.05),
    ))
    axes[1, 1].axhline(0.0, color="black", lw=0.8)
    axes[1, 1].set_ylim(*padded_linear_limits(bernoulli_all, include_zero=True))
    axes[1, 1].set_ylabel(r"$\mathrm{Be}\;[c^2]$")
    axes[1, 1].legend(
        handles=[
            radial_lines["fluid_bernoulli_c2"],
            radial_lines["electromagnetic_bernoulli_c2"],
            radial_lines["bernoulli_c2"],
        ],
        labels=["fluid", "EM", "total"],
        loc="best", fontsize=8, framealpha=0.95,
    )

    mdot_low, mdot_high, mdot_linthresh = signed_log_limits(mdot_all)
    plot_pair(axes[2, 1], "mdot_msun_s")
    axes[2, 1].axhline(0.0, color="black", lw=0.8)
    axes[2, 1].set_yscale("symlog", linthresh=mdot_linthresh)
    axes[2, 1].set_ylim(mdot_low, mdot_high)
    axes[2, 1].set_ylabel(r"$\dot{M}\;[M_\odot\,\mathrm{s^{-1}}]$")

    power_low, power_high, power_linthresh = signed_log_limits(power_all)
    power_axis = axes[3, 1]
    plot_components(power_axis, (
        ("kinetic_power_erg_s", "--", 1.45),
        ("thermal_enthalpy_power_erg_s", "-.", 1.45),
        ("poynting_power_erg_s", ":", 1.55),
        ("gravitational_power_erg_s", "-.", 1.45),
        ("total_power_erg_s", "-", 2.05),
    ))
    power_axis.axhline(0.0, color="black", lw=0.8)
    power_axis.set_yscale("symlog", linthresh=power_linthresh)
    power_axis.set_ylim(power_low, power_high)
    power_axis.set_ylabel(r"$\dot{E}\;[\mathrm{erg\,s^{-1}}]$")
    power_axis.legend(
        handles=[
            radial_lines["kinetic_power_erg_s"],
            radial_lines["thermal_enthalpy_power_erg_s"],
            radial_lines["poynting_power_erg_s"],
            radial_lines["gravitational_power_erg_s"],
            radial_lines["total_power_erg_s"],
        ],
        labels=["kinetic", "thermal/enthalpy", "EM", "gravity", "total"],
        loc="lower right", fontsize=8, ncol=2, framealpha=0.95,
    )

    angular_momentum_low, angular_momentum_high, angular_momentum_linthresh = (
        signed_log_limits(angular_momentum_all)
    )
    angular_momentum_axis = axes[4, 1]
    plot_components(angular_momentum_axis, (
        ("angular_momentum_fluid_erg", "--", 1.45),
        ("angular_momentum_electromagnetic_erg", ":", 1.55),
        ("angular_momentum_total_erg", "-", 2.05),
    ))
    angular_momentum_axis.axhline(0.0, color="black", lw=0.8)
    angular_momentum_axis.set_yscale(
        "symlog", linthresh=angular_momentum_linthresh
    )
    angular_momentum_axis.set_ylim(
        angular_momentum_low, angular_momentum_high
    )
    angular_momentum_axis.set_ylabel(
        r"$\dot{J}\;[\mathrm{erg}]$"
    )
    angular_momentum_axis.legend(
        handles=[
            radial_lines["angular_momentum_fluid_erg"],
            radial_lines["angular_momentum_electromagnetic_erg"],
            radial_lines["angular_momentum_total_erg"],
        ],
        labels=["fluid", "EM", "total"],
        loc="lower right",
        fontsize=8,
    )

    velocity_history_axis = axes[0, 2]
    velocity_histories = (
        (surface_vr, "#0072b2", "-", r"surface $v_r$"),
        (outer_vr, "#009e73", "-", r"outer $v_r$"),
        (surface_vphi, "#d97706", "--", r"surface $v_\phi$"),
        (outer_vphi, "#7c3aed", "--", r"outer $v_\phi$"),
    )
    plot_histories(velocity_history_axis, velocity_histories, width=1.45)
    velocity_history_axis.axhline(0.0, color="black", lw=0.8)
    velocity_history_axis.set_ylim(
        *padded_linear_limits(
            [surface_vr, outer_vr, surface_vphi, outer_vphi], include_zero=True
        )
    )
    velocity_history_axis.set_ylabel(r"$v(r,t)\;[\mathrm{cm\,s^{-1}}]$")
    velocity_history_axis.legend(loc="best", fontsize=8, ncol=2)

    mdot_history_axis = axes[1, 2]
    diagnostic_colors = ("#2878b5", "#d97706", "#2c9a3a", "#7c3aed")
    plot_histories(mdot_history_axis, (
        (history, color, "-", label)
        for (label, _, history), color in zip(mdot_diagnostics, diagnostic_colors)
    ))
    history_low, history_high, history_linthresh = signed_log_limits(
        [item[2] for item in mdot_diagnostics]
    )
    mdot_history_axis.axhline(0.0, color="black", lw=0.8)
    mdot_history_axis.set_yscale("symlog", linthresh=history_linthresh)
    mdot_history_axis.set_ylim(history_low, history_high)
    mdot_history_axis.set_ylabel(r"$\dot{M}(r,t)\;[M_\odot\,\mathrm{s^{-1}}]$")
    mdot_history_axis.legend(loc="best", fontsize=8, ncol=2)

    magnetic_history_axis = axes[2, 2]
    maximum_br = max(
        float(np.nanmax(np.abs(profile["field_r_gauss"])))
        for profile in profiles
    )
    maximum_bphi = max(
        float(np.nanmax(np.abs(profile["field_phi_gauss"])))
        for profile in profiles
    )
    has_toroidal_field = maximum_bphi > maximum_br * 1.0e-12
    if has_toroidal_field:
        magnetic_histories = (surface_bratio, outer_bratio)
        magnetic_ylabel = r"$B_\phi/B_r$"
    else:
        magnetic_histories = (surface_sigma, outer_sigma)
        magnetic_ylabel = r"$\sigma$"

    plot_histories(magnetic_history_axis, (
        (magnetic_histories[0], "#2878b5", "-", "surface"),
        (magnetic_histories[1], "#2c9a3a", "-", "outer"),
    ), width=1.5)
    if has_toroidal_field:
        magnetic_low, magnetic_high, magnetic_linthresh = signed_log_limits(
            list(magnetic_histories)
        )
        magnetic_history_axis.axhline(0.0, color="black", lw=0.8)
        magnetic_history_axis.set_yscale("symlog", linthresh=magnetic_linthresh)
        magnetic_history_axis.set_ylim(magnetic_low, magnetic_high)
    else:
        magnetic_history_axis.set_yscale("log")
        magnetic_history_axis.set_ylim(*padded_log_limits(list(magnetic_histories)))
    magnetic_history_axis.set_ylabel(magnetic_ylabel)
    magnetic_history_axis.legend(loc="best", fontsize=8)

    power_history_axis = axes[3, 2]
    plot_histories(power_history_axis, (
        (history, color, "-", label)
        for (label, _, history), color in zip(power_diagnostics, diagnostic_colors)
    ))
    power_history_low, power_history_high, power_history_linthresh = signed_log_limits(
        [item[2] for item in power_diagnostics]
    )
    power_history_axis.axhline(0.0, color="black", lw=0.8)
    power_history_axis.set_yscale("symlog", linthresh=power_history_linthresh)
    power_history_axis.set_ylim(power_history_low, power_history_high)
    power_history_axis.set_ylabel(r"$\dot{E}\;[\mathrm{erg\,s^{-1}}]$")
    power_history_axis.legend(loc="best", fontsize=8, ncol=2)

    angular_momentum_history_axis = axes[4, 2]
    plot_histories(angular_momentum_history_axis, (
        (history, color, "-", label)
        for (label, _, history), color in zip(angular_momentum_diagnostics, diagnostic_colors)
    ))
    jdot_history_low, jdot_history_high, jdot_history_linthresh = (
        signed_log_limits([item[2] for item in angular_momentum_diagnostics])
    )
    angular_momentum_history_axis.axhline(0.0, color="black", lw=0.8)
    angular_momentum_history_axis.set_yscale(
        "symlog", linthresh=jdot_history_linthresh
    )
    angular_momentum_history_axis.set_ylim(jdot_history_low, jdot_history_high)
    angular_momentum_history_axis.set_ylabel(
        r"$\dot{J}\;[\mathrm{erg}]$"
    )
    angular_momentum_history_axis.legend(loc="best", fontsize=8, ncol=2)

    # Give gamma its own row instead of crowding the speed/component panels.
    # Show the current radial profile only, consistent with the cleaner layout.
    gamma_low, gamma_high = padded_linear_limits(gamma_all)
    gamma_limits = (max(1.0, gamma_low), max(1.01, gamma_high))
    radial_lines["lorentz_gamma"], = gamma_axis.plot(
        radius_km, initial["lorentz_gamma"], **current_style
    )
    gamma_axis.set_ylim(*gamma_limits)
    gamma_axis.set_ylabel(r"$\gamma$")
    gamma_axis.set_xscale("log")
    gamma_axis.set_xlim(radius_km.min(), radius_km.max())
    gamma_axis.axvline(SURFACE_RADIUS_KM, color="0.40", lw=1.0, ls=":")
    gamma_axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    gamma_axis.tick_params(labelbottom=False)

    plot_histories(gamma_history_axis, (
        (history, color, "-", label)
        for (label, _, history), color in zip(gamma_diagnostics, diagnostic_colors)
    ))
    gamma_history_axis.set_ylim(*gamma_limits)
    gamma_history_axis.set_xlim(*_time_limits(times))
    gamma_history_axis.set_ylabel(r"$\gamma$")
    gamma_history_axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    gamma_history_axis.legend(loc="best", fontsize=8, ncol=2)
    gamma_history_axis.tick_params(labelbottom=False)

    radial_lines["plasma_beta"], = plasma_beta_axis.plot(
        radius_km, initial["plasma_beta"], **current_style
    )
    finite_beta = np.concatenate(plasma_beta_all)
    finite_beta = finite_beta[np.isfinite(finite_beta)]
    if finite_beta.size and np.all(finite_beta > 0.0):
        plasma_beta_axis.set_yscale("log")
        plasma_beta_axis.set_ylim(*padded_log_limits(plasma_beta_all))
    elif finite_beta.size:
        beta_low, beta_high, beta_linthresh = signed_log_limits(plasma_beta_all)
        plasma_beta_axis.set_yscale("symlog", linthresh=beta_linthresh)
        plasma_beta_axis.set_ylim(min(beta_low, 0.0), max(beta_high, 1.1))
    else:
        plasma_beta_axis.set_yscale("log")
        plasma_beta_axis.set_ylim(0.1, 10.0)
    plasma_beta_axis.axhline(1.0, color="0.45", lw=1.0, ls=":")
    plasma_beta_axis.set_ylabel(r"$\beta$")
    plasma_beta_axis.set_xscale("log")
    plasma_beta_axis.set_xlim(radius_km.min(), radius_km.max())
    plasma_beta_axis.axvline(SURFACE_RADIUS_KM, color="0.40", lw=1.0, ls=":")
    plasma_beta_axis.tick_params(labelbottom=False)
    beta_status = plasma_beta_axis.text(
        0.5, 0.5, "", transform=plasma_beta_axis.transAxes,
        ha="center", va="center", fontsize=9, visible=False,
    )

    if any(np.any(np.isfinite(values)) for values in specific_angular_momentum_all):
        ell_limits = signed_log_limits(specific_angular_momentum_all)
    else:
        ell_limits = (-1.0, 1.0, 1.0e-3)
    plot_components(specific_angular_momentum_axis, (
        (specific_angular_momentum_keys[0], "--", 1.45),
        (specific_angular_momentum_keys[1], ":", 1.55),
        (specific_angular_momentum_keys[2], "-", 2.05),
    ))
    specific_angular_momentum_axis.legend(
        handles=[radial_lines[key] for key in specific_angular_momentum_keys],
        labels=["fluid", "EM", "total"], loc="best", fontsize=8, ncol=3,
    )
    specific_angular_momentum_axis.set_xscale("log")
    specific_angular_momentum_axis.set_xlim(radius_km.min(), radius_km.max())
    specific_angular_momentum_axis.axvline(
        SURFACE_RADIUS_KM, color="0.40", lw=1.0, ls=":"
    )
    specific_angular_momentum_axis.set_xlabel(r"$r\;[\mathrm{km}]$")
    ell_status = specific_angular_momentum_axis.text(
        0.5, 0.5, r"$\ell$ undefined: zero/nonfinite mass flux",
        transform=specific_angular_momentum_axis.transAxes,
        ha="center", va="center", fontsize=9, visible=False,
    )
    plot_histories(specific_angular_momentum_history_axis, (
        (history, color, "-", label)
        for (label, _, history), color in zip(
            specific_angular_momentum_diagnostics, diagnostic_colors
        )
    ))
    for axis in (specific_angular_momentum_axis, specific_angular_momentum_history_axis):
        axis.axhline(0.0, color="black", lw=0.8)
        axis.set_yscale("symlog", linthresh=ell_limits[2])
        axis.set_ylim(*ell_limits[:2])
        axis.set_ylabel(r"$\ell\;[\mathrm{cm^2\,s^{-1}}]$")
    specific_angular_momentum_history_axis.set_xlim(*_time_limits(times))
    specific_angular_momentum_history_axis.set_xlabel(r"$t\;[\mathrm{s}]$")
    specific_angular_momentum_history_axis.legend(loc="best", fontsize=8, ncol=2)

    for row in range(5):
        for col in range(2):
            axis = axes[row, col]
            axis.set_xscale("log")
            axis.set_xlim(radius_km.min(), radius_km.max())
            axis.axvline(
                SURFACE_RADIUS_KM, color="0.65" if axis is speed_axis else "0.40",
                lw=0.8 if axis is speed_axis else 1.0, ls=":",
            )
            axis.tick_params(labelbottom=False)
        axes[row, 2].set_xlim(*_time_limits(times))
        axes[row, 2].tick_params(labelbottom=False)

    for axis in all_axes:
        if axis.get_yscale() in ("log", "symlog"):
            axis.yaxis.get_major_locator().set_params(numticks=6)
        axis.tick_params(
            axis="both", which="major", direction="in",
            length=7.0, width=1.1, labelsize=10, pad=4,
        )
        axis.tick_params(
            axis="both", which="minor", direction="in", length=4.0, width=0.9
        )
        axis.tick_params(axis="x", which="both", top=True, bottom=True)
        axis.xaxis.label.set_size(12)
        axis.yaxis.label.set_size(11)
        axis.yaxis.labelpad = 8
    speed_axis.tick_params(which="major", length=4.5, width=0.9)
    speed_axis.tick_params(which="minor", length=2.5, width=0.6, color="0.55")
    speed_axis.tick_params(axis="x", which="minor", top=False)
    for axis in history_axes:
        axis.tick_params(axis="y", left=False, labelleft=False, right=True, labelright=True)
        axis.yaxis.set_label_position("right")

    axes[0, 0].legend(
        handles=[
            Line2D([], [], color=initial_color, lw=1.3, label="initial"),
            Line2D([], [], color=current_color, lw=1.6, label="current"),
        ],
        loc="best",
        fontsize=8,
    )

    history_cursors = [
        axis.axvline(times[0], color="0.35", lw=1.0, ls="--")
        for axis in history_axes
    ]
    title = fig.suptitle("", y=0.982, fontsize=14)
    rotation_text = ""
    if "P_ROT_MS" in parameters:
        rotation_text = rf",\quad P_{{\rm rot}}={parameters['P_ROT_MS']:g}\,\mathrm{{ms}}"
    parameter_title = (
        rf"$M_{{\rm NS}}={parameters['M_NS']:g}\,M_\odot"
        rf",\quad c_{{s,0}}={scientific_latex(parameters['CS_REL_0'] * C_CGS)}\,\mathrm{{cm\,s^{{-1}}}}"
        rf",\quad \rho_0={scientific_latex(parameters['RHO_IN'])}"
        rf"\,\mathrm{{g\,cm^{{-3}}}}"
        rf",\quad B_0={scientific_latex(parameters['B_SURF'])}\,\mathrm{{G}}"
        rf"{rotation_text},\quad N_r={radius_km.size}$"
    )
    run_label = "rotating RMHD" if "P_ROT_MS" in parameters else "RMHD"

    def update(frame_index: int):
        profile = profiles[frame_index]
        for key, line in radial_lines.items():
            line.set_ydata(profile[key])
        no_finite_beta = not np.any(np.isfinite(profile["plasma_beta"][active]))
        beta_status.set_visible(no_finite_beta)
        if no_finite_beta:
            beta_status.set_text(
                r"$\beta=\infty$: zero magnetic pressure"
                if np.all(np.isposinf(profile["plasma_beta"][active]))
                else r"No finite $\beta$ in active domain"
            )
        ell_status.set_visible(not np.any(np.isfinite(
            profile["specific_angular_momentum_total_cm2_s"][active]
        )))

        sonic_radius = first_outward_zero_crossing(
            radius_km,
            profile["velocity_cm_s"] - profile["sound_speed_cm_s"],
            active,
        )
        if sonic_radius is None:
            sonic_line.set_visible(False)
            sonic_marker.set_visible(False)
            sonic_text.set_text(r"Sonic: no outward $v_r=c_s$ crossing")
        else:
            sonic_speed = float(
                np.interp(sonic_radius, radius_km, profile["velocity_cm_s"])
            )
            sonic_line.set_data([sonic_radius, sonic_radius], [0.0, sonic_speed])
            sonic_marker.set_data([sonic_radius], [sonic_speed])
            sonic_line.set_visible(True)
            sonic_marker.set_visible(True)
            sonic_text.set_text(
                rf"Sonic: $r_{{\rm s}}={sonic_radius:.0f}\,\mathrm{{km}}$"
            )

        alfven_radius = first_outward_zero_crossing(
            radius_km,
            profile["velocity_cm_s"] - profile["alfven_speed_cm_s"],
            active,
        )
        if alfven_radius is None:
            alfven_radius_line.set_visible(False)
            alfven_radius_marker.set_visible(False)
            alfven_radius_text.set_text(r"$R_A$: no crossing")
            alfven_radius_text.set_visible(True)
        else:
            alfven_speed = float(
                np.interp(alfven_radius, radius_km, profile["velocity_cm_s"])
            )
            alfven_radius_line.set_data([alfven_radius, alfven_radius], [0.0, alfven_speed])
            alfven_radius_marker.set_data([alfven_radius], [alfven_speed])
            alfven_radius_line.set_visible(True)
            alfven_radius_marker.set_visible(True)
            alfven_radius_text.set_text(
                rf"$R_A={alfven_radius:.0f}\,\mathrm{{km}}$"
            )
            alfven_radius_text.set_visible(True)

        end = frame_index + 1
        for line, history in history_lines:
            line.set_data(times[:end], history[:end])
        for cursor in history_cursors:
            cursor.set_xdata([times[frame_index], times[frame_index]])

        title.set_text(
            f"{run_label}: {parameter_title}, "
            rf"$t={times[frame_index]:.4f}\,\mathrm{{s}}$"
        )

    update(0)
    if args.pdf:
        # Keep labels clear of ticks with consistent PDF renderer margins.
        for (_, col), axis in np.ndenumerate(axes):
            axis.yaxis.set_label_coords(1.22 if col == 2 else -0.22, 0.5)
        gamma_axis.yaxis.set_label_coords(-0.22, 0.5)
        plasma_beta_axis.yaxis.set_label_coords(-0.22, 0.5)
        gamma_history_axis.yaxis.set_label_coords(1.22, 0.5)
        specific_angular_momentum_axis.yaxis.set_label_coords(-0.10, 0.5)
        specific_angular_momentum_history_axis.yaxis.set_label_coords(1.22, 0.5)

    # Use the locator's sparse decades, then drop labels that crowd zero.
    # Limits are fixed across frames, so the same ticks work for every export.
    fig.canvas.draw()
    min_spacing = 20.0 * fig.dpi / 72.0
    for axis in all_axes:
        if axis.get_yscale() not in ("log", "symlog"):
            continue
        low, high = axis.get_ylim()
        ticks = [tick for tick in axis.get_yticks() if low <= tick <= high]
        kept, positions = [], []
        for tick in sorted(ticks, key=abs):
            position = axis.transData.transform((0.0, tick))[1]
            if all(abs(position - other) >= min_spacing for other in positions):
                kept.append(tick)
                positions.append(position)
        axis.set_yticks(sorted(kept))
        axis.set_ylim(low, high)

    if args.preview or args.pdf:
        update(len(snapshot_numbers) - 1)
        frame_path = (
            output.with_suffix(".pdf") if args.pdf
            else output.with_name(output.stem + "_preview.png")
        )
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        if args.pdf:
            fig.canvas.draw()
            fig.savefig(frame_path, bbox_inches="tight", pad_inches=0.20)
        else:
            fig.savefig(frame_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {frame_path}")
        print(
            f"Latest completed snapshot: {snapshot_numbers[-1]} "
            f"({times[-1]:.6f} s)"
        )
        return frame_path

    animation = FuncAnimation(
        fig,
        update,
        frames=len(snapshot_numbers),
        interval=1000.0 / args.fps,
        blit=False,
        repeat=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(
        output,
        writer=FFMpegWriter(
            fps=args.fps,
            codec="gif",
            extra_args=[
                "-threads", "1", "-filter_complex_threads", "1",
                "-filter_complex",
                "split[a][b];[a]palettegen=stats_mode=single[p];"
                "[b][p]paletteuse=new=1",
            ],
        ),
        dpi=args.dpi,
        progress_callback=lambda frame, total: print(
            f"Rendering frame {frame + 1}/{total}", end="\r", flush=True
        ),
    )
    plt.close(fig)
    print(f"\nSaved {output}")
    print(f"Frames: {len(snapshot_numbers)}")
    print(f"Duration: {len(snapshot_numbers) / args.fps:.2f} s per loop")
    return output


def main() -> None:
    animate_rmhd_profiles()


if __name__ == "__main__":
    main()
