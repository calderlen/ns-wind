"""Animate saved 1D relativistic-magnetohydrodynamic wind profiles."""

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
    C_CGS, REPO_DIR, derive_profiles, first_outward_zero_crossing,
    read_definitions, read_catalog, read_grid, read_parameters, read_snapshot,
)
from plot_helpers import (
    padded_linear_limits, padded_log_limits, signed_log_limits,
    scientific_latex, sample_histories,
)


DEFAULT_RUN_DIR = REPO_DIR / "problems" / "rmhd_mpi"
DEFAULT_OUTPUT_NAME = "wind_evolution_rmhd.gif"
SURFACE_RADIUS_KM = 12.0

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


def derive_rmhd_profiles(
    snapshot: dict[str, np.ndarray | float],
    radius_km: np.ndarray,
    units: dict[str, float],
    parameters: dict[str, float],
) -> dict[str, np.ndarray]:
    """Convert RMHD primitives to cgs profiles and presentation diagnostics."""
    required = {"rho", "vx1", "vx2", "vx3", "Bx1", "Bx2", "Bx3", "prs"}
    missing = required - snapshot.keys()
    if missing:
        raise ValueError(f"RMHD snapshot is missing variables: {sorted(missing)}")

    density_unit = units["UNIT_DENSITY"]
    length_unit = units["UNIT_LENGTH"]
    velocity_unit = units["UNIT_VELOCITY"]
    fluid = derive_profiles(snapshot, radius_km, "RMHD", units, parameters)
    density_code = np.asarray(snapshot["rho"])
    beta_r = fluid["velocity_c"]
    beta_theta = np.asarray(snapshot["vx2"]) * velocity_unit / C_CGS
    beta_phi = np.asarray(snapshot["vx3"]) * velocity_unit / C_CGS
    gamma = fluid["lorentz_gamma"]
    enthalpy = fluid["enthalpy_c2"]
    radius_cm = radius_km * length_unit

    field_unit = velocity_unit * np.sqrt(4.0 * np.pi * density_unit)
    field_r_code = np.asarray(snapshot["Bx1"])
    field_theta_code = np.asarray(snapshot["Bx2"])
    field_phi_code = np.asarray(snapshot["Bx3"])
    field_r = field_r_code * field_unit
    field_theta = field_theta_code * field_unit
    field_phi = field_phi_code * field_unit

    beta_dot_field_code = (
        beta_r * field_r_code
        + beta_theta * field_theta_code
        + beta_phi * field_phi_code
    )
    comoving_field_squared_code = (
        (field_r_code**2 + field_theta_code**2 + field_phi_code**2) / gamma**2
        + beta_dot_field_code**2
    )
    magnetization = comoving_field_squared_code / (density_code * enthalpy)
    alfven_speed_c = np.sqrt(magnetization / (1.0 + magnetization))

    # Axial angular-momentum flux through a spherical surface.  The stress
    # T^r_phi is written in the same covariant-b formulation used by PLUTO's
    # RMHD flux function.  Multiplying by 4*pi*r^3 gives the 1D spherical,
    # equatorial-equivalent torque (angular momentum per unit time).
    b0_code = gamma * beta_dot_field_code
    b_r_code = field_r_code / gamma + b0_code * beta_r
    b_phi_code = field_phi_code / gamma + b0_code * beta_phi
    fluid_rphi_code = density_code * enthalpy * gamma**2 * beta_r * beta_phi
    electromagnetic_rphi_code = (
        comoving_field_squared_code * gamma**2 * beta_r * beta_phi
        - b_r_code * b_phi_code
    )
    torque_scale = (
        4.0
        * np.pi
        * radius_cm**3
        * density_unit
        * velocity_unit**2
    )
    angular_momentum_fluid = torque_scale * fluid_rphi_code
    angular_momentum_electromagnetic = torque_scale * electromagnetic_rphi_code
    angular_momentum_total = (
        angular_momentum_fluid + angular_momentum_electromagnetic
    )

    # Ideal-MHD radial Poynting luminosity. Expanding the transverse terms
    # avoids catastrophic cancellation when velocity and field are radial.
    velocity_r = beta_r * C_CGS
    velocity_theta = beta_theta * C_CGS
    velocity_phi = beta_phi * C_CGS
    poynting_power = radius_cm**2 * (
        velocity_r * (field_theta**2 + field_phi**2)
        - field_r * (velocity_theta * field_theta + velocity_phi * field_phi)
    )
    total_power = fluid["edot_kin_erg_s"] + poynting_power

    field_ratio = np.divide(
        field_phi,
        field_r,
        out=np.full_like(field_phi, np.nan),
        where=field_r != 0.0,
    )
    return {
        "radius_km": fluid["radius_km"],
        "density": fluid["density"],
        "pressure": fluid["pressure"],
        "temperature": fluid["temperature"],
        "velocity_c": beta_r,
        "velocity_phi_c": beta_phi,
        "sound_speed_c": fluid["sound_speed_c"],
        "alfven_speed_c": alfven_speed_c,
        "escape_velocity_c": fluid["escape_velocity_c"],
        "fluid_bernoulli_c2": fluid["bernoulli_c2"],
        "mdot_g_s": fluid["mdot_g_s"],
        "kinetic_power_erg_s": fluid["edot_kin_erg_s"],
        "poynting_power_erg_s": poynting_power,
        "total_power_erg_s": total_power,
        "angular_momentum_fluid_erg": angular_momentum_fluid,
        "angular_momentum_electromagnetic_erg": angular_momentum_electromagnetic,
        "angular_momentum_total_erg": angular_momentum_total,
        "field_r_gauss": field_r,
        "field_phi_gauss": field_phi,
        "field_phi_over_r": field_ratio,
        "magnetization": magnetization,
    }


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
    """Build an RMHD preview or GIF and return the written path."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=default_run_dir)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output GIF (default: <run-dir>/plots/<animation-name>.gif)",
    )
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Animate every Nth snapshot while always including the latest one.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Save the latest-frame layout as a PNG instead of encoding a GIF.",
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
    times_ms: list[float] = []
    time_unit = units["UNIT_LENGTH"] / units["UNIT_VELOCITY"]
    for number in snapshot_numbers:
        snapshot = read_snapshot(run_dir, number, radius_km, catalog)
        profiles.append(derive_rmhd_profiles(snapshot, radius_km, units, parameters))
        times_ms.append(float(snapshot["time"]) * time_unit * 1.0e3)

    initial = profiles[0]
    times = np.asarray(times_ms)
    active = radius_km > SURFACE_RADIUS_KM
    if not np.any(active):
        raise ValueError("The radial grid has no active cells outside 12 km")
    surface_index = int(np.flatnonzero(active)[0])

    density_all = [profile["density"] for profile in profiles]
    pressure_all = [profile["pressure"] for profile in profiles]
    temperature_all = [profile["temperature"] for profile in profiles]
    velocity_all = [profile["velocity_c"] for profile in profiles]
    velocity_phi_all = [profile["velocity_phi_c"] for profile in profiles]
    sound_all = [profile["sound_speed_c"] for profile in profiles]
    alfven_all = [profile["alfven_speed_c"] for profile in profiles]
    field_all = [
        profile[key]
        for profile in profiles
        for key in ("field_r_gauss", "field_phi_gauss")
    ]
    bernoulli_all = [profile["fluid_bernoulli_c2"] for profile in profiles]
    mdot_all = [profile["mdot_g_s"] for profile in profiles]
    power_all = [
        profile[key]
        for profile in profiles
        for key in (
            "kinetic_power_erg_s",
            "poynting_power_erg_s",
            "total_power_erg_s",
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
        [profile["velocity_c"][surface_index] for profile in profiles]
    )
    outer_vr = np.asarray([profile["velocity_c"][-1] for profile in profiles])
    surface_vphi = np.asarray(
        [profile["velocity_phi_c"][surface_index] for profile in profiles]
    )
    outer_vphi = np.asarray(
        [profile["velocity_phi_c"][-1] for profile in profiles]
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
    mdot_diagnostics = sample_histories(profiles, radius_km, "mdot_g_s")
    power_diagnostics = sample_histories(profiles, radius_km, "total_power_erg_s")
    angular_momentum_diagnostics = sample_histories(
        profiles, radius_km, "angular_momentum_total_erg"
    )

    fig = plt.figure(figsize=(13.4, 14.6))
    grid = fig.add_gridspec(
        5, 3, hspace=0.0, wspace=0.36, width_ratios=(1.0, 1.0, 1.08)
    )
    axes = np.empty((5, 3), dtype=object)
    for row in range(5):
        for col in range(3):
            sharex = axes[0, col] if row else None
            axes[row, col] = fig.add_subplot(grid[row, col], sharex=sharex)
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

    def plot_components(axis, components, *, initial_width=1.25):
        for key, linestyle, _ in components:
            axis.plot(
                radius_km, initial[key], color=initial_color,
                lw=initial_width, ls=linestyle,
            )
        for key, linestyle, width in components:
            radial_lines[key], = axis.plot(
                radius_km, initial[key], color=current_color,
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
    axes[1, 0].set_ylabel(r"$P_{\rm gas}\;[\mathrm{erg\,cm^{-3}}]$")

    plot_pair(axes[2, 0], "temperature")
    axes[2, 0].set_yscale("log")
    axes[2, 0].set_ylim(*padded_log_limits(temperature_all))
    axes[2, 0].set_ylabel(r"$T\;[\mathrm{K}]$")

    field_low, field_high, field_linthresh = signed_log_limits(field_all)
    plot_components(axes[3, 0], (
        ("field_r_gauss", "-", 1.6),
        ("field_phi_gauss", "--", 1.6),
    ), initial_width=1.3)
    axes[3, 0].axhline(0.0, color="black", lw=0.8)
    axes[3, 0].set_yscale("symlog", linthresh=field_linthresh)
    axes[3, 0].set_ylim(field_low, field_high)
    axes[3, 0].set_ylabel(r"$B\;[\mathrm{G}]$")
    axes[3, 0].legend(
        handles=[
            Line2D([], [], color="black", lw=1.5, label=r"$B_r$"),
            Line2D([], [], color="black", lw=1.5, ls="--", label=r"$B_\phi$"),
        ],
        loc="best",
        fontsize=8,
    )

    plot_pair(axes[4, 0], "magnetization")
    axes[4, 0].set_yscale("log")
    axes[4, 0].set_ylim(*padded_log_limits(magnetization_all))
    axes[4, 0].set_ylabel(r"$\sigma=b^2/(\rho h)$")

    speed_axis = axes[0, 1]
    plot_components(speed_axis, (
        ("velocity_c", "-", 1.6),
        ("velocity_phi_c", "--", 1.6),
        ("sound_speed_c", ":", 1.55),
        ("alfven_speed_c", (0, (3, 1, 1, 1)), 1.45),
    ))
    speed_axis.plot(
        radius_km, initial["escape_velocity_c"],
        color="black", lw=1.15, ls="-.",
    )
    speed_axis.axhline(0.0, color="black", lw=0.8)
    sonic_line = speed_axis.axvline(
        radius_km[0], color="#d97706", lw=1.1, ls="--", visible=False
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
            + [initial["escape_velocity_c"]],
            include_zero=True,
        )
    )
    speed_axis.set_ylabel(r"$v\;[c]$")
    speed_axis.legend(
        handles=[
            Line2D([], [], color="black", lw=1.5, label=r"$v_r$"),
            Line2D([], [], color="black", lw=1.5, ls="--", label=r"$v_\phi$"),
            Line2D([], [], color="black", lw=1.5, ls=":", label=r"$c_s$"),
            Line2D(
                [], [], color="black", lw=1.4, ls=(0, (3, 1, 1, 1)),
                label=r"$v_A$",
            ),
            Line2D([], [], color="black", lw=1.15, ls="-.", label=r"$v_{\rm esc}$"),
        ],
        loc="best",
        fontsize=8,
        ncol=2,
    )
    sonic_text = speed_axis.text(
        0.97, 0.06, "", transform=speed_axis.transAxes,
        ha="right", va="bottom", fontsize=8, color="#b45309",
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
        visible=False,
    )
    alfven_radius_line = speed_axis.axvline(
        radius_km[0], color="#7c3aed", lw=1.1, ls="--", visible=False
    )
    alfven_radius_marker, = speed_axis.plot(
        [], [], marker="s", ms=4.8, mec="white", mew=0.6,
        color="#7c3aed", ls="none", visible=False,
    )
    alfven_radius_text = speed_axis.text(
        0.97, 0.17, "", transform=speed_axis.transAxes,
        ha="right", va="bottom", fontsize=8, color="#6d28d9",
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
        visible=False,
    )

    plot_pair(axes[1, 1], "fluid_bernoulli_c2")
    axes[1, 1].axhline(0.0, color="black", lw=0.8)
    axes[1, 1].set_ylim(*padded_linear_limits(bernoulli_all, include_zero=True))
    axes[1, 1].set_ylabel(r"$\mathrm{Be}_{\rm fluid}\;[c^2]$")
    axes[1, 1].text(
        0.97, 0.06, "excludes magnetic energy", transform=axes[1, 1].transAxes,
        ha="right", va="bottom", fontsize=7.5, color="0.35",
    )

    mdot_low, mdot_high, mdot_linthresh = signed_log_limits(mdot_all)
    plot_pair(axes[2, 1], "mdot_g_s")
    axes[2, 1].axhline(0.0, color="black", lw=0.8)
    axes[2, 1].set_yscale("symlog", linthresh=mdot_linthresh)
    axes[2, 1].set_ylim(mdot_low, mdot_high)
    axes[2, 1].set_ylabel(r"$\dot{M}\;[\mathrm{g\,s^{-1}}]$")

    power_low, power_high, power_linthresh = signed_log_limits(power_all)
    power_axis = axes[3, 1]
    plot_components(power_axis, (
        ("kinetic_power_erg_s", "-", 1.55),
        ("poynting_power_erg_s", "--", 1.55),
        ("total_power_erg_s", ":", 1.55),
    ))
    power_axis.axhline(0.0, color="black", lw=0.8)
    power_axis.set_yscale("symlog", linthresh=power_linthresh)
    power_axis.set_ylim(power_low, power_high)
    power_axis.set_ylabel(r"$\dot{E}\;[\mathrm{erg\,s^{-1}}]$")
    power_axis.legend(
        handles=[
            Line2D([], [], color="black", lw=1.5, label=r"$\dot E_{\rm kin}$"),
            Line2D(
                [], [], color="black", lw=1.5, ls="--",
                label=r"$\dot E_{\rm EM}$ (Poynting)",
            ),
            Line2D([], [], color="black", lw=1.5, ls=":", label="sum"),
        ],
        loc="best",
        fontsize=8,
    )

    angular_momentum_low, angular_momentum_high, angular_momentum_linthresh = (
        signed_log_limits(angular_momentum_all)
    )
    angular_momentum_axis = axes[4, 1]
    plot_components(angular_momentum_axis, (
        ("angular_momentum_fluid_erg", "-", 1.55),
        ("angular_momentum_electromagnetic_erg", "--", 1.55),
        ("angular_momentum_total_erg", ":", 1.55),
    ))
    angular_momentum_axis.axhline(0.0, color="black", lw=0.8)
    angular_momentum_axis.set_yscale(
        "symlog", linthresh=angular_momentum_linthresh
    )
    angular_momentum_axis.set_ylim(
        angular_momentum_low, angular_momentum_high
    )
    angular_momentum_axis.set_ylabel(
        r"$\dot{J}_{\rm eq,4\pi}\;[\mathrm{erg}]$"
    )
    angular_momentum_axis.legend(
        handles=[
            Line2D([], [], color="black", lw=1.5, label="fluid"),
            Line2D([], [], color="black", lw=1.5, ls="--", label="EM"),
            Line2D([], [], color="black", lw=1.5, ls=":", label="sum"),
        ],
        loc="best",
        fontsize=8,
    )

    velocity_history_axis = axes[0, 2]
    velocity_histories = (
        (surface_vr, "#2878b5", "-", r"surface $v_r$"),
        (outer_vr, "#2c9a3a", "-", r"outer $v_r$"),
        (surface_vphi, "#2878b5", "--", r"surface $v_\phi$"),
        (outer_vphi, "#2c9a3a", "--", r"outer $v_\phi$"),
    )
    plot_histories(velocity_history_axis, velocity_histories, width=1.45)
    velocity_history_axis.axhline(0.0, color="black", lw=0.8)
    velocity_history_axis.set_ylim(
        *padded_linear_limits(
            [surface_vr, outer_vr, surface_vphi, outer_vphi], include_zero=True
        )
    )
    velocity_history_axis.set_ylabel(r"$v(r,t)\;[c]$")
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
    mdot_history_axis.set_ylabel(r"$\dot{M}(r,t)\;[\mathrm{g\,s^{-1}}]$")
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
        magnetic_ylabel = r"$\sigma=b^2/(\rho h)$"

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
    power_history_axis.set_ylabel(r"$\dot{E}_{\rm kin+EM}(r,t)\;[\mathrm{erg\,s^{-1}}]$")
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
        r"$\dot{J}_{\rm eq,4\pi}(r,t)\;[\mathrm{erg}]$"
    )
    angular_momentum_history_axis.legend(loc="best", fontsize=8, ncol=2)

    for row in range(5):
        for col in range(2):
            axis = axes[row, col]
            axis.set_xscale("log")
            axis.set_xlim(radius_km.min(), radius_km.max())
            axis.axvline(SURFACE_RADIUS_KM, color="0.40", lw=1.0, ls=":")
            if row < 4:
                axis.tick_params(labelbottom=False)
        axes[row, 2].set_xlim(*_time_limits(times))
        if row < 4:
            axes[row, 2].tick_params(labelbottom=False)

    axes[4, 0].set_xlabel(r"$r\;[\mathrm{km}]$")
    axes[4, 1].set_xlabel(r"$r\;[\mathrm{km}]$")
    axes[4, 2].set_xlabel(r"$t\;[\mathrm{ms}]$")

    for axis in axes.ravel():
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
    for axis in axes[:, 2]:
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
        for axis in axes[:, 2]
    ]
    title = fig.suptitle("", y=0.982, fontsize=14)
    rotation_text = ""
    if "P_ROT_MS" in parameters:
        rotation_text = rf",\quad P_{{\rm rot}}={parameters['P_ROT_MS']:g}\,\mathrm{{ms}}"
    parameter_title = (
        rf"$M_{{\rm NS}}={parameters['M_NS']:g}\,M_\odot"
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

        sonic_radius = first_outward_zero_crossing(
            radius_km,
            profile["velocity_c"] - profile["sound_speed_c"],
            active,
        )
        if sonic_radius is None:
            sonic_line.set_visible(False)
            sonic_marker.set_visible(False)
            sonic_text.set_visible(False)
        else:
            sonic_speed = float(
                np.interp(sonic_radius, radius_km, profile["velocity_c"])
            )
            sonic_line.set_xdata([sonic_radius, sonic_radius])
            sonic_marker.set_data([sonic_radius], [sonic_speed])
            sonic_line.set_visible(True)
            sonic_marker.set_visible(True)
            sonic_text.set_text(
                rf"fluid $r_{{\rm s}}={sonic_radius:.0f}\,\mathrm{{km}}$"
            )
            sonic_text.set_visible(True)

        alfven_radius = first_outward_zero_crossing(
            radius_km,
            profile["velocity_c"] - profile["alfven_speed_c"],
            active,
        )
        if alfven_radius is None:
            alfven_radius_line.set_visible(False)
            alfven_radius_marker.set_visible(False)
            active_difference = (
                profile["velocity_c"][active]
                - profile["alfven_speed_c"][active]
            )
            if np.all(active_difference > 0.0):
                alfven_radius_text.set_text(
                    r"$R_A$: no crossing; $v_r>v_A$ at surface"
                )
            elif np.all(active_difference < 0.0):
                alfven_radius_text.set_text(
                    r"$R_A$: no crossing; sub-Alfvénic in domain"
                )
            else:
                alfven_radius_text.set_text(r"$R_A$: no outward crossing")
            alfven_radius_text.set_visible(True)
        else:
            alfven_speed = float(
                np.interp(alfven_radius, radius_km, profile["velocity_c"])
            )
            alfven_radius_line.set_xdata([alfven_radius, alfven_radius])
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
            rf"$t={times[frame_index]:.1f}\,\mathrm{{ms}}$"
        )

    update(0)
    if args.preview:
        update(len(snapshot_numbers) - 1)
        preview_path = output.with_name(output.stem + "_preview.png")
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {preview_path}")
        print(
            f"Latest completed snapshot: {snapshot_numbers[-1]} "
            f"({times[-1]:.3f} ms)"
        )
        return preview_path

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
