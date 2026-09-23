"""Animate saved 1D relativistic-hydrodynamic wind profiles."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.lines import Line2D
import numpy as np

from wind_common import (
    REPO_DIR, derive_profiles, first_outward_zero_crossing,
    read_catalog, read_definitions, read_grid, read_parameters, read_snapshot,
)
from plot_helpers import (
    padded_linear_limits, padded_log_limits, signed_log_limits,
    scientific_latex, sample_histories,
)


DEFAULT_RUN_DIR = REPO_DIR / "problems" / "rhd"

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


def freeze_legend_position(legend, axis) -> None:
    """Replace a dynamic ``loc='best'`` result with a fixed axes position."""
    bounds = legend.get_window_extent()
    lower_left = axis.transAxes.inverted().transform((bounds.x0, bounds.y0))
    legend.set_loc("lower left")
    legend.borderaxespad = 0.0
    legend.set_bbox_to_anchor(
        (float(lower_left[0]), float(lower_left[1])),
        transform=axis.transAxes,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Animate every Nth snapshot while always including the last one.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Save the final-frame layout as a PNG instead of encoding a GIF.",
    )
    args = parser.parse_args()

    if args.fps <= 0.0:
        parser.error("--fps must be positive")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    if args.stride <= 0:
        parser.error("--stride must be positive")

    run_dir = args.run_dir.resolve()
    output = args.output or run_dir / "wind_evolution_rhd.gif"

    physics, units = read_definitions(run_dir / "definitions.h")
    if physics != "RHD":
        raise ValueError(f"Expected an RHD run in {run_dir}; found {physics}")

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
        profile = derive_profiles(snapshot, radius_km, physics, units, parameters)
        profiles.append(profile)
        times_ms.append(float(snapshot["time"]) * time_unit * 1.0e3)

    initial = profiles[0]
    density_all = [profile["density"] for profile in profiles]
    pressure_all = [profile["pressure"] for profile in profiles]
    temperature_all = [profile["temperature"] for profile in profiles]
    velocity_all = [profile["velocity_c"] for profile in profiles]
    sound_all = [profile["sound_speed_c"] for profile in profiles]
    bernoulli_all = [profile["bernoulli_c2"] for profile in profiles]
    mdot_all = [profile["mdot_g_s"] for profile in profiles]
    edot_all = [profile["edot_kin_erg_s"] for profile in profiles]

    time_values = np.asarray(times_ms)
    active = radius_km > 12.0
    surface_index = int(np.flatnonzero(active)[0])
    surface_velocity = np.asarray(
        [profile["velocity_c"][surface_index] for profile in profiles]
    )
    outer_velocity = np.asarray(
        [profile["velocity_c"][-1] for profile in profiles]
    )
    mdot_diagnostics = sample_histories(profiles, radius_km, "mdot_g_s")
    edot_diagnostics = sample_histories(profiles, radius_km, "edot_kin_erg_s")

    fig = plt.figure(figsize=(13.2, 11.2))
    grid = fig.add_gridspec(
        12, 3, hspace=0.0, wspace=0.34,
        width_ratios=(1.0, 1.0, 1.08),
    )
    axes = np.empty((3, 2), dtype=object)
    axes[0, 0] = fig.add_subplot(grid[0:4, 0])
    axes[1, 0] = fig.add_subplot(grid[4:8, 0], sharex=axes[0, 0])
    axes[2, 0] = fig.add_subplot(grid[8:12, 0], sharex=axes[0, 0])
    axes[0, 1] = fig.add_subplot(grid[0:3, 1], sharex=axes[0, 0])
    axes[1, 1] = fig.add_subplot(grid[3:6, 1], sharex=axes[0, 0])
    axes[2, 1] = fig.add_subplot(grid[6:9, 1], sharex=axes[0, 0])
    edot_radial_axis = fig.add_subplot(grid[9:12, 1], sharex=axes[0, 0])
    radial_axes = [*axes.ravel(), edot_radial_axis]
    velocity_time_axis = fig.add_subplot(grid[0:4, 2])
    mdot_time_axis = fig.add_subplot(
        grid[4:8, 2], sharex=velocity_time_axis
    )
    edot_time_axis = fig.add_subplot(
        grid[8:12, 2], sharex=velocity_time_axis
    )
    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.075, top=0.945)

    initial_color = "0.60"
    current_color = "#1b6ca8"
    initial_style = {"color": initial_color, "lw": 1.35, "ls": "--"}
    current_style = {"color": current_color, "lw": 1.65}

    axes[0, 0].plot(radius_km, initial["density"], **initial_style)
    density_line, = axes[0, 0].plot(radius_km, initial["density"], **current_style)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylim(*padded_log_limits(density_all))
    axes[0, 0].set_ylabel(r"$\rho\;[\mathrm{g\,cm^{-3}}]$")

    speed_axis = axes[0, 1]
    speed_axis.plot(radius_km, initial["velocity_c"], **initial_style)
    velocity_line, = speed_axis.plot(radius_km, initial["velocity_c"], **current_style)
    speed_axis.plot(
        radius_km, initial["sound_speed_c"],
        color=initial_color, lw=1.35, ls=":",
    )
    sound_line, = speed_axis.plot(
        radius_km, initial["sound_speed_c"],
        color=current_color, lw=1.65, ls=":",
    )
    speed_axis.plot(
        radius_km, initial["escape_velocity_c"],
        color="black", lw=1.2, ls="-.",
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
            velocity_all + sound_all + [initial["escape_velocity_c"]],
            include_zero=True,
        )
    )
    speed_axis.set_ylabel(r"$v\;[c]$")
    speed_legend = speed_axis.legend(
        handles=[
            Line2D([], [], color="black", lw=1.5, label=r"$v_r$"),
            Line2D([], [], color="black", lw=1.5, ls=":", label=r"$c_s$"),
            Line2D([], [], color="black", lw=1.2, ls="-.", label=r"$v_{\rm esc}$"),
        ],
        loc="best",
        fontsize=8,
    )
    sonic_text = speed_axis.text(
        0.97, 0.06, "", transform=speed_axis.transAxes,
        ha="right", va="bottom", fontsize=8, color="#b45309",
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
        visible=False,
    )

    axes[1, 0].plot(radius_km, initial["pressure"], **initial_style)
    pressure_line, = axes[1, 0].plot(radius_km, initial["pressure"], **current_style)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylim(*padded_log_limits(pressure_all))
    axes[1, 0].set_ylabel(r"$P\;[\mathrm{erg\,cm^{-3}}]$")

    axes[1, 1].plot(radius_km, initial["bernoulli_c2"], **initial_style)
    bernoulli_line, = axes[1, 1].plot(
        radius_km, initial["bernoulli_c2"], **current_style
    )
    axes[1, 1].axhline(0.0, color="black", lw=0.8)
    unbound_line = axes[1, 1].axvline(
        radius_km[0], color="#7c3aed", lw=1.1, ls="--", visible=False
    )
    unbound_marker, = axes[1, 1].plot(
        [], [], marker="D", ms=4.5, mec="white", mew=0.6,
        color="#7c3aed", ls="none", visible=False,
    )
    axes[1, 1].set_ylim(
        *padded_linear_limits(bernoulli_all, include_zero=True)
    )
    axes[1, 1].set_ylabel(r"$\mathrm{Be}\;[c^2]$")
    unbound_text = axes[1, 1].text(
        0.97, 0.06, "", transform=axes[1, 1].transAxes,
        ha="right", va="bottom", fontsize=8, color="#6d28d9",
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
        visible=False,
    )

    axes[2, 0].plot(radius_km, initial["temperature"], **initial_style)
    temperature_line, = axes[2, 0].plot(
        radius_km, initial["temperature"], **current_style
    )
    axes[2, 0].set_yscale("log")
    axes[2, 0].set_ylim(*padded_log_limits(temperature_all))
    axes[2, 0].set_ylabel(r"$T\;[\mathrm{K}]$")

    mdot_low, mdot_high, mdot_linthresh = signed_log_limits(mdot_all, min_fraction=1.0e-8)
    axes[2, 1].plot(radius_km, initial["mdot_g_s"], **initial_style)
    mdot_line, = axes[2, 1].plot(radius_km, initial["mdot_g_s"], **current_style)
    axes[2, 1].axhline(0.0, color="black", lw=0.8)
    axes[2, 1].set_yscale("symlog", linthresh=mdot_linthresh)
    axes[2, 1].set_ylim(mdot_low, mdot_high)
    axes[2, 1].set_ylabel(r"$\dot{M}\;[\mathrm{g\,s^{-1}}]$")

    edot_low, edot_high, edot_linthresh = signed_log_limits(edot_all, min_fraction=1.0e-8)
    edot_radial_axis.plot(
        radius_km, initial["edot_kin_erg_s"], **initial_style
    )
    edot_radial_line, = edot_radial_axis.plot(
        radius_km, initial["edot_kin_erg_s"], **current_style
    )
    edot_radial_axis.axhline(0.0, color="black", lw=0.8)
    edot_radial_axis.set_yscale("symlog", linthresh=edot_linthresh)
    edot_radial_axis.yaxis.get_major_locator().set_params(numticks=7)
    edot_radial_axis.set_ylim(edot_low, edot_high)
    edot_radial_axis.set_ylabel(
        r"$\dot{E}_{\rm kin}\;[\mathrm{erg\,s^{-1}}]$"
    )

    surface_time_line, = velocity_time_axis.plot(
        time_values[:1], surface_velocity[:1],
        color="#2878b5", lw=1.65, label=r"surface $v_r$",
    )
    outer_time_line, = velocity_time_axis.plot(
        time_values[:1], outer_velocity[:1],
        color="#2c9a3a", lw=1.65, label=r"outer $v_r$",
    )
    velocity_time_axis.axhline(0.0, color="black", lw=0.8)
    velocity_time_axis.set_ylim(
        *padded_linear_limits(
            [surface_velocity, outer_velocity], include_zero=True
        )
    )
    velocity_time_axis.set_ylabel(r"$v\;[c]$")
    velocity_time_legend = velocity_time_axis.legend(
        loc="best", fontsize=8
    )

    mdot_histories = [item[2] for item in mdot_diagnostics]
    history_low, history_high, history_linthresh = signed_log_limits(
        mdot_histories, min_fraction=1.0e-8
    )
    mdot_time_lines = []
    mdot_colors = ("#2878b5", "#d97706", "#2c9a3a", "#7c3aed")
    for (label, _, history), color in zip(mdot_diagnostics, mdot_colors):
        line, = mdot_time_axis.plot(
            time_values[:1], history[:1], color=color, lw=1.45, label=label
        )
        mdot_time_lines.append(line)
    mdot_time_axis.axhline(0.0, color="black", lw=0.8)
    mdot_time_axis.set_yscale("symlog", linthresh=history_linthresh)
    mdot_time_axis.yaxis.get_major_locator().set_params(numticks=7)
    mdot_time_axis.set_ylim(history_low, history_high)
    mdot_time_axis.set_ylabel(r"$\dot{M}(r,t)\;[\mathrm{g\,s^{-1}}]$")
    mdot_time_legend = mdot_time_axis.legend(
        loc="best", fontsize=8, ncol=2
    )

    edot_time_lines = []
    for (label, _, history), color in zip(edot_diagnostics, mdot_colors):
        line, = edot_time_axis.plot(
            time_values[:1], history[:1], color=color, lw=1.45, label=label
        )
        edot_time_lines.append(line)
    edot_time_axis.axhline(0.0, color="black", lw=0.8)
    edot_time_axis.set_yscale("symlog", linthresh=edot_linthresh)
    edot_time_axis.yaxis.get_major_locator().set_params(numticks=7)
    edot_time_axis.set_ylim(edot_low, edot_high)
    edot_time_axis.set_ylabel(
        r"$\dot{E}_{\rm kin}(r,t)\;[\mathrm{erg\,s^{-1}}]$"
    )
    edot_time_axis.set_xlabel(r"$t\;[\mathrm{ms}]$")

    velocity_time_axis.tick_params(labelbottom=False)
    mdot_time_axis.tick_params(labelbottom=False)
    velocity_time_axis.set_xlim(time_values[0], time_values[-1])
    velocity_time_cursor = velocity_time_axis.axvline(
        time_values[0], color="0.35", lw=1.0, ls="--"
    )
    mdot_time_cursor = mdot_time_axis.axvline(
        time_values[0], color="0.35", lw=1.0, ls="--"
    )
    edot_time_cursor = edot_time_axis.axvline(
        time_values[0], color="0.35", lw=1.0, ls="--"
    )

    all_axes = [
        *radial_axes, velocity_time_axis, mdot_time_axis, edot_time_axis
    ]
    for axis in all_axes:
        axis.tick_params(
            axis="both", which="major", direction="in",
            length=7.0, width=1.1, labelsize=11, pad=4,
        )
        axis.tick_params(
            axis="both", which="minor", direction="in",
            length=4.0, width=0.9,
        )
        axis.xaxis.label.set_size(12)
        axis.yaxis.label.set_size(12)
        axis.yaxis.labelpad = 8
        axis.tick_params(
            axis="x", which="both", top=True, bottom=True,
            labeltop=False,
        )

    for axis in (velocity_time_axis, mdot_time_axis, edot_time_axis):
        axis.tick_params(
            axis="y", left=False, labelleft=False,
            right=True, labelright=True,
        )
        axis.yaxis.set_label_position("right")

    for row in range(3):
        axes[row, 0].tick_params(
            axis="y", left=True, labelleft=True,
            right=False, labelright=False,
        )
        axes[row, 1].tick_params(
            axis="y", left=True, labelleft=True,
            right=False, labelright=False, pad=4,
        )
        axes[row, 1].yaxis.set_label_position("left")

    edot_radial_axis.tick_params(
        axis="y", left=True, labelleft=True,
        right=False, labelright=False, pad=4,
    )
    edot_radial_axis.yaxis.set_label_position("left")

    for axis in radial_axes:
        axis.set_xscale("log")
        axis.set_xlim(radius_km.min(), radius_km.max())
        axis.axvline(12.0, color="0.40", lw=1.0, ls=":")

    for row in range(2):
        axes[row, 0].tick_params(labelbottom=False)
    for row in range(3):
        axes[row, 1].tick_params(labelbottom=False)

    axes[2, 0].set_xlabel(r"$r\;[\mathrm{km}]$")
    edot_radial_axis.set_xlabel(r"$r\;[\mathrm{km}]$")
    density_legend = axes[0, 0].legend(
        handles=[
            Line2D([], [], color=initial_color, lw=1.35, ls="--", label="initial"),
            Line2D([], [], color=current_color, lw=1.65, label="current"),
        ],
        loc="best",
        fontsize=8,
    )

    title = fig.suptitle("", y=0.982, fontsize=14)
    parameter_title = (
        rf"$M_{{\rm NS}}={parameters['M_NS']:g}\,M_\odot"
        rf",\quad \rho_0={scientific_latex(parameters['RHO_IN'])}"
        rf"\,\mathrm{{g\,cm^{{-3}}}}"
        rf",\quad c_{{s,0}}={parameters['CS_REL_0']:g}\,c"
        rf",\quad N_r={radius_km.size}$"
    )
    def update(frame_index: int):
        profile = profiles[frame_index]
        density_line.set_ydata(profile["density"])
        velocity_line.set_ydata(profile["velocity_c"])
        sound_line.set_ydata(profile["sound_speed_c"])
        pressure_line.set_ydata(profile["pressure"])
        bernoulli_line.set_ydata(profile["bernoulli_c2"])
        temperature_line.set_ydata(profile["temperature"])
        mdot_line.set_ydata(profile["mdot_g_s"])
        edot_radial_line.set_ydata(profile["edot_kin_erg_s"])

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
            sonic_text.set_text(rf"$r_{{\rm s}}={sonic_radius:.0f}\,\mathrm{{km}}$")
            sonic_text.set_visible(True)

        unbound_radius = first_outward_zero_crossing(
            radius_km, profile["bernoulli_c2"], active
        )
        if unbound_radius is None:
            unbound_line.set_visible(False)
            unbound_marker.set_visible(False)
            unbound_text.set_visible(False)
        else:
            unbound_line.set_xdata([unbound_radius, unbound_radius])
            unbound_marker.set_data([unbound_radius], [0.0])
            unbound_line.set_visible(True)
            unbound_marker.set_visible(True)
            unbound_text.set_text(
                rf"$r(\mathrm{{Be}}=0)={unbound_radius:.0f}\,\mathrm{{km}}$"
            )
            unbound_text.set_visible(True)

        end = frame_index + 1
        surface_time_line.set_data(
            time_values[:end], surface_velocity[:end]
        )
        outer_time_line.set_data(time_values[:end], outer_velocity[:end])
        for line, (_, _, history) in zip(mdot_time_lines, mdot_diagnostics):
            line.set_data(time_values[:end], history[:end])
        for line, (_, _, history) in zip(edot_time_lines, edot_diagnostics):
            line.set_data(time_values[:end], history[:end])
        current_time = time_values[frame_index]
        velocity_time_cursor.set_xdata([current_time, current_time])
        mdot_time_cursor.set_xdata([current_time, current_time])
        edot_time_cursor.set_xdata([current_time, current_time])
        title.set_text(
            f"RHD: {parameter_title}, "
            rf"$t={times_ms[frame_index]:.1f}\,\mathrm{{ms}}$"
        )

    # Choose each ``best`` location from the fully evolved final state, then
    # convert it to a fixed axes-relative position before animation begins.
    update(len(snapshot_numbers) - 1)
    fig.canvas.draw()
    for legend, axis in (
        (density_legend, axes[0, 0]),
        (speed_legend, speed_axis),
        (velocity_time_legend, velocity_time_axis),
        (mdot_time_legend, mdot_time_axis),
    ):
        freeze_legend_position(legend, axis)
    update(0)

    if args.preview:
        update(len(snapshot_numbers) - 1)
        preview_path = output.with_name(output.stem + "_preview.png")
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {preview_path}")
        return

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


if __name__ == "__main__":
    main()
