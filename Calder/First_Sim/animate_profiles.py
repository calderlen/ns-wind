"""Animate the saved 1D HD neutron-star wind profiles."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
import numpy as np

from plot_profiles import (
    diagnostics,
    read_definitions,
    read_grid,
    read_output_catalog,
    read_parameters,
    read_snapshot,
)


RUN_DIR = Path(__file__).resolve().parent


def padded_log_limits(values: np.ndarray) -> tuple[float, float]:
    """Return positive logarithmic limits with a modest margin."""
    finite = values[np.isfinite(values) & (values > 0.0)]
    return float(finite.min() / 1.35), float(finite.max() * 1.35)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument(
        "--output",
        type=Path,
        default=RUN_DIR / "wind_evolution_hd.gif",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Save only the final-frame layout preview instead of encoding the GIF",
    )
    args = parser.parse_args()

    units = read_definitions(RUN_DIR / "definitions.h")
    parameters = read_parameters(RUN_DIR / "pluto.ini")
    radius_km = read_grid(RUN_DIR / "grid.out")
    catalog = read_output_catalog(RUN_DIR / "dbl.out")
    snapshot_numbers = sorted(catalog)
    time_unit = units["UNIT_LENGTH"] / units["UNIT_VELOCITY"]

    snapshots = []
    profiles = []
    for number in snapshot_numbers:
        snapshot = read_snapshot(number, radius_km, catalog)
        snapshots.append(snapshot)
        profiles.append(diagnostics(snapshot, radius_km, parameters, units))

    density_all = np.concatenate([profile["density"] for profile in profiles])
    pressure_all = np.concatenate([profile["pressure"] for profile in profiles])
    temperature_all = np.concatenate([profile["temperature"] for profile in profiles])
    mdot_all = np.concatenate([profile["mdot_g_s"] for profile in profiles])

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(8.1, 9.2),
        sharex=True,
        gridspec_kw={"hspace": 0.0, "wspace": 0.055},
    )
    # The lower-right logarithmic tick labels are wide, so reserve enough
    # outer margin for the rotated Mdot axis label in both PNG and GIF output.
    fig.subplots_adjust(left=0.105, right=0.855, bottom=0.075, top=0.955)
    initial = profiles[0]
    blue = "#1f77b4"
    orange = "#ff7f0e"

    # Top left: density. This legend establishes the time/color convention.
    axes[0, 0].loglog(radius_km, initial["density"], color=blue, lw=1.4, label="initial")
    density_line, = axes[0, 0].loglog(
        radius_km, initial["density"], color=orange, lw=1.4, label="current"
    )
    axes[0, 0].set_ylabel(r"$\rho\;[\mathrm{g\,cm^{-3}}]$")
    axes[0, 0].set_ylim(*padded_log_limits(density_all))
    axes[0, 0].legend(loc="upper right", frameon=True)

    # Top right: radial velocity, sound speed, and Newtonian escape speed.
    axes[0, 1].semilogx(radius_km, initial["velocity_c"], color=blue, lw=1.4)
    velocity_line, = axes[0, 1].semilogx(
        radius_km, initial["velocity_c"], color=orange, lw=1.4
    )
    axes[0, 1].semilogx(
        radius_km, initial["sound_speed_c"], color=blue, lw=1.4, ls="--"
    )
    sound_line, = axes[0, 1].semilogx(
        radius_km, initial["sound_speed_c"], color=orange, lw=1.4, ls="--"
    )
    axes[0, 1].semilogx(
        radius_km,
        initial["escape_velocity_c"],
        color="black",
        lw=1.4,
        ls=":",
    )
    axes[0, 1].set_ylabel(r"$v\;[c]$")
    axes[0, 1].set_ylim(-0.08, 1.04)
    axes[0, 1].legend(
        handles=[
            Line2D([], [], color="black", ls=":", lw=1.4, label=r"$v_{\rm esc}$"),
            Line2D([], [], color="black", ls="--", lw=1.4, label=r"$c_s$"),
        ],
        loc="upper right",
        frameon=True,
    )

    # Middle left: pressure. 1 erg cm^-3 is identical to 1 dyn cm^-2.
    axes[1, 0].loglog(radius_km, initial["pressure"], color=blue, lw=1.4)
    pressure_line, = axes[1, 0].loglog(
        radius_km, initial["pressure"], color=orange, lw=1.4
    )
    axes[1, 0].set_ylabel(r"$P\;[\mathrm{ergs\,cm^{-3}}]$")
    axes[1, 0].set_ylim(*padded_log_limits(pressure_all))

    # Middle right: Newtonian Bernoulli parameter in units of c^2.
    axes[1, 1].semilogx(radius_km, initial["bernoulli_c2"], color=blue, lw=1.4)
    bernoulli_line, = axes[1, 1].semilogx(
        radius_km, initial["bernoulli_c2"], color=orange, lw=1.4
    )
    axes[1, 1].axhline(0.0, color="black", lw=1.4, ls=":", label=r"$\mathrm{Be}=0$")
    axes[1, 1].set_ylabel(r"$\mathrm{Be}\;[c^2]$")
    axes[1, 1].set_ylim(-0.05, 0.085)
    axes[1, 1].legend(loc="upper right", frameon=True)

    # Bottom left: radiation temperature, P = a T^4 / 3.
    axes[2, 0].loglog(radius_km, initial["temperature"], color=blue, lw=1.4)
    temperature_line, = axes[2, 0].loglog(
        radius_km, initial["temperature"], color=orange, lw=1.4
    )
    axes[2, 0].set_ylabel(r"$T\;[\mathrm{K}]$")
    axes[2, 0].set_xlabel(r"$r\;[\mathrm{km}]$")
    axes[2, 0].set_ylim(*padded_log_limits(temperature_all))

    # Bottom right: mass flux in cgs units.
    axes[2, 1].loglog(radius_km, initial["mdot_g_s"], color=blue, lw=1.4)
    mdot_line, = axes[2, 1].loglog(
        radius_km, initial["mdot_g_s"], color=orange, lw=1.4
    )
    axes[2, 1].set_ylabel(r"$\dot{M}\;[\mathrm{g\,s^{-1}}]$")
    axes[2, 1].set_xlabel(r"$r\;[\mathrm{km}]$")
    axes[2, 1].set_ylim(*padded_log_limits(mdot_all))

    # Make the panels touch and put all right-column scales on the right.
    for row in range(3):
        axes[row, 0].tick_params(direction="out", which="both", right=False)
        axes[row, 1].tick_params(
            direction="out", which="both", left=False, labelleft=False,
            right=True, labelright=True,
        )
        axes[row, 1].yaxis.set_label_position("right")
    for axis in axes.ravel():
        axis.axvline(12.0, color="0.40", lw=1.15, ls=":", zorder=0.5)
        axis.set_xlim(radius_km.min(), radius_km.max())

    title = fig.suptitle("")
    animated_lines = (
        density_line,
        velocity_line,
        sound_line,
        pressure_line,
        bernoulli_line,
        temperature_line,
        mdot_line,
    )

    def update(frame_index: int):
        snapshot = snapshots[frame_index]
        profile = profiles[frame_index]
        density_line.set_ydata(profile["density"])
        velocity_line.set_ydata(profile["velocity_c"])
        sound_line.set_ydata(profile["sound_speed_c"])
        pressure_line.set_ydata(profile["pressure"])
        bernoulli_line.set_ydata(profile["bernoulli_c2"])
        temperature_line.set_ydata(profile["temperature"])
        mdot_line.set_ydata(profile["mdot_g_s"])
        physical_time_ms = float(snapshot["time"]) * time_unit * 1.0e3
        title.set_text(rf"$t = {physical_time_ms:.1f}\,\mathrm{{ms}}$")
        return (*animated_lines, title)

    if args.preview:
        update(len(snapshot_numbers) - 1)
        preview_path = RUN_DIR / "wind_layout_preview.png"
        fig.savefig(preview_path, dpi=args.dpi, bbox_inches="tight")
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(
        args.output,
        writer=PillowWriter(fps=args.fps),
        dpi=args.dpi,
        progress_callback=lambda frame, total: print(
            f"Rendering frame {frame + 1}/{total}", end="\r", flush=True
        ),
    )
    print(f"\nSaved {args.output}")
    print(f"Frames: {len(snapshot_numbers)}")
    print(f"Duration: {len(snapshot_numbers) / args.fps:.2f} s per loop")


if __name__ == "__main__":
    main()
