"""Animate overlapping HD and RHD neutron-star wind profiles."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
import numpy as np

from compare_hd_rhd import DEFAULT_HD_DIR, DEFAULT_RHD_DIR
from wind_common import (
    REPO_DIR, derive_profiles, read_definitions, read_parameters,
    read_grid, read_catalog, read_snapshot,
)
from plot_helpers import padded_log_limits, padded_linear_limits


def load_animation_data(run_dir: Path) -> dict[str, object]:
    """Load metadata needed to derive every completed snapshot in a run."""
    run_dir = run_dir.resolve()
    physics, units = read_definitions(run_dir / "definitions.h")
    parameters = read_parameters(run_dir / "pluto.ini")
    radius = read_grid(run_dir / "grid.out")
    catalog = read_catalog(run_dir / "dbl.out")
    return {
        "run_dir": run_dir,
        "physics": physics,
        "units": units,
        "parameters": parameters,
        "radius": radius,
        "catalog": catalog,
        "time_unit": units["UNIT_LENGTH"] / units["UNIT_VELOCITY"],
    }


def derive_frame(run: dict[str, object], number: int) -> tuple[dict[str, np.ndarray], float]:
    """Read and derive one snapshot, returning its physical time in ms."""
    snapshot = read_snapshot(
        run["run_dir"], number, run["radius"], run["catalog"]
    )
    profile = derive_profiles(
        snapshot,
        run["radius"],
        run["physics"],
        run["units"],
        run["parameters"],
    )
    time_ms = float(snapshot["time"]) * run["time_unit"] * 1.0e3
    return profile, time_ms


def animate_hd_rhd_comparison(
    hd_dir: Path | str = DEFAULT_HD_DIR,
    rhd_dir: Path | str = DEFAULT_RHD_DIR,
    *,
    output: Path | str = REPO_DIR / "runs" / "hd_rhd_evolution.gif",
    fps: float = 25.0,
    dpi: int = 120,
    preview: bool = False,
) -> Path:
    """Overlay synchronized HD and RHD evolution on the same six axes.

    The common completed snapshot numbers are used. Initial profiles remain
    visible in lighter colors while the darker HD/RHD curves evolve.
    """
    hd = load_animation_data(Path(hd_dir))
    rhd = load_animation_data(Path(rhd_dir))
    if hd["physics"] != "HD" or rhd["physics"] != "RHD":
        raise ValueError("hd_dir must contain HD output and rhd_dir RHD output")

    frame_numbers = sorted(set(hd["catalog"]) & set(rhd["catalog"]))
    if not frame_numbers or frame_numbers[0] != 0:
        raise ValueError("The HD and RHD runs do not share snapshot zero")

    hd_profiles: list[dict[str, np.ndarray]] = []
    rhd_profiles: list[dict[str, np.ndarray]] = []
    hd_times: list[float] = []
    rhd_times: list[float] = []
    for number in frame_numbers:
        hd_profile, hd_time = derive_frame(hd, number)
        rhd_profile, rhd_time = derive_frame(rhd, number)
        hd_profiles.append(hd_profile)
        rhd_profiles.append(rhd_profile)
        hd_times.append(hd_time)
        rhd_times.append(rhd_time)

    hd_initial = hd_profiles[0]
    rhd_initial = rhd_profiles[0]
    colors = {
        "hd_initial": "#9ecae1",
        "hd_current": "#08519c",
        "rhd_initial": "#fdae6b",
        "rhd_current": "#d94801",
    }

    fig, axes = plt.subplots(
        3, 2, figsize=(8.6, 9.2), sharex=True,
        gridspec_kw={"hspace": 0.0, "wspace": 0.055},
    )
    fig.subplots_adjust(left=0.10, right=0.855, bottom=0.075, top=0.955)

    # Values from every frame establish fixed animation limits.
    all_profiles = hd_profiles + rhd_profiles
    panel_specs = [
        ((0, 0), "density", r"$\rho\;[\mathrm{g\,cm^{-3}}]$", "log"),
        ((1, 0), "pressure", r"$P\;[\mathrm{ergs\,cm^{-3}}]$", "log"),
        ((2, 0), "temperature", r"$T\;[\mathrm{K}]$", "log"),
        ((1, 1), "bernoulli_c2", r"$\mathrm{Be}\;[c^2]$", "linear"),
        ((2, 1), "mdot_g_s", r"$\dot{M}\;[\mathrm{g\,s^{-1}}]$", "log"),
    ]

    current_lines: dict[str, tuple[object, object]] = {}
    for (row, col), key, ylabel, scale in panel_specs:
        axis = axes[row, col]
        axis.plot(
            hd_initial["radius_km"], hd_initial[key],
            color=colors["hd_initial"], lw=1.4, label="HD initial",
        )
        hd_line, = axis.plot(
            hd_initial["radius_km"], hd_initial[key],
            color=colors["hd_current"], lw=1.65, label="HD current",
        )
        axis.plot(
            rhd_initial["radius_km"], rhd_initial[key],
            color=colors["rhd_initial"], lw=1.4, label="RHD initial",
        )
        rhd_line, = axis.plot(
            rhd_initial["radius_km"], rhd_initial[key],
            color=colors["rhd_current"], lw=1.65, label="RHD current",
        )
        current_lines[key] = (hd_line, rhd_line)
        axis.set_xscale("log")
        axis.set_yscale(scale)
        axis.set_ylabel(ylabel)
        values = [profile[key] for profile in all_profiles]
        if scale == "log":
            axis.set_ylim(*padded_log_limits(values))
        else:
            axis.set_ylim(*padded_linear_limits(values, include_zero=True))

    axes[0, 0].legend(loc="upper right", fontsize=8, frameon=True)
    axes[1, 1].axhline(0.0, color="black", lw=1.2, ls=":")

    # Speed panel: color identifies state and line style identifies quantity.
    speed_axis = axes[0, 1]
    speed_axis.plot(
        hd_initial["radius_km"], hd_initial["velocity_c"],
        color=colors["hd_initial"], lw=1.4,
    )
    hd_velocity, = speed_axis.plot(
        hd_initial["radius_km"], hd_initial["velocity_c"],
        color=colors["hd_current"], lw=1.65,
    )
    speed_axis.plot(
        rhd_initial["radius_km"], rhd_initial["velocity_c"],
        color=colors["rhd_initial"], lw=1.4,
    )
    rhd_velocity, = speed_axis.plot(
        rhd_initial["radius_km"], rhd_initial["velocity_c"],
        color=colors["rhd_current"], lw=1.65,
    )
    speed_axis.plot(
        hd_initial["radius_km"], hd_initial["sound_speed_c"],
        color=colors["hd_initial"], lw=1.35, ls="--",
    )
    hd_sound, = speed_axis.plot(
        hd_initial["radius_km"], hd_initial["sound_speed_c"],
        color=colors["hd_current"], lw=1.5, ls="--",
    )
    speed_axis.plot(
        rhd_initial["radius_km"], rhd_initial["sound_speed_c"],
        color=colors["rhd_initial"], lw=1.35, ls="--",
    )
    rhd_sound, = speed_axis.plot(
        rhd_initial["radius_km"], rhd_initial["sound_speed_c"],
        color=colors["rhd_current"], lw=1.5, ls="--",
    )
    speed_axis.plot(
        hd_initial["radius_km"], hd_initial["escape_velocity_c"],
        color="black", lw=1.4, ls=":",
    )
    speed_axis.set_xscale("log")
    speed_axis.set_ylabel(r"$v\;[c]$")
    speed_axis.set_ylim(-0.08, 1.04)
    speed_axis.legend(
        handles=[
            Line2D([], [], color="black", lw=1.5, label=r"$v_r$"),
            Line2D([], [], color="black", lw=1.4, ls="--", label=r"$c_s$"),
            Line2D([], [], color="black", lw=1.4, ls=":", label=r"$v_{\rm esc}$"),
        ],
        loc="upper right", fontsize=8, frameon=True,
    )

    for row in range(3):
        axes[row, 0].tick_params(direction="out", which="both", right=False)
        axes[row, 1].tick_params(
            direction="out", which="both", left=False, labelleft=False,
            right=True, labelright=True,
        )
        axes[row, 1].yaxis.set_label_position("right")
    xmin = min(hd_initial["radius_km"].min(), rhd_initial["radius_km"].min())
    xmax = max(hd_initial["radius_km"].max(), rhd_initial["radius_km"].max())
    for axis in axes.ravel():
        axis.axvline(12.0, color="0.40", lw=1.15, ls=":", zorder=0.5)
        axis.set_xlim(xmin, xmax)
    axes[2, 0].set_xlabel(r"$r\;[\mathrm{km}]$")
    axes[2, 1].set_xlabel(r"$r\;[\mathrm{km}]$")
    title = fig.suptitle("")

    def update(frame_index: int):
        hd_profile = hd_profiles[frame_index]
        rhd_profile = rhd_profiles[frame_index]
        for key, (hd_line, rhd_line) in current_lines.items():
            hd_line.set_ydata(hd_profile[key])
            rhd_line.set_ydata(rhd_profile[key])
        hd_velocity.set_ydata(hd_profile["velocity_c"])
        rhd_velocity.set_ydata(rhd_profile["velocity_c"])
        hd_sound.set_ydata(hd_profile["sound_speed_c"])
        rhd_sound.set_ydata(rhd_profile["sound_speed_c"])
        time_ms = 0.5 * (hd_times[frame_index] + rhd_times[frame_index])
        title.set_text(rf"$t = {time_ms:.1f}\,\mathrm{{ms}}$")

    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if preview:
        update(len(frame_numbers) - 1)
        preview_path = output_path.with_name(output_path.stem + "_preview.png")
        fig.savefig(preview_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {preview_path}")
        return preview_path

    animation = FuncAnimation(
        fig, update, frames=len(frame_numbers), interval=1000.0 / fps,
        blit=False, repeat=True,
    )
    animation.save(
        output_path,
        writer=PillowWriter(fps=fps),
        dpi=dpi,
        progress_callback=lambda frame, total: print(
            f"Rendering frame {frame + 1}/{total}", end="\r", flush=True
        ),
    )
    plt.close(fig)
    print(f"\nSaved {output_path}")
    print(f"Frames: {len(frame_numbers)}")
    print(f"Duration: {len(frame_numbers) / fps:.2f} s per loop")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hd-dir", type=Path, default=DEFAULT_HD_DIR)
    parser.add_argument("--rhd-dir", type=Path, default=DEFAULT_RHD_DIR)
    parser.add_argument("--output", type=Path, default=REPO_DIR / "runs" / "hd_rhd_evolution.gif")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    animate_hd_rhd_comparison(
        args.hd_dir,
        args.rhd_dir,
        output=args.output,
        fps=args.fps,
        dpi=args.dpi,
        preview=args.preview,
    )


if __name__ == "__main__":
    main()
