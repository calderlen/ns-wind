"""Plot HD and RHD initial/current neutron-star wind profiles together."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


from wind_common import (
    REPO_DIR, derive_profiles, read_definitions, read_parameters,
    read_grid, read_catalog, read_snapshot,
)
from plot_helpers import padded_log_limits, padded_linear_limits

DEFAULT_HD_DIR = REPO_DIR / "problems" / "hd"
DEFAULT_RHD_DIR = REPO_DIR / "problems" / "rhd"


def load_run(
    run_dir: Path,
    current_snapshot: int | None = None,
) -> dict[str, object]:
    """Load the initial and requested/latest completed state from one run."""
    run_dir = run_dir.resolve()
    physics, units = read_definitions(run_dir / "definitions.h")
    parameters = read_parameters(run_dir / "pluto.ini")
    radius = read_grid(run_dir / "grid.out")
    catalog = read_catalog(run_dir / "dbl.out")
    current_number = max(catalog) if current_snapshot is None else current_snapshot
    initial_snapshot = read_snapshot(run_dir, 0, radius, catalog)
    current = read_snapshot(run_dir, current_number, radius, catalog)
    time_unit = units["UNIT_LENGTH"] / units["UNIT_VELOCITY"]
    return {
        "physics": physics,
        "initial": derive_profiles(initial_snapshot, radius, physics, units, parameters),
        "current": derive_profiles(current, radius, physics, units, parameters),
        "current_number": current_number,
        "current_time_ms": float(current["time"]) * time_unit * 1.0e3,
    }


def plot_hd_rhd_comparison(
    hd_dir: Path | str = DEFAULT_HD_DIR,
    rhd_dir: Path | str = DEFAULT_RHD_DIR,
    *,
    hd_snapshot: int | None = None,
    rhd_snapshot: int | None = None,
    output: Path | str = REPO_DIR / "runs" / "hd_rhd_comparison.png",
) -> Path:
    """Plot HD/RHD initial and current states on the same six axes.

    ``None`` selects the latest completed snapshot listed in that run's
    ``dbl.out``. The returned path points to the saved PNG.
    """
    hd = load_run(Path(hd_dir), hd_snapshot)
    rhd = load_run(Path(rhd_dir), rhd_snapshot)
    if hd["physics"] != "HD" or rhd["physics"] != "RHD":
        raise ValueError("hd_dir must contain HD output and rhd_dir RHD output")

    states = [
        ("HD initial", hd["initial"], "#73a9d8"),
        ("HD current", hd["current"], "#08519c"),
        ("RHD initial", rhd["initial"], "#fdae6b"),
        ("RHD current", rhd["current"], "#d94801"),
    ]

    fig, axes = plt.subplots(
        3, 2, figsize=(8.6, 9.2), sharex=True,
        gridspec_kw={"hspace": 0.0, "wspace": 0.055},
    )
    fig.subplots_adjust(left=0.10, right=0.855, bottom=0.075, top=0.935)

    # Density, pressure, temperature, Bernoulli, and mass flux.
    panel_specs = [
        ((0, 0), "density", r"$\rho\;[\mathrm{g\,cm^{-3}}]$", "log"),
        ((1, 0), "pressure", r"$P\;[\mathrm{ergs\,cm^{-3}}]$", "log"),
        ((2, 0), "temperature", r"$T\;[\mathrm{K}]$", "log"),
        ((1, 1), "bernoulli_c2", r"$\mathrm{Be}\;[c^2]$", "linear"),
        ((2, 1), "mdot_g_s", r"$\dot{M}\;[\mathrm{g\,s^{-1}}]$", "log"),
    ]
    for (row, col), key, ylabel, scale in panel_specs:
        axis = axes[row, col]
        for label, profile, color in states:
            axis.plot(profile["radius_km"], profile[key], color=color, lw=1.55, label=label)
        axis.set_xscale("log")
        axis.set_yscale(scale)
        axis.set_ylabel(ylabel)
        arrays = [profile[key] for _, profile, _ in states]
        if scale == "log":
            axis.set_ylim(*padded_log_limits(arrays))
        else:
            axis.set_ylim(*padded_linear_limits(arrays, include_zero=True))

    axes[0, 0].legend(loc="upper right", fontsize=8, frameon=True)
    axes[1, 1].axhline(0.0, color="black", lw=1.2, ls=":")

    # Velocity and sound speed. Color identifies the state; line style
    # identifies the plotted speed.
    speed_axis = axes[0, 1]
    for _, profile, color in states:
        speed_axis.plot(profile["radius_km"], profile["velocity_c"], color=color, lw=1.55)
        speed_axis.plot(
            profile["radius_km"], profile["sound_speed_c"],
            color=color, lw=1.4, ls="--",
        )
    hd_initial = hd["initial"]
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
    for axis in axes.ravel():
        axis.axvline(12.0, color="0.40", lw=1.15, ls=":", zorder=0.5)
        axis.set_xlim(
            min(hd["initial"]["radius_km"].min(), rhd["initial"]["radius_km"].min()),
            max(hd["initial"]["radius_km"].max(), rhd["initial"]["radius_km"].max()),
        )
    axes[2, 0].set_xlabel(r"$r\;[\mathrm{km}]$")
    axes[2, 1].set_xlabel(r"$r\;[\mathrm{km}]$")
    fig.suptitle(
        f"HD current: {hd['current_time_ms']:.1f} ms (#{hd['current_number']})   |   "
        f"RHD current: {rhd['current_time_ms']:.1f} ms (#{rhd['current_number']})"
    )

    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")
    print(f"HD current snapshot: {hd['current_number']} ({hd['current_time_ms']:.3f} ms)")
    print(f"RHD current snapshot: {rhd['current_number']} ({rhd['current_time_ms']:.3f} ms)")
    return output_path


if __name__ == "__main__":
    plot_hd_rhd_comparison()
