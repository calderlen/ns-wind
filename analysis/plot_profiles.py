"""Plot the diagnostics requested for the 1D PLUTO wind calculation.

This script reads PLUTO's ``single_file`` double-precision output directly.
It intentionally does not use the bundled pyPLUTO reader, which still calls
the removed Python ``array.fromstring`` method.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


from wind_common import (
    REPO_DIR, C_CGS, M_SUN, derive_profiles, read_definitions, read_parameters,
    read_grid, read_catalog, read_snapshot,
)

DEFAULT_RUN_DIR = REPO_DIR / "problems" / "hd"


def diagnostics(snapshot, radius_code, parameters, units):
    """Use shared physics while retaining the HD figures' velocity-unit scale.

    Their historical c label uses UNIT_VELOCITY (a rounded c in this run).
    The HD/RHD comparison and numerical exports use C_CGS instead.
    """
    profile = derive_profiles(snapshot, radius_code, "HD", units, parameters)
    scale = C_CGS / units["UNIT_VELOCITY"]
    for key in ("velocity_c", "sound_speed_c", "escape_velocity_c"):
        profile[key] = profile[key] * scale
    profile["bernoulli_c2"] = profile["bernoulli_c2"] * scale**2
    return profile


def relative_spread(values: np.ndarray, mask: np.ndarray) -> float:
    """Return the robust 1st-to-99th percentile spread relative to the median."""
    selected = values[mask]
    median = np.median(selected)
    low, high = np.percentile(selected, [1.0, 99.0])
    return float((high - low) / abs(median))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--initial", type=int, default=0)
    parser.add_argument(
        "--final",
        type=int,
        default=None,
        help="Final snapshot number (default: last entry in dbl.out)",
    )
    parser.add_argument("--show", action="store_true", help="Open an interactive window")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG (default: <run-dir>/wind_profiles_hd.png)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    physics, units = read_definitions(run_dir / "definitions.h")
    if physics != "HD":
        raise ValueError(f"Expected an HD run; found {physics}")
    parameters = read_parameters(run_dir / "pluto.ini")
    radius_km = read_grid(run_dir / "grid.out")
    catalog = read_catalog(run_dir / "dbl.out")
    final_number = max(catalog) if args.final is None else args.final

    initial_snapshot = read_snapshot(run_dir, args.initial, radius_km, catalog)
    final_snapshot = read_snapshot(run_dir, final_number, radius_km, catalog)
    initial = diagnostics(initial_snapshot, radius_km, parameters, units)
    final = diagnostics(final_snapshot, radius_km, parameters, units)

    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
    axes = axes.ravel()
    initial_style = {"color": "#d95f02", "alpha": 0.85, "lw": 2.0}
    final_style = {"color": "#1b6ca8", "lw": 2.0}

    axes[0].loglog(radius_km, initial["density"], label="Initial", **initial_style)
    axes[0].loglog(radius_km, final["density"], label="Final", **final_style)
    axes[0].set_ylabel(r"Density $\rho$ [g cm$^{-3}$]")

    axes[1].loglog(radius_km, initial["pressure"], label="Initial", **initial_style)
    axes[1].loglog(radius_km, final["pressure"], label="Final", **final_style)
    axes[1].set_ylabel(r"Pressure $P$ [dyn cm$^{-2}$]")

    axes[2].semilogx(radius_km, initial["velocity_c"], label=r"Initial $v_r$", **initial_style)
    axes[2].semilogx(radius_km, initial["sound_speed_c"], ls="--", label=r"Initial $c_s$", **initial_style)
    axes[2].semilogx(radius_km, final["velocity_c"], label=r"Final $v_r$", **final_style)
    axes[2].semilogx(radius_km, final["sound_speed_c"], ls="--", label=r"Final $c_s$", **final_style)
    axes[2].set_ylabel(r"Speed [$c$]")

    axes[3].semilogx(radius_km, (initial["mdot_g_s"] / M_SUN), label="Initial", **initial_style)
    axes[3].semilogx(radius_km, (final["mdot_g_s"] / M_SUN), label="Final", **final_style)
    axes[3].set_ylabel(r"$\dot{M}$ [$M_\odot$ s$^{-1}$]")
    axes[3].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    axes[4].semilogx(radius_km, initial["bernoulli_c2"], label="Initial", **initial_style)
    axes[4].semilogx(radius_km, final["bernoulli_c2"], label="Final", **final_style)
    axes[4].set_ylabel(r"Bernoulli parameter [$c^2$]")

    mach_initial = initial["velocity_c"] / initial["sound_speed_c"]
    mach_final = final["velocity_c"] / final["sound_speed_c"]
    axes[5].semilogx(radius_km, mach_initial, label="Initial", **initial_style)
    axes[5].semilogx(radius_km, mach_final, label="Final", **final_style)
    axes[5].axhline(1.0, color="0.3", ls=":", label="Sonic")
    axes[5].set_ylabel(r"Mach number $v_r/c_s$")

    for axis in axes:
        axis.axvline(12.0, color="0.45", ls=":", lw=1.2)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=9)
    for axis in axes[-2:]:
        axis.set_xlabel("Radius [km]")

    time_unit = units["UNIT_LENGTH"] / units["UNIT_VELOCITY"]
    final_time_seconds = float(final_snapshot["time"]) * time_unit
    fig.suptitle(
        "1D Newtonian neutron-star wind (HD)\n"
        f"snapshot {args.initial} vs {final_number}; "
        f"final physical time = {final_time_seconds:.4f} s"
    )
    fig.tight_layout()
    output_path = args.output or run_dir / "wind_profiles_hd.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")

    # Exclude the prescribed r <= 12 km region and a few edge cells when
    # measuring how constant the final invariants are.
    active_mask = (radius_km > 12.0) & (radius_km < 0.98 * radius_km.max())
    mdot_median = np.median((final["mdot_g_s"] / M_SUN)[active_mask])
    bernoulli_median = np.median(final["bernoulli_c2"][active_mask])
    print(f"Saved {output_path}")
    print(f"Final snapshot: {final_number}")
    print(f"Final physical time: {final_time_seconds:.6f} s")
    print(f"Median final Mdot: {mdot_median:.6e} Msun/s")
    print(
        "Final Mdot 1st-99th percentile relative spread: "
        f"{100.0 * relative_spread(final['mdot_g_s'] / M_SUN, active_mask):.4f}%"
    )
    print(f"Median final Bernoulli parameter: {bernoulli_median:.6e} c^2")
    print(
        "Final Bernoulli 1st-99th percentile relative spread: "
        f"{100.0 * relative_spread(final['bernoulli_c2'], active_mask):.4f}%"
    )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
