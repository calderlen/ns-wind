"""Plot the diagnostics requested for the 1D PLUTO wind calculation.

This script reads PLUTO's ``single_file`` double-precision output directly.
It intentionally does not use the bundled pyPLUTO reader, which still calls
the removed Python ``array.fromstring`` method.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_DIR / "problems" / "hd"
G_CGS = 6.674e-8
M_SUN = 1.988e33
A_RAD_CGS = 7.5657e-15


def read_definitions(path: Path) -> dict[str, float]:
    """Read numerical UNIT_* macros from definitions.h."""
    values: dict[str, float] = {}
    pattern = re.compile(r"^\s*#define\s+(UNIT_\w+)\s+([^/\s]+)")
    for line in path.read_text().splitlines():
        match = pattern.match(line)
        if match:
            values[match.group(1)] = float(match.group(2))
    return values


def read_parameters(path: Path) -> dict[str, float]:
    """Read values from the [Parameters] block in pluto.ini."""
    values: dict[str, float] = {}
    in_parameters = False
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            in_parameters = line.lower() == "[parameters]"
            continue
        if in_parameters:
            name, value, *_ = line.split()
            values[name] = float(value)
    return values


def read_grid(path: Path) -> np.ndarray:
    """Return the radial cell centers from grid.out."""
    lines = [line for line in path.read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    n1 = int(lines[0])
    edges = np.array(
        [[float(value) for value in line.split()[1:3]]
         for line in lines[1:n1 + 1]]
    )
    return edges.mean(axis=1)


def read_output_catalog(path: Path) -> dict[int, dict[str, object]]:
    """Read snapshot metadata from dbl.out."""
    catalog: dict[int, dict[str, object]] = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        number = int(fields[0])
        catalog[number] = {
            "time": float(fields[1]),
            "endian": fields[5],
            "variables": fields[6:],
        }
    return catalog


def read_snapshot(
    run_dir: Path,
    number: int,
    radius: np.ndarray,
    catalog: dict[int, dict[str, object]],
) -> dict[str, np.ndarray | float]:
    """Read one variable-major PLUTO .dbl snapshot."""
    metadata = catalog[number]
    variables = metadata["variables"]
    endian = "<" if metadata["endian"] == "little" else ">"
    raw = np.fromfile(run_dir / f"data.{number:04d}.dbl", dtype=f"{endian}f8")
    expected_size = len(variables) * radius.size
    if raw.size != expected_size:
        raise ValueError(
            f"Snapshot {number} contains {raw.size} values; expected "
            f"{expected_size} ({len(variables)} variables x {radius.size} cells)."
        )
    arrays = raw.reshape(len(variables), radius.size)
    snapshot: dict[str, np.ndarray | float] = dict(zip(variables, arrays))
    snapshot["time"] = float(metadata["time"])
    return snapshot


def diagnostics(
    snapshot: dict[str, np.ndarray | float],
    radius_km: np.ndarray,
    parameters: dict[str, float],
    units: dict[str, float],
) -> dict[str, np.ndarray]:
    """Convert primitive variables to cgs and derive wind diagnostics."""
    rho_code = np.asarray(snapshot["rho"])
    velocity_code = np.asarray(snapshot["vx1"])
    pressure_code = np.asarray(snapshot["prs"])

    density_unit = units["UNIT_DENSITY"]
    length_unit = units["UNIT_LENGTH"]
    velocity_unit = units["UNIT_VELOCITY"]
    pressure_unit = density_unit * velocity_unit**2
    gamma = parameters["GAMMA"]

    radius_cm = radius_km * length_unit
    density = rho_code * density_unit
    velocity = velocity_code * velocity_unit
    pressure = pressure_code * pressure_unit
    sound_speed = np.sqrt(gamma * pressure / density)
    mdot_g_s = 4.0 * np.pi * radius_cm**2 * density * velocity
    phi = -G_CGS * parameters["M_NS"] * M_SUN / radius_cm
    escape_velocity = np.sqrt(-2.0 * phi)
    bernoulli = (
        0.5 * velocity**2
        + gamma / (gamma - 1.0) * pressure / density
        + phi
    ) / velocity_unit**2

    return {
        "density": density,
        "pressure": pressure,
        "velocity_c": velocity / velocity_unit,
        "sound_speed_c": sound_speed / velocity_unit,
        "escape_velocity_c": escape_velocity / velocity_unit,
        "temperature": (3.0 * pressure / A_RAD_CGS) ** 0.25,
        "mdot_g_s": mdot_g_s,
        "mdot": mdot_g_s / M_SUN,
        "bernoulli_c2": bernoulli,
    }


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
    units = read_definitions(run_dir / "definitions.h")
    parameters = read_parameters(run_dir / "pluto.ini")
    radius_km = read_grid(run_dir / "grid.out")
    catalog = read_output_catalog(run_dir / "dbl.out")
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

    axes[3].semilogx(radius_km, initial["mdot"], label="Initial", **initial_style)
    axes[3].semilogx(radius_km, final["mdot"], label="Final", **final_style)
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
    mdot_median = np.median(final["mdot"][active_mask])
    bernoulli_median = np.median(final["bernoulli_c2"][active_mask])
    print(f"Saved {output_path}")
    print(f"Final snapshot: {final_number}")
    print(f"Final physical time: {final_time_seconds:.6f} s")
    print(f"Median final Mdot: {mdot_median:.6e} Msun/s")
    print(
        "Final Mdot 1st-99th percentile relative spread: "
        f"{100.0 * relative_spread(final['mdot'], active_mask):.4f}%"
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
