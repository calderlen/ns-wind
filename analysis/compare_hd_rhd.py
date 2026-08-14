"""Plot HD and RHD initial/current neutron-star wind profiles together."""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_HD_DIR = REPO_DIR / "problems" / "hd"
DEFAULT_RHD_DIR = REPO_DIR / "problems" / "rhd"

C_CGS = 2.99792458e10
G_CGS = 6.674e-8
M_SUN = 1.988e33
A_RAD_CGS = 7.5657e-15


def read_definitions(path: Path) -> tuple[str, dict[str, float]]:
    """Read the physics module and numerical UNIT_* definitions."""
    physics = ""
    units: dict[str, float] = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 3 or fields[0] != "#define":
            continue
        if fields[1] == "PHYSICS":
            physics = fields[2]
        elif fields[1].startswith("UNIT_"):
            try:
                units[fields[1]] = float(fields[2])
            except ValueError:
                pass
    if physics not in {"HD", "RHD"}:
        raise ValueError(f"Expected HD or RHD in {path}; found {physics!r}")
    # RHD uses c=1 internally. PLUTO supplies UNIT_VELOCITY=CONST_c when the
    # user correctly omits an explicit velocity unit from definitions.h.
    units.setdefault("UNIT_VELOCITY", C_CGS if physics == "RHD" else np.nan)
    return physics, units


def read_parameters(path: Path) -> dict[str, float]:
    """Read numerical values from the [Parameters] block in pluto.ini."""
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
    """Return radial cell centers from PLUTO's grid.out."""
    lines = [
        line for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    n1 = int(lines[0])
    edges = np.array(
        [[float(value) for value in line.split()[1:3]]
         for line in lines[1:n1 + 1]]
    )
    return edges.mean(axis=1)


def read_catalog(path: Path) -> dict[int, dict[str, object]]:
    """Read the completed-snapshot catalog in dbl.out."""
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
    """Read one variable-major, double-precision PLUTO snapshot."""
    metadata = catalog[number]
    variables = metadata["variables"]
    endian = "<" if metadata["endian"] == "little" else ">"
    path = run_dir / f"data.{number:04d}.dbl"
    raw = np.fromfile(path, dtype=f"{endian}f8")
    expected = len(variables) * radius.size
    if raw.size != expected:
        raise ValueError(f"{path} has {raw.size} values; expected {expected}")
    arrays = raw.reshape(len(variables), radius.size)
    snapshot: dict[str, np.ndarray | float] = dict(zip(variables, arrays))
    snapshot["time"] = float(metadata["time"])
    return snapshot


def derive_profiles(
    snapshot: dict[str, np.ndarray | float],
    radius_km: np.ndarray,
    physics: str,
    units: dict[str, float],
    parameters: dict[str, float],
) -> dict[str, np.ndarray]:
    """Convert primitive variables and calculate HD/RHD wind diagnostics."""
    density_unit = units["UNIT_DENSITY"]
    length_unit = units["UNIT_LENGTH"]
    velocity_unit = units["UNIT_VELOCITY"]
    gamma = parameters["GAMMA"]

    radius_cm = radius_km * length_unit
    density = np.asarray(snapshot["rho"]) * density_unit
    velocity = np.asarray(snapshot["vx1"]) * velocity_unit
    pressure = np.asarray(snapshot["prs"]) * density_unit * velocity_unit**2
    beta = velocity / C_CGS
    potential = -G_CGS * parameters["M_NS"] * M_SUN / radius_cm
    escape_velocity_c = np.sqrt(-2.0 * potential) / C_CGS

    if physics == "HD":
        sound_speed_c = np.sqrt(gamma * pressure / density) / C_CGS
        lorentz = np.ones_like(beta)
        bernoulli_c2 = (
            0.5 * velocity**2
            + gamma / (gamma - 1.0) * pressure / density
            + potential
        ) / C_CGS**2
    else:
        if np.any(np.abs(beta) >= 1.0):
            raise ValueError("RHD snapshot contains |v| >= c")
        lorentz = 1.0 / np.sqrt(1.0 - beta**2)
        # Ideal relativistic EOS: h/c^2 = 1 + Gamma/(Gamma-1) P/(rho c^2).
        enthalpy = 1.0 + gamma / (gamma - 1.0) * pressure / (density * C_CGS**2)
        sound_speed_c = np.sqrt(gamma * pressure / (density * C_CGS**2 * enthalpy))
        # Rest-mass-subtracted SR Bernoulli parameter with the imposed
        # Newtonian gravitational potential. It reduces to the HD expression
        # in the non-relativistic limit.
        bernoulli_c2 = enthalpy * lorentz - 1.0 + potential / C_CGS**2

    mdot_g_s = 4.0 * np.pi * radius_cm**2 * density * lorentz * velocity
    return {
        "radius_km": radius_km,
        "density": density,
        "velocity_c": beta,
        "sound_speed_c": sound_speed_c,
        "escape_velocity_c": escape_velocity_c,
        "pressure": pressure,
        "bernoulli_c2": bernoulli_c2,
        "temperature": (3.0 * pressure / A_RAD_CGS) ** 0.25,
        "mdot_g_s": mdot_g_s,
    }


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


def padded_log_limits(arrays: list[np.ndarray]) -> tuple[float, float]:
    values = np.concatenate(arrays)
    values = values[np.isfinite(values) & (values > 0.0)]
    return float(values.min() / 1.35), float(values.max() * 1.35)


def padded_linear_limits(
    arrays: list[np.ndarray],
    include_zero: bool = False,
) -> tuple[float, float]:
    values = np.concatenate(arrays)
    values = values[np.isfinite(values)]
    low, high = float(values.min()), float(values.max())
    if include_zero:
        low, high = min(low, 0.0), max(high, 0.0)
    span = high - low or max(abs(low), 1.0)
    return low - 0.12 * span, high + 0.12 * span


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
