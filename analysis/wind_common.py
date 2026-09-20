"""Shared readers and physical quantities for saved 1D PLUTO wind runs.

No plotting imports: numerical exports can use these functions independently.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]
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
    if physics not in {"HD", "RHD", "RMHD"}:
        raise ValueError(f"Expected HD, RHD, or RMHD in {path}; found {physics!r}")
    # Relativistic modules use c=1. PLUTO supplies UNIT_VELOCITY=CONST_c when the
    # user correctly omits an explicit velocity unit from definitions.h.
    units.setdefault("UNIT_VELOCITY", C_CGS if physics in {"RHD", "RMHD"} else np.nan)
    missing = {"UNIT_DENSITY", "UNIT_LENGTH", "UNIT_VELOCITY"} - units.keys()
    if missing or any(not np.isfinite(value) or value <= 0 for value in units.values()):
        raise ValueError(f"Missing or invalid unit definitions in {path}: {units}")
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


def derive_profiles(snapshot, radius_code, physics, units, parameters):
    """Shared fluid quantities in cgs; include transverse velocity for RMHD.

    The Bernoulli quantity is the SR-plus-potential comparison, not an
    escape criterion. Magnetic stresses are added by the RMHD caller.
    gamma is the Lorentz factor; Gamma is the adiabatic index (GAMMA input).
    """
    if physics not in {"HD", "RHD", "RMHD"}:
        raise ValueError(f"Unsupported physics: {physics}")
    radius_cm = np.asarray(radius_code) * units["UNIT_LENGTH"]
    density = np.asarray(snapshot["rho"]) * units["UNIT_DENSITY"]
    velocity = np.asarray(snapshot["vx1"]) * units["UNIT_VELOCITY"]
    pressure = np.asarray(snapshot["prs"]) * units["UNIT_DENSITY"] * units["UNIT_VELOCITY"]**2
    beta = velocity / C_CGS
    beta_squared = beta**2
    if physics == "RMHD":
        beta_squared += sum(
            (np.asarray(snapshot[key]) * units["UNIT_VELOCITY"] / C_CGS)**2
            for key in ("vx2", "vx3")
        )
    Gamma = parameters["GAMMA"]
    theta = pressure / (density * C_CGS**2)
    h_minus_one = Gamma / (Gamma - 1.0) * theta
    enthalpy = 1.0 + h_minus_one
    potential_c2 = -G_CGS * parameters["M_NS"] * M_SUN / radius_cm / C_CGS**2
    if physics == "HD":
        gamma = np.ones_like(beta)
        sound_speed_squared = Gamma * pressure / density
        kinetic_specific = 0.5 * velocity**2
        bernoulli_c2 = (
            kinetic_specific + Gamma / (Gamma - 1.0) * pressure / density
            + potential_c2 * C_CGS**2
        ) / C_CGS**2
    else:
        if np.any(beta_squared >= 1.0):
            raise ValueError(f"{physics} snapshot contains |v| >= c")
        gamma = 1.0 / np.sqrt(1.0 - beta_squared)
        sound_speed_squared = Gamma * pressure / (density * enthalpy)
        # Stable at small speeds, where direct subtraction gamma - 1 loses precision.
        gamma_minus_one = beta_squared * gamma**2 / (gamma + 1.0)
        kinetic_specific = gamma_minus_one * C_CGS**2
        bernoulli_c2 = h_minus_one + enthalpy * gamma_minus_one + potential_c2
    sound_speed = np.sqrt(sound_speed_squared)
    mdot = 4.0 * np.pi * radius_cm**2 * density * gamma * velocity
    return {
        "radius_km": radius_cm / 1e5,
        "density": density,
        "pressure": pressure,
        "velocity_c": beta,
        "sound_speed_c": sound_speed / C_CGS,
        "sound_speed_cm_s": sound_speed,
        "sound_speed_squared_cm_s2": sound_speed_squared,
        "escape_velocity_c": np.sqrt(-2.0 * potential_c2),
        "temperature": (3.0 * pressure / A_RAD_CGS)**0.25,
        "lorentz_gamma": gamma,
        "enthalpy_c2": enthalpy,
        "mdot_g_s": mdot,
        "edot_kin_erg_s": mdot * kinetic_specific,
        "bernoulli_c2": bernoulli_c2,
    }


@dataclass(frozen=True)
class Crossing:
    radius_km: float
    left_index: int
    right_index: int
    direction: int  # +1: negative to positive with increasing radius


def zero_crossings(radius_km, values, active) -> list[Crossing]:
    """Find sign changes without crossing invalid/masked gaps.

    An exact-zero plateau is represented once, at its midpoint, only when
    bracketed by opposite signs. Endpoint zeros and tangencies are not
    evidence for a crossing inside the observed domain.
    """
    radius = np.asarray(radius_km)
    values = np.asarray(values)
    valid = np.asarray(active, dtype=bool) & np.isfinite(radius) & np.isfinite(values)
    result = []
    previous = None
    for index in range(radius.size):
        if not valid[index]:
            previous = None
            continue
        if values[index] == 0:
            continue
        if previous is not None and np.sign(values[index]) != np.sign(values[previous]):
            if index == previous + 1:
                fraction = -values[previous] / (values[index] - values[previous])
                root = radius[previous] + fraction * (radius[index] - radius[previous])
            else:
                root = 0.5 * (radius[previous + 1] + radius[index - 1])
            result.append(Crossing(float(root), previous, index, int(np.sign(values[index]))))
        previous = index
    return result


def first_outward_zero_crossing(radius, values, active):
    """Return the innermost outward crossing, without bridging invalid cells."""
    return next(
        (root.radius_km for root in zero_crossings(radius, values, active)
         if root.direction == 1),
        None,
    )
