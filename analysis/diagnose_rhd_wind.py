"""Export numerical wind diagnostics from every saved RHD output (no plots).

The supported equation contract is the current spherical, ideal-EOS,
VECTOR-gravity, ENTROPY_SWITCH=ALWAYS setup. See DIAGNOSTICS.md for the
source-consistent Bernoulli balance and the two critical numerators.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile

import numpy as np

from wind_common import (
    C_CGS, G_CGS, M_SUN, REPO_DIR, read_catalog, read_definitions,
    read_grid, read_parameters, read_snapshot, derive_profiles, zero_crossings,
)


def cumulative_trapezoid(values, radius):
    return np.r_[0.0, np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(radius))]


def radial_diagnostics(snapshot, radius_code, units, parameters, active):
    """Return cgs radial arrays, preserving the sign of velocity/flux."""
    radius_cm = np.asarray(radius_code) * units["UNIT_LENGTH"]
    rho = np.asarray(snapshot["rho"]) * units["UNIT_DENSITY"]
    velocity = np.asarray(snapshot["vx1"]) * units["UNIT_VELOCITY"]
    pressure = np.asarray(snapshot["prs"]) * units["UNIT_DENSITY"] * units["UNIT_VELOCITY"]**2
    if any(array.shape != radius_cm.shape for array in (rho, velocity, pressure)):
        raise ValueError("Primitive arrays must match the radial grid")
    if not all(np.all(np.isfinite(array)) for array in (rho, velocity, pressure)):
        raise ValueError("Non-finite primitive variables")
    if np.any(rho <= 0) or np.any(pressure <= 0) or np.any(np.abs(velocity) >= C_CGS):
        raise ValueError("Require rho > 0, P > 0, and |v_r| < c")
    for key in ("vx2", "vx3"):
        if key in snapshot and np.any(np.asarray(snapshot[key]) != 0):
            raise ValueError("This exporter supports purely radial velocity only")
    Gamma = parameters["GAMMA"]
    if not 1 < Gamma <= 2 or parameters["M_NS"] <= 0:
        raise ValueError("Require 1 < GAMMA <= 2 and M_NS > 0")
    beta = velocity / C_CGS
    fluid = derive_profiles(snapshot, radius_code, "RHD", units, parameters)
    gamma = fluid["lorentz_gamma"]
    theta = pressure / (rho * C_CGS**2)
    h_minus_one = Gamma / (Gamma - 1.0) * theta
    h = fluid["enthalpy_c2"]
    gravity = -G_CGS * parameters["M_NS"] * M_SUN / radius_cm**2
    phi_c2 = gravity * radius_cm / C_CGS**2
    # From S_m = (rho h gamma^2 - P) g, with advected K and mass continuity.
    force_factor = 1.0 - theta / (h * gamma**2)
    bernoulli_balance = np.full(radius_cm.size, np.nan)
    indices = np.flatnonzero(active)
    if indices.size < 2 or np.any(np.diff(indices) != 1):
        raise ValueError("Need at least two contiguous active cells")
    log_hgamma = np.log1p(h_minus_one) - 0.5 * np.log1p(-beta**2)
    integral = cumulative_trapezoid(
        force_factor[active] * gravity[active] / C_CGS**2, radius_cm[active]
    )
    bernoulli_balance[active] = log_hgamma[active] + phi_c2[indices[0]] - integral
    return {
        "rho_g_cm3": rho,
        "pressure_erg_cm3": pressure,
        "velocity_cm_s": velocity,
        "sound_speed_cm_s": fluid["sound_speed_cm_s"],
        "lorentz_gamma": gamma,
        "enthalpy_c2": h,
        "mach": velocity / fluid["sound_speed_cm_s"],
        "mdot_g_s": fluid["mdot_g_s"],
        "K_cgs": pressure / rho**Gamma,
        "potential_c2": phi_c2,
        "gravity_cm_s2": gravity,
        "gravity_factor_impl": force_factor,
        "N_newtonian_cm_s2": 2.0 * fluid["sound_speed_squared_cm_s2"] / radius_cm + gravity,
        "N_impl_cm_s2": 2.0 * fluid["sound_speed_squared_cm_s2"] / radius_cm + force_factor * gravity,
        "bernoulli_sr_potential_c2": fluid["bernoulli_c2"],
        "bernoulli_balance_c2": bernoulli_balance,
        "edot_kin_erg_s": fluid["edot_kin_erg_s"],
    }


def relative_span(values):
    values = np.asarray(values)
    scale = float(np.max(np.abs(values)))
    return float(np.ptp(values) / scale) if scale else 0.0


def scalar_diagnostics(profile, radius_km, active, *, tolerance_cells=2.0, tolerance_rel=0.05):
    """Summarize roots and sample already-derived fluxes at exact radii."""
    roots = {
        "sonic": zero_crossings(radius_km, profile["mach"] - 1.0, active),
        "stagnation": zero_crossings(radius_km, profile["velocity_cm_s"], active),
        "critical_newtonian": zero_crossings(radius_km, profile["N_newtonian_cm_s2"], active),
        "critical_impl": zero_crossings(radius_km, profile["N_impl_cm_s2"], active),
    }
    outward_sonic = [root for root in roots["sonic"] if root.direction == 1]
    sonic = outward_sonic[0] if outward_sonic else None
    row = {
        "r_sonic_km": sonic.radius_km if sonic else np.nan,
        "r_stagnation_km": roots["stagnation"][0].radius_km if roots["stagnation"] else np.nan,
    }
    x = radius_km[active]
    for radius in (20, 100, 1000):
        in_range = x[0] <= radius <= x[-1]
        for prefix, field in (("mdot", "mdot_g_s"), ("edot_kin", "edot_kin_erg_s")):
            suffix = "g_s" if prefix == "mdot" else "erg_s"
            row[f"{prefix}_{radius}_{suffix}"] = (
                float(np.interp(radius, x, profile[field][active])) if in_range else np.nan
            )
    row.update(
        r_out_km=float(x[-1]),
        mdot_out_g_s=float(profile["mdot_g_s"][active][-1]),
        edot_kin_out_erg_s=float(profile["edot_kin_erg_s"][active][-1]),
        n_sonic_crossings=len(roots["sonic"]),
        n_sonic_outward_crossings=len(outward_sonic),
        n_stagnation_crossings=len(roots["stagnation"]),
        mdot_relative_span=relative_span(profile["mdot_g_s"][active]),
        K_relative_span=relative_span(profile["K_cgs"][active]),
        bernoulli_balance_span_c2=float(np.ptp(profile["bernoulli_balance_c2"][active])),
        bernoulli_sr_potential_span_c2=float(np.ptp(profile["bernoulli_sr_potential_c2"][active])),
        mach_min=float(np.min(profile["mach"][active])),
        mach_max=float(np.max(profile["mach"][active])),
    )
    for kind in ("newtonian", "impl"):
        candidates = roots[f"critical_{kind}"]
        row[f"n_critical_{kind}_crossings"] = len(candidates)
        closest = min(candidates, key=lambda root: abs(root.radius_km - sonic.radius_km)) if sonic and candidates else None
        row[f"r_critical_{kind}_nearest_km"] = closest.radius_km if closest else np.nan
        residual = distance = separation_cells = mach_residual = np.nan
        N_at_sonic = np.nan
        if sonic:
            rs = sonic.radius_km
            N_at_sonic = float(np.interp(rs, radius_km, profile[f"N_{kind}_cm_s2"]))
            pressure_term = 2.0 * profile["sound_speed_cm_s"]**2 / (radius_km * 1e5)
            gravity_term = profile["gravity_cm_s2"]
            if kind == "impl":
                gravity_term = gravity_term * profile["gravity_factor_impl"]
            scale = np.interp(rs, radius_km, pressure_term + np.abs(gravity_term))
            residual = float(N_at_sonic / scale)
            if closest:
                distance = closest.radius_km - rs
                widths = np.gradient(radius_km)
                spacing = max(np.interp(rs, radius_km, widths), np.interp(closest.radius_km, radius_km, widths))
                separation_cells = abs(distance) / spacing
                mach_residual = float(np.interp(closest.radius_km, radius_km, profile["mach"]) - 1.0)
        row[f"N_{kind}_at_sonic_cm_s2"] = N_at_sonic
        row[f"N_{kind}_at_sonic_relative"] = residual
        row[f"critical_{kind}_minus_sonic_km"] = distance
        row[f"critical_{kind}_separation_cells"] = separation_cells
        row[f"mach_minus_one_at_critical_{kind}"] = mach_residual
        # Candidate only, never a declaration of steadiness or regularity.
        row[f"critical_candidate_{kind}"] = int(
            closest is not None and separation_cells <= tolerance_cells and abs(residual) <= tolerance_rel
        )
    return row, roots


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_model(run_dir):
    """Fail closed for unsupported equation/boundary/source conventions."""
    definitions = (run_dir / "definitions.h").read_text()
    macros = dict(re.findall(r"^\s*#define\s+(\w+)\s+(\S+)", definitions, re.M))
    required = {"PHYSICS": "RHD", "DIMENSIONS": "1", "GEOMETRY": "SPHERICAL", "EOS": "IDEAL", "BODY_FORCE": "VECTOR", "ENTROPY_SWITCH": "ALWAYS", "COOLING": "NO", "RADIATION": "NO", "INTERNAL_BOUNDARY": "YES"}
    for key, value in required.items():
        if macros.get(key) != value:
            raise ValueError(f"Unsupported {key}={macros.get(key)} in {run_dir}; expected {value}")
    if macros.get("RMHD_REDUCED_ENERGY", "NO") != "NO":
        raise ValueError("Reduced-energy overrides are not supported by this RHD contract")
    _, units = read_definitions(run_dir / "definitions.h")
    if any(not np.isfinite(units.get(key, np.nan)) or units[key] <= 0 for key in ("UNIT_LENGTH", "UNIT_DENSITY", "UNIT_VELOCITY")):
        raise ValueError("Invalid/missing physical units")
    if not np.isclose(units["UNIT_VELOCITY"], C_CGS, rtol=1e-12):
        raise ValueError("RHD requires UNIT_VELOCITY=c")
    init_text = (run_dir / "init.c").read_text()
    init_text = re.sub(r"/\*.*?\*/|//[^\n]*", "", init_text, flags=re.S)
    radii = {float(value) for value in re.findall(r"\bR_in\s*=\s*([\d.eE+-]+)\s*;", init_text)}
    if len(radii) != 1:
        raise ValueError("Cannot identify one constant R_in in saved init.c")
    compact_init = re.sub(r"\s+", "", init_text)
    if "g[IDIR]=-G*M_ns/pow(x1,2);" not in compact_init:
        raise ValueError("Saved gravity implementation is not the supported point-mass force")
    for expression in (
        "G=6.674e-8*UNIT_DENSITY*pow(UNIT_LENGTH/UNIT_VELOCITY,2);",
        "M_ns=g_inputParam[M_NS]*1.988e33/UNIT_DENSITY/pow(UNIT_LENGTH,3);",
    ):
        if expression not in compact_init:
            raise ValueError("Saved gravitational constants/scaling differ from this exporter")
    makefile = (run_dir / "makefile").read_text()
    match = re.search(r"^PLUTO_DIR\s*=\s*(\S+)", makefile, re.M)
    if not match:
        raise ValueError("Cannot locate PLUTO_DIR from saved makefile")
    pluto_dir = Path(match.group(1))
    core_files = ["Src/RMHD/rhs_source.c", "Src/RHD/mappers.c", "Src/RHD/rhd_entropy_solve.c", "Src/EOS/Ideal/eos.c", "Src/flag_shock.c", "Src/adv_flux.c", "Src/RHD/makefile"]
    source = re.sub(r"\s+", "", (pluto_dir / core_files[0]).read_text())
    if "rhs[i][MX1]+=dt*Et*g[IDIR];" not in source or "rhs[i][ENG]+=dt*u[MX1]*g[IDIR];" not in source:
        raise ValueError("PLUTO force source differs from the supported equation contract")
    core_hashes = {str(pluto_dir / name): sha256(pluto_dir / name) for name in core_files}
    return units, read_parameters(run_dir / "pluto.ini"), radii.pop() * units["UNIT_LENGTH"] / 1e5, core_hashes


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_run(run_dir, *, tolerance_cells=2.0, tolerance_rel=0.05):
    run_dir = Path(run_dir).resolve()
    analysis_hashes = {str(Path(path).resolve()): sha256(path) for path in (__file__, read_snapshot.__code__.co_filename)}
    input_names = ("definitions.h", "init.c", "pluto.ini", "makefile", "grid.out", "dbl.out")
    input_hashes = {name: sha256(run_dir / name) for name in input_names}
    units, parameters, inner_radius_km, core_hashes = read_model(run_dir)
    radius_code = read_grid(run_dir / "grid.out")
    radius_km = radius_code * units["UNIT_LENGTH"] / 1e5
    if not np.all(np.isfinite(radius_km)) or np.any(radius_km <= 0) or np.any(np.diff(radius_km) <= 0):
        raise ValueError("Radial grid must be finite, positive, and strictly increasing")
    active = radius_km > inner_radius_km
    catalog_lines = [line.split() for line in (run_dir / "dbl.out").read_text().splitlines() if line.strip()]
    if any(len(fields) < 7 or fields[4] != "single_file" for fields in catalog_lines):
        raise ValueError("Only single_file binary catalogs are supported")
    if len({fields[0] for fields in catalog_lines}) != len(catalog_lines):
        raise ValueError("Duplicate snapshot numbers in dbl.out")
    catalog = read_catalog(run_dir / "dbl.out")
    numbers = sorted(catalog)
    if not numbers:
        raise ValueError("No saved snapshots")
    for metadata in catalog.values():
        if metadata["endian"] not in {"little", "big"}:
            raise ValueError("Unknown snapshot endianness")
    times_code = np.array([catalog[number]["time"] for number in numbers])
    if not np.all(np.isfinite(times_code)) or np.any(np.diff(times_code) <= 0):
        raise ValueError("Snapshot times must be finite and strictly increasing")
    times_s = times_code * units["UNIT_LENGTH"] / units["UNIT_VELOCITY"]
    profiles = {}
    scalar_rows, crossing_rows, snapshot_hashes = [], [], []
    for index, number in enumerate(numbers):
        snapshot_path = run_dir / f"data.{number:04d}.dbl"
        snapshot_hash = sha256(snapshot_path)
        snapshot = read_snapshot(run_dir, number, radius_code, catalog)
        if sha256(snapshot_path) != snapshot_hash:
            raise RuntimeError(f"Snapshot changed while reading: {snapshot_path}")
        try:
            profile = radial_diagnostics(snapshot, radius_code, units, parameters, active)
        except ValueError as error:
            raise ValueError(f"{run_dir.name}, snapshot {number}: {error}") from error
        if not profiles:
            profiles = {key: np.empty((len(numbers), radius_km.size)) for key in profile}
        for key, array in profile.items():
            profiles[key][index] = array
        summary, roots = scalar_diagnostics(profile, radius_km, active, tolerance_cells=tolerance_cells, tolerance_rel=tolerance_rel)
        coordinates = {"snapshot": number, "t_s": float(times_s[index]), "t_ms": float(times_s[index] * 1e3), "t_code": float(times_code[index])}
        scalar_rows.append({**coordinates, **summary})
        for kind, crossings in roots.items():
            for root in crossings:
                crossing_rows.append({**coordinates, "kind": kind, **asdict(root)})
        snapshot_hashes.append({"snapshot": number, "sha256": snapshot_hash})
    if any(sha256(run_dir / name) != digest for name, digest in input_hashes.items()):
        raise RuntimeError("Run inputs/catalog changed during export; rerun after output completes")
    if any(sha256(path) != digest for path, digest in core_hashes.items()):
        raise RuntimeError("PLUTO source changed during export")
    if any(sha256(path) != digest for path, digest in analysis_hashes.items()):
        raise RuntimeError("Analysis source changed during export")
    destination = run_dir / "diagnostics"
    destination.mkdir(exist_ok=True)
    metadata = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(run_dir), "parameters": parameters, "units": units,
        "inner_radius_km": inner_radius_km, "snapshots": len(numbers),
        "radial_cells": radius_km.size, "active_cells": int(active.sum()),
        "sampling": "Linear interpolation of derived flux in radius at exactly 20, 100, 1000 km; outer is last active cell center. No extrapolation.",
        "missing_values": "NaN: unavailable crossing, out-of-range sampling radius, or Bernoulli balance inside the fixed boundary.",
        "root_convention": "r_sonic is innermost negative-to-positive crossing of signed Mach-1. r_stagnation is innermost velocity sign change of either direction. All roots and directions are in crossings.csv.",
        "candidate_tolerances": {"separation_cells": tolerance_cells, "relative_N_residual": tolerance_rel},
        "candidate_warning": "Geometric candidates only; not proof of steadiness, smoothness, convergence, or a regular transonic solution.",
        "equation_contract": "1D spherical RHD, IDEAL EOS, VECTOR g=-GM/r^2, S_m=E_total*g, ENTROPY_SWITCH=ALWAYS (advected K, energy reset).",
        "bernoulli_balance": "b/c^2=ln(h*gamma)+Phi(r_ref)/c^2-integral[r_ref,r] (1-P/(rho*h*c^2*gamma^2))*g/c^2 dr. Trapezoidal integral; r_ref is first active center. Steady, smooth, isentropic branch diagnostic; not an escape/unbound classifier.",
        "N_newtonian": "2*c_s^2/r-G*M/r^2 (requested classical comparison).",
        "N_impl": "2*c_s^2/r-(1-P/(rho*h*c^2*gamma^2))*G*M/r^2 (mass+momentum+advected K).",
        "radial_array_axes": ["snapshot", "radial_cell"],
        "K_units": "(erg cm^-3)/(g cm^-3)^GAMMA",
        "relative_span": "(max-min)/max(abs(values)) over active cells; zero if all values are zero.",
        "input_sha256": input_hashes, "snapshot_sha256": snapshot_hashes,
        "analysis_source_sha256": analysis_hashes, "current_pluto_source_sha256": core_hashes,
        "provenance_caveat": "Core source hashes describe the current PLUTO checkout; historical core hashes were not saved with these simulation binaries.",
        "definitions_document": str(Path(__file__).with_name("DIAGNOSTICS.md")),
    }
    with tempfile.TemporaryDirectory(prefix=".export-", dir=destination) as staging_name:
        staging = Path(staging_name)
        write_csv(staging / "scalars.csv", scalar_rows, list(scalar_rows[0]))
        crossing_fields = ["snapshot", "t_s", "t_ms", "t_code", "kind", "radius_km", "left_index", "right_index", "direction"]
        write_csv(staging / "crossings.csv", crossing_rows, crossing_fields)
        np.savez_compressed(staging / "radial_profiles.npz", snapshot=np.asarray(numbers), t_s=times_s, t_ms=times_s*1e3, radius_km=radius_km, active=active, **profiles)
        metadata["output_sha256"] = {name: sha256(staging / name) for name in ("scalars.csv", "crossings.csv", "radial_profiles.npz")}
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        for name in ("radial_profiles.npz", "crossings.csv", "scalars.csv", "metadata.json"):
            (staging / name).replace(destination / name)
    print(f"{run_dir.name}: {len(numbers)} times x {radius_km.size} cells -> {destination}", flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", help="Repeat for multiple runs; default: all runs in the suite manifest")
    parser.add_argument("--suite-dir", type=Path, default=REPO_DIR / "runs/rhd_experiments")
    parser.add_argument("--critical-tolerance-cells", type=float, default=2.0)
    parser.add_argument("--critical-relative-tolerance", type=float, default=0.05)
    args = parser.parse_args()
    if not np.isfinite(args.critical_tolerance_cells) or args.critical_tolerance_cells <= 0 or not 0 < args.critical_relative_tolerance < 1:
        parser.error("Require finite positive cell tolerance and 0 < relative tolerance < 1")
    run_dirs = args.run_dir
    if run_dirs is None:
        manifest = json.loads((args.suite_dir / "suite_manifest.json").read_text())
        run_dirs = [args.suite_dir / item["name"] for item in manifest["experiments"]]
    outputs = []
    for run_dir in run_dirs:
        outputs.append(export_run(run_dir, tolerance_cells=args.critical_tolerance_cells, tolerance_rel=args.critical_relative_tolerance))
    if args.run_dir is None:
        combined = []
        for destination in outputs:
            with (destination / "scalars.csv").open(newline="") as stream:
                combined.extend({"run": destination.parent.name, **row} for row in csv.DictReader(stream))
        if not combined:
            raise ValueError("Suite manifest contains no diagnostic rows")
        destination = args.suite_dir / "diagnostics"
        destination.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".export-", dir=destination) as staging:
            path = Path(staging) / "scalars_all_runs.csv"
            write_csv(path, combined, list(combined[0]))
            path.replace(destination / path.name)
        print(f"Combined: {len(combined)} rows -> {destination / 'scalars_all_runs.csv'}", flush=True)


if __name__ == "__main__":
    main()
