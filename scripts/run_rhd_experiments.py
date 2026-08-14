#!/usr/bin/env python3
"""Prepare and run the high-resolution 1D RHD parameter-study suite.

Each PLUTO calculation uses MPI internally.  Calculations are deliberately
launched sequentially so multiple eight-rank jobs do not compete for the same
CPU cores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_DIR / "problems" / "rhd_mpi"
SUITE_DIR = REPO_DIR / "runs" / "rhd_experiments"
FILES_TO_COPY = ("init.c", "definitions.h", "makefile", "pluto.ini", "pluto")
HIGH_RES_X1_GRID = (
    "X1-grid    2    11.00    200    u    12.00    1850    s    10000.0"
)


@dataclass(frozen=True)
class Experiment:
    name: str
    group: str
    varied_quantity: str
    value: str
    description: str
    x1_grid: str | None = None


EXPERIMENTS = (
    Experiment(
        "resolution_2x",
        "resolution",
        "radial_cells",
        "2050",
        "Twice the baseline radial resolution in both X1 grid patches.",
        HIGH_RES_X1_GRID,
    ),
    Experiment("M_NS_1p2", "parameters", "M_NS", "1.2", "Neutron-star mass 1.2 Msun."),
    Experiment("M_NS_1p8", "parameters", "M_NS", "1.8", "Neutron-star mass 1.8 Msun."),
    Experiment("M_NS_2p0", "parameters", "M_NS", "2.0", "Neutron-star mass 2.0 Msun."),
    Experiment("V_INF_0p15", "parameters", "V_INF", "0.15", "Terminal speed 0.15 c."),
    Experiment("V_INF_0p20", "parameters", "V_INF", "0.20", "Terminal speed 0.20 c."),
    Experiment("V_INF_0p30", "parameters", "V_INF", "0.30", "Terminal speed 0.30 c."),
    Experiment("V_IN_0p03", "parameters", "V_IN", "0.03", "Inner-boundary speed 0.03 c."),
    Experiment("V_IN_0p05", "parameters", "V_IN", "0.05", "Inner-boundary speed 0.05 c."),
    Experiment("V_IN_0p10", "parameters", "V_IN", "0.10", "Inner-boundary speed 0.10 c."),
    Experiment("RHO_IN_1e7", "parameters", "RHO_IN", "1.0e7", "Inner density 1e7 g cm^-3."),
    Experiment("RHO_IN_1e9", "parameters", "RHO_IN", "1.0e9", "Inner density 1e9 g cm^-3."),
)

BASELINE_VALUES = {
    "M_NS": "1.4",
    "GAMMA": "1.333333333333",
    "RHO_IN": "1.0e8",
    "V_IN": "0.07",
    "V_INF": "0.25",
}


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_parameter(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^(?P<prefix>\s*{re.escape(key)}\s+)\S+(?P<suffix>\s*)$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {key} line in pluto.ini; found {len(matches)}")
    return pattern.sub(lambda match: f"{match.group('prefix')}{value}{match.group('suffix')}", text)


def replace_x1_grid(text: str, replacement: str) -> str:
    pattern = re.compile(r"^\s*X1-grid\s+.*$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one X1-grid line in pluto.ini; found {len(matches)}")
    return pattern.sub(replacement, text)


def configured_pluto_ini(experiment: Experiment) -> str:
    """Render an experiment from the original control configuration."""
    ini_text = (BASELINE_DIR / "pluto.ini").read_text()
    ini_text = replace_x1_grid(ini_text, HIGH_RES_X1_GRID)
    if experiment.varied_quantity in BASELINE_VALUES:
        ini_text = replace_parameter(
            ini_text, experiment.varied_quantity, experiment.value
        )
    elif experiment.varied_quantity != "radial_cells":
        raise RuntimeError(f"No pluto.ini edit defined for {experiment.name}")
    return ini_text


def select_experiments(names: list[str] | None, group: str) -> list[Experiment]:
    selected = list(EXPERIMENTS)
    if group != "all":
        selected = [experiment for experiment in selected if experiment.group == group]
    if names:
        wanted = set(names)
        known = {experiment.name for experiment in EXPERIMENTS}
        unknown = wanted - known
        if unknown:
            raise SystemExit(f"Unknown experiment(s): {', '.join(sorted(unknown))}")
        selected = [experiment for experiment in selected if experiment.name in wanted]
    return selected


def ensure_baseline() -> None:
    missing = [name for name in FILES_TO_COPY if not (BASELINE_DIR / name).is_file()]
    if missing:
        raise SystemExit(
            f"Baseline directory {BASELINE_DIR} is missing: {', '.join(missing)}"
        )


def prepare_experiment(experiment: Experiment) -> Path:
    run_dir = SUITE_DIR / experiment.name
    metadata_path = run_dir / "experiment.json"
    expected_ini = configured_pluto_ini(experiment)

    if run_dir.exists():
        if not metadata_path.is_file():
            raise RuntimeError(
                f"Refusing to use existing unrecognized directory: {run_dir}"
            )
        existing = json.loads(metadata_path.read_text())
        if existing.get("experiment") != asdict(experiment):
            raise RuntimeError(f"Experiment definition changed for {run_dir}")
        ini_path = run_dir / "pluto.ini"
        if ini_path.read_text() != expected_ini:
            if has_partial_output(run_dir):
                raise RuntimeError(
                    f"Cannot update the grid in {run_dir}: it already contains run output"
                )
            ini_path.write_text(expected_ini)
            existing["updated_at"] = timestamp()
            existing["copied_file_sha256"]["pluto.ini"] = sha256(ini_path)
            metadata_path.write_text(json.dumps(existing, indent=2) + "\n")
            print(f"Updated to high resolution: {experiment.name}")
        else:
            print(f"Prepared already: {experiment.name}")
        return run_dir

    run_dir.mkdir(parents=True)
    for filename in FILES_TO_COPY:
        shutil.copy2(BASELINE_DIR / filename, run_dir / filename)

    ini_path = run_dir / "pluto.ini"
    ini_path.write_text(expected_ini)

    metadata = {
        "experiment": asdict(experiment),
        "prepared_at": timestamp(),
        "baseline_directory": str(BASELINE_DIR),
        "baseline_values": BASELINE_VALUES,
        "copied_file_sha256": {
            filename: sha256(run_dir / filename) for filename in FILES_TO_COPY
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Prepared: {experiment.name}")
    return run_dir


def write_suite_manifest() -> None:
    SUITE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "baseline": {
            "directory": str(BASELINE_DIR),
            "values": BASELINE_VALUES,
            "description": "Completed 1025-cell control used only for convergence comparison.",
        },
        "survey_control": {
            "directory": str(SUITE_DIR / "resolution_2x"),
            "values": BASELINE_VALUES,
            "description": "2050-cell control used as the reference for every parameter variant.",
        },
        "experiments": [asdict(experiment) for experiment in EXPERIMENTS],
        "grid": HIGH_RES_X1_GRID,
        "execution": "All experiments use the 2050-cell grid and run sequentially; each uses MPI internally.",
    }
    (SUITE_DIR / "suite_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def read_tstop(ini_path: Path) -> float:
    pattern = re.compile(r"^\s*tstop\s+(\S+)", re.MULTILINE)
    match = pattern.search(ini_path.read_text())
    if match is None:
        raise RuntimeError(f"Could not read tstop from {ini_path}")
    return float(match.group(1))


def final_output_record(run_dir: Path) -> tuple[int, float] | None:
    dbl_out = run_dir / "dbl.out"
    if not dbl_out.is_file():
        return None
    records = [
        line.split()
        for line in dbl_out.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not records:
        return None
    return int(records[-1][0]), float(records[-1][1])


def validate_completion(run_dir: Path) -> tuple[int, float]:
    record = final_output_record(run_dir)
    if record is None:
        raise RuntimeError(f"Run exited successfully but {run_dir / 'dbl.out'} is incomplete")
    output_number, final_time = record
    tstop = read_tstop(run_dir / "pluto.ini")
    tolerance = max(1.0e-8, abs(tstop) * 1.0e-8)
    if final_time < tstop - tolerance:
        raise RuntimeError(
            f"Run stopped at t={final_time:g}, before requested tstop={tstop:g}"
        )
    output_path = run_dir / f"data.{output_number:04d}.dbl"
    if not output_path.is_file():
        raise RuntimeError(f"Final output listed in dbl.out is missing: {output_path}")
    return output_number, final_time


def has_partial_output(run_dir: Path) -> bool:
    output_names = ("dbl.out", "grid.out", "restart.out", "run.log")
    return any((run_dir / name).exists() for name in output_names) or any(
        run_dir.glob("data.*.dbl")
    )


def run_experiment(experiment: Experiment, ranks: int) -> str:
    run_dir = SUITE_DIR / experiment.name
    complete_path = run_dir / "run_complete.json"
    if complete_path.is_file():
        completed = json.loads(complete_path.read_text())
        print(
            f"Skipping completed: {experiment.name} "
            f"({completed.get('elapsed_seconds', '?')} s)"
        )
        return "skipped"

    if has_partial_output(run_dir):
        raise RuntimeError(
            f"{run_dir} contains output from an incomplete run. Move that directory "
            "aside or remove its generated output before trying again."
        )

    command = ["mpirun", "-n", str(ranks), "./pluto", "-dec", str(ranks)]
    started_at = timestamp()
    start = time.monotonic()
    status_path = run_dir / "run_status.json"
    status_path.write_text(
        json.dumps(
            {
                "state": "running",
                "started_at": started_at,
                "command": command,
                "mpi_ranks": ranks,
            },
            indent=2,
        )
        + "\n"
    )

    print("\n" + "=" * 78)
    print(f"Starting {experiment.name} at {started_at}")
    print(f"Directory: {run_dir}")
    print(f"Command: {' '.join(command)}")
    print("=" * 78)

    log_path = run_dir / "run.log"
    process: subprocess.Popen[str] | None = None
    try:
        with log_path.open("w", buffering=1) as log:
            process = subprocess.Popen(
                command,
                cwd=run_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
            return_code = process.wait()
    except KeyboardInterrupt:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        raise

    elapsed = time.monotonic() - start
    finished_at = timestamp()
    if return_code != 0:
        status_path.write_text(
            json.dumps(
                {
                    "state": "failed",
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "elapsed_seconds": round(elapsed, 2),
                    "return_code": return_code,
                    "command": command,
                },
                indent=2,
            )
            + "\n"
        )
        raise RuntimeError(f"{experiment.name} failed with exit code {return_code}")

    output_number, final_time = validate_completion(run_dir)
    complete = {
        "state": "complete",
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": round(elapsed, 2),
        "mpi_ranks": ranks,
        "command": command,
        "final_output_number": output_number,
        "final_simulation_time": final_time,
    }
    complete_path.write_text(json.dumps(complete, indent=2) + "\n")
    status_path.write_text(json.dumps(complete, indent=2) + "\n")
    print(f"Completed {experiment.name} in {elapsed / 60.0:.2f} minutes")
    return "complete"


def list_experiments() -> None:
    print(f"Original 1025-cell convergence control: {BASELINE_DIR}")
    print("Every listed experiment uses the 2050-cell high-resolution grid.")
    print("resolution_2x supplies the baseline parameter point for the survey.\n")
    print(f"{'experiment':<18} {'group':<11} {'varied quantity':<17} value")
    print("-" * 65)
    for experiment in EXPERIMENTS:
        print(
            f"{experiment.name:<18} {experiment.group:<11} "
            f"{experiment.varied_quantity:<17} {experiment.value}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, default=8, help="MPI ranks per PLUTO run (default: 8)")
    parser.add_argument(
        "--group",
        choices=("all", "resolution", "parameters"),
        default="all",
        help="Run the full suite or one group (default: all)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="NAME",
        help="Run only the named experiment(s); see --list",
    )
    parser.add_argument("--prepare-only", action="store_true", help="Create run directories without launching PLUTO")
    parser.add_argument("--list", action="store_true", help="List experiments and exit")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue to the next experiment if one run fails",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ranks < 1:
        raise SystemExit("--ranks must be at least 1")
    if args.list:
        list_experiments()
        return

    ensure_baseline()
    selected = select_experiments(args.only, args.group)
    if not selected:
        raise SystemExit("No experiments selected")

    write_suite_manifest()
    for experiment in selected:
        prepare_experiment(experiment)

    if args.prepare_only:
        print(f"\nPrepared {len(selected)} experiment(s) under {SUITE_DIR}")
        return

    completed = 0
    skipped = 0
    failed: list[str] = []
    for experiment in selected:
        try:
            outcome = run_experiment(experiment, args.ranks)
            if outcome == "complete":
                completed += 1
            else:
                skipped += 1
        except KeyboardInterrupt:
            print("\nInterrupted. Completed experiments will be skipped next time.")
            raise SystemExit(130)
        except Exception as error:
            failed.append(experiment.name)
            print(f"ERROR: {error}", file=sys.stderr)
            if not args.keep_going:
                raise SystemExit(1)

    print("\nSuite summary")
    print(f"  newly completed: {completed}")
    print(f"  already complete: {skipped}")
    print(f"  failed: {len(failed)}")
    if failed:
        print(f"  failed names: {', '.join(failed)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
