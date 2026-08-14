# High-resolution RHD convergence and parameter suite

The suite uses the completed 1,025-cell `First_Sim_RHD_MPI` calculation as the
low-resolution side of the convergence test. **Every new calculation uses the
2,050-cell high-resolution grid:**

```text
X1-grid    2    11.00    200    u    12.00    1850    s    10000.0
```

The `resolution_2x` calculation is also the high-resolution control point for
the parameter survey:

- `M_NS = 1.4`
- `GAMMA = 4/3`
- `RHO_IN = 1.0e8`
- `V_IN = 0.07`
- `V_INF = 0.25`
- 2,050 radial cells

It prepares and runs:

- One high-resolution control calculation.
- Three high-resolution neutron-star mass variants: 1.2, 1.8, and 2.0 solar masses.
- Three high-resolution terminal-speed variants: 0.15, 0.20, and 0.30 c.
- Three high-resolution inner-speed variants: 0.03, 0.05, and 0.10 c.
- Two high-resolution inner-density variants: 1e7 and 1e9 g cm^-3.

Together with the control values, this samples:

- `M_NS = 1.2, 1.4, 1.8, 2.0`
- `V_INF = 0.15, 0.20, 0.25, 0.30`
- `V_IN = 0.03, 0.05, 0.07, 0.10`
- `RHO_IN = 1e7, 1e8, 1e9`

This is a one-parameter-at-a-time study: 12 high-resolution runs, not the 192
runs required by the full Cartesian product of every possible combination.

Each PLUTO calculation uses eight MPI ranks by default. Calculations run one
after another, so only one eight-rank PLUTO job is active at any given time.

## Commands

From `/Users/calder/code/PLUTO/Calder`:

```bash
./run_rhd_suite.sh --list
./run_rhd_suite.sh --prepare-only
./run_rhd_suite.sh
```

Run only the convergence calculation:

```bash
./run_rhd_suite.sh --group resolution
```

Run only the parameter survey:

```bash
./run_rhd_suite.sh --group parameters
```

Run selected experiments:

```bash
./run_rhd_suite.sh --only M_NS_1p2 V_INF_0p15
```

Use a different number of MPI ranks:

```bash
RANKS=4 ./run_rhd_suite.sh --only M_NS_1p2
```

## Output and restart behavior

Runs are stored under `Calder/RHD_experiments/<experiment-name>/`. Terminal
output is also recorded in `run.log`, and successful runs receive a
`run_complete.json` marker containing their elapsed time and final snapshot.

Re-running the launcher skips calculations with a completion marker. If a run
was interrupted after writing simulation output, the launcher stops rather
than silently overwriting that output. Move the incomplete experiment directory
aside (or inspect it for a PLUTO checkpoint restart) before relaunching it.
