# Numerical RHD wind diagnostics

The GIFs are presentation products. The exporter and plots share their PLUTO
reader, fluid calculations, and crossing finder in `wind_common.py`, which
has no Matplotlib dependency. Axis limits, formatting, and nearest-cell
presentation histories live in `plot_helpers.py`. The exporter retains its
exact-radius interpolation.

Run the exporter independently of plotting:

```sh
conda run -n retrieval python -B analysis/diagnose_rhd_wind.py
```

This processes every output listed in each of the six suite runs' `dbl.out`,
without time decimation. For another saved run, use `--run-dir PATH` (repeatable).
It does not rerun PLUTO, change simulation inputs, or regenerate GIFs. Rerun it
after new outputs are available; this is a post-processing command, not an
in-simulation callback or background watcher.

The default suite command also writes
`runs/rhd_experiments/diagnostics/scalars_all_runs.csv`, adding a `run` column
to the combined scalar rows. Per-run metadata and radial archives remain
beside their respective simulation outputs.

Each run gets `diagnostics/` containing:

- `scalars.csv`: one row per output time, including snapshot ID, seconds,
  milliseconds, sonic/stagnation radii, signed mass and kinetic-energy fluxes
  at 20/100/1000 km and the outermost active cell center, crossing counts,
  critical-point residuals/separations, and radial variation summaries.
- `radial_profiles.npz`: full-precision arrays for every snapshot and cell.
  Profile arrays have axes `(snapshot, radial_cell)`; the one-dimensional
  coordinates are `snapshot`, `t_s`, `t_ms`, `radius_km`, and `active`.
- `crossings.csv`: every sign-changing sonic, stagnation, and critical-numerator
  root, with direction and zero-based bracketing cell indices. Multiple roots
  are not silently collapsed into one.
- `metadata.json`: units, parameters, definitions, thresholds, and input/source
  hashes. The current PLUTO source is recorded; the runs did not preserve a
  historical core-source hash, so this is not proof of historical build identity.

## Radial definitions

`gamma` denotes the Lorentz factor and `Gamma` the adiabatic index
(the `GAMMA` parameter in `pluto.ini`). New exports use schema version 2
and the radial-array key `lorentz_gamma`; previously generated archives
retain `lorentz_W` until regenerated.

Let `rho` be rest-mass density, `v` signed radial velocity, and
`h = 1 + Gamma/(Gamma-1) P/(rho c^2)` the dimensionless specific enthalpy.
The numerical constants match the problem: `G=6.674e-8`, `M_sun=1.988e33`,
`c=2.99792458e10` in cgs. Grid coordinates are converted with `UNIT_LENGTH`,
not assumed to be km.

- `lorentz_gamma = 1/sqrt(1-v^2/c^2)`.
- `sound_speed_cm_s = sqrt(Gamma P/(rho h))`: the actual RHD EOS sound speed,
  not the classical `sqrt(Gamma P/rho)`.
- `mach = v/c_s`: signed Mach number, not a four-velocity Mach convention.
- `mdot_g_s = 4 pi r^2 rho gamma v`.
- `K_cgs = P/rho^Gamma`; units depend on Gamma as given in metadata.
- `edot_kin_erg_s = mdot (gamma-1)c^2`: kinetic only, not total energy flux.
- `bernoulli_sr_potential_c2 = h*gamma - 1 + Phi/c^2`, `Phi=-GM/r`:
  retained to compare with the existing GIFs; **not an exact invariant of the
  configured source terms**.

All physical primitive arrays, both critical numerators, the gravity factor,
and both Bernoulli diagnostics are saved. Fixed-boundary cells are included in
the archive and marked by `active=False`; they are excluded from root searches
and scalar variation statistics. The source-balanced Bernoulli integral is
undefined there and is stored as NaN.

## The force implementation and entropy closure matter

Verified source locations in the configured PLUTO checkout:

- `Src/RHD/makefile`: RHD shares `rhs_source.c` with RMHD.
- `Src/RHD/mappers.c`, `Src/RHD/fluxes.c`: in c=1 units,
  `D=rho gamma`, `m=rho h gamma^2 v`, `E=rho h gamma^2-P`, energy flux `m`.
- `Src/RMHD/rhs_source.c`: `S_m=E g` and `S_E=m g`, with
  `g=-GM/r^2` from the saved problem's `init.c`.
- `Src/flag_shock.c`: `ENTROPY_SWITCH=ALWAYS` flags all cells for entropy
  inversion. `Src/adv_flux.c` advects `D K` with no gravity source.
- `Src/RHD/rhd_entropy_solve.c`: reconstructs pressure from entropy and
  **redefines energy**. Thus the independently updated energy equation is not
  the thermodynamic closure of these saved runs.

For a steady smooth branch, mass conservation and advected entropy give
`r^2 rho gamma v = const` and `K = const`. Combining these with the implemented
momentum equation gives, in physical units,

```text
Q = 1 - P/(rho h c^2 gamma^2)
gamma^2 (v - c_s^2/v) dv/dr = 2 c_s^2/r + Q g.
```

Therefore two numerators are exported separately:

```text
N_newtonian = 2 c_s^2/r - GM/r^2              [requested comparison]
N_impl      = 2 c_s^2/r - Q GM/r^2           [configured entropy closure]
```

The regular outward sonic-point requirement for this smooth steady equation
is `Mach=1` and `N_impl=0` at the same radius. The requested `N_newtonian`
becomes the same condition in the cold/nonrelativistic limit (`Q -> 1`). Neither
condition by itself proves steadiness, a smooth crossing, or convergence.

The same equations imply `d ln(h*gamma)/dr = Q g/c^2`, so the exported
source-consistent Bernoulli-balance diagnostic is

```text
bernoulli_balance_c2(r) = ln[h(r) gamma(r)] + Phi(r_ref)/c^2
                        - integral(r_ref to r) Q(s) g(s)/c^2 ds.
```

Here `r_ref` is the first active cell center, and the integral uses the
trapezoidal rule. Its nonrelativistic limit is the classical Bernoulli
quantity divided by c^2. It should be radially constant on a steady, smooth,
isentropic branch. This reference-dependent diagnostic is **not** a local
escape-energy classifier, and its sign does not establish that gas is unbound.
Do not demand its global constancy across shocks or disconnected inflow/outflow
branches. Changing `ENTROPY_SWITCH` requires rederiving the diagnostic; the
exporter deliberately rejects unsupported equation configurations.

For comparison only: without the entropy overwrite, the implemented energy
equation would imply `h*gamma exp(Phi/c^2)=const` in steady flow. That is not used as
the invariant of these `ALWAYS` runs.

## Roots, sampling, and tolerances

- `r_sonic_km` is the innermost negative-to-positive crossing of `Mach-1`.
  Inward supersonic flow (`Mach=-1`) is not labelled an outward sonic point.
- `r_stagnation_km` is the innermost velocity sign change of either direction.
  Counts and `crossings.csv` reveal ambiguity; no nearest-to-zero substitution
  is made when a crossing does not exist.
- Roots use adjacent valid active cells and linear interpolation in radius.
  An exact-zero plateau bracketed by opposite signs gives one midpoint root.
  Tangencies, endpoint zeros, and all-zero profiles are not internal crossings.
- Scalar fluxes are linearly interpolated **after** calculating the radial
  flux, at exactly 20, 100, and 1000 km. No extrapolation is performed. Thus
  these samples can differ slightly from the GIF's nearest-cell histories.
- Critical roots nearest to the selected sonic root are reported along with
  signed radial separation, separation in local cell spacings, Mach residual
  at the critical root, and N residual at the sonic root.
- `N_at_sonic_relative = N/(2 c_s^2/r + abs(gravity term))` uses the matching
  gravity term for each definition. A `critical_candidate_*` flag requires
  both a separation <= 2 local cell spacings and |relative N| <= 0.05.
  These explicit configurable tolerances are screens, not validation criteria.
- Missing quantities are NaN. A candidate flag of 0 also covers missing roots;
  distinguish that case using the counts and NaN radii.
- Relative spans use `(max-min)/max(abs(values))` over all active cells;
  Bernoulli spans are absolute in c^2 units. These are diagnostics, not an
  automatic declaration that a run has reached steady state.

Example radial access:

```python
import numpy as np
data = np.load("runs/rhd_experiments/resolution_2x/diagnostics/radial_profiles.npz")
r = data["radius_km"][data["active"]]
mach_final = data["mach"][-1, data["active"]]
```

Tests: `conda run -n retrieval python -B -m unittest discover -s analysis -p 'test_diagnose_rhd_wind.py'`.

## RMHD plotting Bernoulli diagnostic

`rmhd_diagnostics.py` supplies the nonrotating and rotating RMHD plotters with
the following quantities, expressed in units of `c^2`:

```text
fluid_bernoulli_c2 = h gamma - 1 - GM/(r c^2)
electromagnetic_bernoulli_c2 = poynting_power_erg_s / (mdot_g_s c^2)
bernoulli_c2 = fluid_bernoulli_c2 + electromagnetic_bernoulli_c2
```

The `Be` panel shows fluid, electromagnetic, and total contributions separately.
The fluid term includes internal energy and pressure through enthalpy, not just
kinetic energy. The electromagnetic term is transported Poynting power per unit
rest-mass flux, not local magnetic energy density divided by density. It uses
the full velocity vector in `gamma` and all three field/velocity components in
the radial Poynting flux. A nonrotating, purely radial field carries no radial
Poynting flux, so total `Be` then equals the fluid term wherever mass flux is
nonzero.

Zero or nonfinite mass flux and nonfinite Poynting power give a NaN ratio and
NaN total `Be`; they are not replaced with zero. A finite ratio can become very
large near stagnation. This retains the `-GM/r` comparison convention and is
**not an exact conserved integral of PLUTO's configured gravity source terms**.

The radial `Edot` panel shows kinetic, thermal/enthalpy, electromagnetic, and
total power separately. The time-history panel shows only total power at each
sampled radius. The totals have rest-mass energy subtracted:

```text
total_power_erg_s = mdot_g_s (h gamma - 1) c^2 + poynting_power_erg_s
thermal_enthalpy_power_erg_s = mdot_g_s gamma (h - 1) c^2
total_power_erg_s = kinetic_power_erg_s + thermal_enthalpy_power_erg_s
                  + poynting_power_erg_s
```

This includes kinetic energy, internal energy, pressure work, and electromagnetic
transport. It does not include gravitational potential energy, unlike `Be`.
It is therefore not an exact conserved gravity-inclusive budget. The power is
computed directly, not by multiplying `Be` by mass flux, so it remains defined
at zero radial mass flux when the underlying primitive state is finite.
`kinetic_plus_poynting_erg_s` retains the old partial diagnostic for other
analysis consumers; it is not the plotted total. The radial `Be`, `Edot`, and
`Jdot` panels consistently use a thicker solid green line for their total.
Both RMHD plotters also show the Lorentz factor `gamma` in a dedicated
row: the current radial profile and histories at 20, 100, 1000 km, and the outer
cell. This is `1/sqrt(1 - (v_r^2 + v_theta^2 + v_phi^2)/c^2)`, not a radial-only
Lorentz factor. The axis is labelled `gamma` (dimensionless), without a gray
initial overlay, scientific offset, or logarithmic scaling.

The radial plasma-beta panel is labelled `beta` (dimensionless) and shows only
the current profile. It compares gas pressure to **comoving** magnetic pressure:
`plasma_beta = 2 prs_code / b_squared_code`, where PLUTO absorbs `sqrt(4 pi)`
into its magnetic-field units. Equivalently this is `8 pi P / B_prime^2` in
Gaussian cgs. A dotted `beta=1` reference separates magnetic-pressure dominance
from gas-pressure dominance. Positive gas pressure with zero magnetic pressure
gives infinity; zero divided by zero or nonfinite inputs give NaN. Infinite or
undefined values are not replaced by finite values for plotting. Positive
finite profiles use a log axis; profiles containing zero use symlog.

The final row shows flux-based specific angular momentum `ell` in `cm^2/s`:
fluid, electromagnetic, and total radial profiles, plus total histories at
20, 100, 1000 km and the outer cell. For each component it is `Jdot / mdot`,
using the same equatorial-equivalent geometry as the torque panels. The fluid
term is `h gamma r v_phi` where the mass flux is nonzero, not simply `r v_phi`.
Zero or nonfinite mass flux, or a nonfinite torque numerator, gives NaN. Signed
symlog axes retain negative transport and zero values. No gray initial overlays
are added to these new radial panels.

Multi-component radial panels show only the current profiles, without gray
initial-condition overlays. Single-quantity radial panels retain the gray
initial/current comparison; time histories are unchanged.

The speed panel has a fixed, frameless legend above the data axes and two compact
crossing-status lines in a separate footer below them. No status boxes overlay
the curves. Sonic and Alfvén markers have short dotted guides from zero to the
crossing speed, rather than full-height vertical lines. The crossing calculations
are unchanged: these remain `v_r=c_s` and `v_r=v_A` speed-crossing diagnostics,
not a full RMHD characteristic/critical-surface calculation.

Tests: `conda run -n retrieval python -B -m unittest discover -s analysis -p 'test_analyze_rmhd_suites.py'`.
