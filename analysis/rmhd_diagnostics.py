"""Pure numerical diagnostics shared by RMHD plotting and suite analysis."""

from __future__ import annotations

import numpy as np

from wind_common import C_CGS, derive_profiles, zero_crossings


SURFACE_RADIUS_KM = 12.0


def derive_rmhd_profiles(
    snapshot: dict[str, np.ndarray | float],
    radius_code: np.ndarray,
    units: dict[str, float],
    parameters: dict[str, float],
) -> dict[str, np.ndarray]:
    """Convert RMHD primitives to cgs profiles and derived fluxes.

    ``kinetic_power_erg_s + poynting_power_erg_s`` is a plotted diagnostic,
    not the full conserved energy flux: it omits fluid enthalpy and gravity.
    The angular-momentum flux is the 1D spherical, equatorial-equivalent
    torque obtained from the RMHD radial-azimuthal stress.
    """
    required = {"rho", "vx1", "vx2", "vx3", "Bx1", "Bx2", "Bx3", "prs"}
    missing = required - snapshot.keys()
    if missing:
        raise ValueError(f"RMHD snapshot is missing variables: {sorted(missing)}")

    density_unit = units["UNIT_DENSITY"]
    length_unit = units["UNIT_LENGTH"]
    velocity_unit = units["UNIT_VELOCITY"]
    radius_code = np.asarray(radius_code)
    fluid = derive_profiles(snapshot, radius_code, "RMHD", units, parameters)
    density_code = np.asarray(snapshot["rho"])
    beta_r = fluid["velocity_c"]
    beta_theta = np.asarray(snapshot["vx2"]) * velocity_unit / C_CGS
    beta_phi = np.asarray(snapshot["vx3"]) * velocity_unit / C_CGS
    gamma = fluid["lorentz_gamma"]
    enthalpy = fluid["enthalpy_c2"]
    radius_cm = radius_code * length_unit

    field_unit = velocity_unit * np.sqrt(4.0 * np.pi * density_unit)
    field_r_code = np.asarray(snapshot["Bx1"])
    field_theta_code = np.asarray(snapshot["Bx2"])
    field_phi_code = np.asarray(snapshot["Bx3"])
    field_r = field_r_code * field_unit
    field_theta = field_theta_code * field_unit
    field_phi = field_phi_code * field_unit

    beta_dot_field_code = (
        beta_r * field_r_code
        + beta_theta * field_theta_code
        + beta_phi * field_phi_code
    )
    comoving_field_squared_code = (
        (field_r_code**2 + field_theta_code**2 + field_phi_code**2) / gamma**2
        + beta_dot_field_code**2
    )
    magnetization = comoving_field_squared_code / (density_code * enthalpy)
    alfven_speed_c = np.sqrt(magnetization / (1.0 + magnetization))

    b0_code = gamma * beta_dot_field_code
    b_r_code = field_r_code / gamma + b0_code * beta_r
    b_phi_code = field_phi_code / gamma + b0_code * beta_phi
    fluid_rphi_code = density_code * enthalpy * gamma**2 * beta_r * beta_phi
    electromagnetic_rphi_code = (
        comoving_field_squared_code * gamma**2 * beta_r * beta_phi
        - b_r_code * b_phi_code
    )
    torque_scale = (
        4.0 * np.pi * radius_cm**3 * density_unit * velocity_unit**2
    )
    angular_momentum_fluid = torque_scale * fluid_rphi_code
    angular_momentum_electromagnetic = torque_scale * electromagnetic_rphi_code
    angular_momentum_total = (
        angular_momentum_fluid + angular_momentum_electromagnetic
    )

    velocity_r = beta_r * C_CGS
    velocity_theta = beta_theta * C_CGS
    velocity_phi = beta_phi * C_CGS
    poynting_power = radius_cm**2 * (
        velocity_r * (field_theta**2 + field_phi**2)
        - field_r * (velocity_theta * field_theta + velocity_phi * field_phi)
    )
    field_ratio = np.divide(
        field_phi,
        field_r,
        out=np.full_like(field_phi, np.nan),
        where=field_r != 0.0,
    )
    return {
        "radius_km": fluid["radius_km"],
        "density": fluid["density"],
        "pressure": fluid["pressure"],
        "temperature": fluid["temperature"],
        "velocity_c": beta_r,
        "velocity_phi_c": beta_phi,
        "sound_speed_c": fluid["sound_speed_c"],
        "alfven_speed_c": alfven_speed_c,
        "escape_velocity_c": fluid["escape_velocity_c"],
        "fluid_bernoulli_c2": fluid["bernoulli_c2"],
        "mdot_g_s": fluid["mdot_g_s"],
        "kinetic_power_erg_s": fluid["edot_kin_erg_s"],
        "poynting_power_erg_s": poynting_power,
        "kinetic_plus_poynting_erg_s": fluid["edot_kin_erg_s"] + poynting_power,
        # Backward-compatible animation key.
        "total_power_erg_s": fluid["edot_kin_erg_s"] + poynting_power,
        "angular_momentum_fluid_erg": angular_momentum_fluid,
        "angular_momentum_electromagnetic_erg": angular_momentum_electromagnetic,
        "angular_momentum_total_erg": angular_momentum_total,
        "field_r_gauss": field_r,
        "field_phi_gauss": field_phi,
        "field_phi_over_r": field_ratio,
        "magnetization": magnetization,
    }


def interpolate_profile(
    radius_km: np.ndarray,
    values: np.ndarray,
    requested_radius_km: float,
) -> float:
    """Linearly interpolate a derived profile at an exact physical radius."""
    radius_km = np.asarray(radius_km)
    values = np.asarray(values)
    valid = np.isfinite(radius_km) & np.isfinite(values)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    x = radius_km[valid]
    y = values[valid]
    if requested_radius_km < x[0] or requested_radius_km > x[-1]:
        return float("nan")
    return float(np.interp(requested_radius_km, x, y))


def outward_crossing_radii(
    radius_km: np.ndarray,
    profile: dict[str, np.ndarray],
    active: np.ndarray,
) -> dict[str, list[float]]:
    """Return every outward speed crossing without bridging invalid gaps."""
    pairs = {
        "sonic": profile["velocity_c"] - profile["sound_speed_c"],
        "alfven": profile["velocity_c"] - profile["alfven_speed_c"],
    }
    return {
        name: [root.radius_km for root in zero_crossings(radius_km, values, active)
               if root.direction == 1]
        for name, values in pairs.items()
    }


def relative_span(values: np.ndarray) -> float:
    """Peak-to-peak span divided by the largest absolute value."""
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan")
    scale = float(np.max(np.abs(values)))
    return float(np.ptp(values) / scale) if scale else 0.0
