"""Focused tests for RMHD suite physics and reduction helpers."""

import unittest

import numpy as np

from analyze_rmhd_suites import _json_safe, _robust_relative_span
from rmhd_diagnostics import (
    derive_rmhd_profiles, interpolate_profile, outward_crossing_radii,
)
from wind_common import C_CGS


UNITS = {
    "UNIT_DENSITY": 1.0e7,
    "UNIT_LENGTH": 1.0e5,
    "UNIT_VELOCITY": C_CGS,
}
PARAMETERS = {"GAMMA": 4.0 / 3.0, "M_NS": 1.4}


def snapshot(*, beta_r=0.2, beta_phi=0.0, br=0.1, bphi=0.0):
    count = 4
    return {
        "rho": np.ones(count) * 10.0,
        "prs": np.ones(count) * 0.2,
        "vx1": np.ones(count) * beta_r,
        "vx2": np.zeros(count),
        "vx3": np.ones(count) * beta_phi,
        "Bx1": np.ones(count) * br,
        "Bx2": np.zeros(count),
        "Bx3": np.ones(count) * bphi,
    }


class RMHDProfileTests(unittest.TestCase):
    def setUp(self):
        self.radius = np.array([10.0, 20.0, 40.0, 80.0])

    def test_radial_nonrotating_field_has_zero_poynting_and_torque(self):
        profile = derive_rmhd_profiles(snapshot(), self.radius, UNITS, PARAMETERS)
        np.testing.assert_allclose(profile["poynting_power_erg_s"], 0.0, atol=0.0)
        np.testing.assert_allclose(profile["angular_momentum_total_erg"], 0.0, atol=0.0)
        np.testing.assert_allclose(profile["field_phi_over_r"], 0.0, atol=0.0)
        self.assertTrue(np.all(profile["magnetization"] > 0.0))
        np.testing.assert_allclose(profile["electromagnetic_bernoulli_c2"], 0.0)
        np.testing.assert_allclose(
            profile["bernoulli_c2"], profile["fluid_bernoulli_c2"]
        )

    def test_lorentz_factor_uses_all_three_velocity_components(self):
        for beta_r, beta_theta, beta_phi in ((0.0, 0.0, 0.0), (0.2, 0.1, 0.3)):
            with self.subTest(velocity=(beta_r, beta_theta, beta_phi)):
                state = snapshot(beta_r=beta_r, beta_phi=beta_phi)
                state["vx2"][:] = beta_theta
                profile = derive_rmhd_profiles(state, self.radius, UNITS, PARAMETERS)
                expected = 1.0 / np.sqrt(
                    1.0 - beta_r**2 - beta_theta**2 - beta_phi**2
                )
                np.testing.assert_allclose(profile["lorentz_gamma"], expected)

    def test_plasma_beta_uses_comoving_magnetic_pressure(self):
        state = snapshot(beta_r=0.2, beta_phi=0.3, br=0.1, bphi=-0.04)
        state["vx2"][:] = 0.05
        state["Bx2"][:] = 0.07
        profile = derive_rmhd_profiles(state, self.radius, UNITS, PARAMETERS)
        beta_squared = 0.2**2 + 0.05**2 + 0.3**2
        comoving_b_squared = (
            (0.1**2 + 0.07**2 + (-0.04)**2) * (1.0 - beta_squared)
            + (0.2*0.1 + 0.05*0.07 + 0.3*(-0.04))**2
        )
        np.testing.assert_allclose(profile["plasma_beta"], 2.0*0.2/comoving_b_squared)

    def test_plasma_beta_zero_pressure_and_zero_field(self):
        with np.errstate(divide="raise", invalid="raise"):
            field_free = snapshot(br=0.0, bphi=0.0)
            profile = derive_rmhd_profiles(field_free, self.radius, UNITS, PARAMETERS)
            self.assertTrue(np.all(np.isposinf(profile["plasma_beta"])))
            field_free["prs"][:] = 0.0
            profile = derive_rmhd_profiles(field_free, self.radius, UNITS, PARAMETERS)
            self.assertTrue(np.all(np.isnan(profile["plasma_beta"])))
            cold_magnetic = snapshot(br=0.1)
            cold_magnetic["prs"][:] = 0.0
            profile = derive_rmhd_profiles(cold_magnetic, self.radius, UNITS, PARAMETERS)
            np.testing.assert_allclose(profile["plasma_beta"], 0.0)

    def test_specific_angular_momentum_is_torque_per_mass_flux(self):
        for beta_r in (0.2, -0.2):
            with self.subTest(beta_r=beta_r):
                profile = derive_rmhd_profiles(
                    snapshot(beta_r=beta_r, beta_phi=0.1, bphi=-0.03),
                    self.radius, UNITS, PARAMETERS,
                )
                for component in ("fluid", "electromagnetic", "total"):
                    np.testing.assert_allclose(
                        profile[f"specific_angular_momentum_{component}_cm2_s"],
                        profile[f"angular_momentum_{component}_erg"] / profile["mdot_g_s"],
                    )
                gamma = 1.0 / np.sqrt(1.0 - beta_r**2 - 0.1**2)
                enthalpy = 1.0 + 4.0 * 0.2 / 10.0
                np.testing.assert_allclose(
                    profile["specific_angular_momentum_fluid_cm2_s"],
                    enthalpy * gamma * self.radius * UNITS["UNIT_LENGTH"] * 0.1 * C_CGS,
                )
                np.testing.assert_allclose(
                    profile["specific_angular_momentum_total_cm2_s"],
                    profile["specific_angular_momentum_fluid_cm2_s"]
                    + profile["specific_angular_momentum_electromagnetic_cm2_s"],
                )

    def test_specific_angular_momentum_undefined_at_zero_mass_flux(self):
        with np.errstate(divide="raise", invalid="raise"):
            profile = derive_rmhd_profiles(
                snapshot(beta_r=0.0, beta_phi=0.1, bphi=-0.03),
                self.radius, UNITS, PARAMETERS,
            )
        for component in ("fluid", "electromagnetic", "total"):
            self.assertTrue(np.all(np.isnan(
                profile[f"specific_angular_momentum_{component}_cm2_s"]
            )))

    def test_bernoulli_includes_poynting_per_rest_mass_flux(self):
        for beta_r in (0.2, -0.2):
            with self.subTest(beta_r=beta_r):
                beta_phi, br, bphi = 0.1, 0.1, -0.03
                profile = derive_rmhd_profiles(
                    snapshot(beta_r=beta_r, beta_phi=beta_phi, br=br, bphi=bphi),
                    self.radius, UNITS, PARAMETERS,
                )
                gamma = 1.0 / np.sqrt(1.0 - beta_r**2 - beta_phi**2)
                # PLUTO fields absorb sqrt(4*pi); velocities here are in c units.
                expected_em = (beta_r*bphi**2 - beta_phi*br*bphi) / (
                    10.0 * gamma * beta_r
                )
                np.testing.assert_allclose(
                    profile["electromagnetic_bernoulli_c2"], expected_em
                )
                np.testing.assert_allclose(
                    profile["bernoulli_c2"],
                    profile["fluid_bernoulli_c2"] + expected_em,
                )
                np.testing.assert_allclose(
                    profile["electromagnetic_bernoulli_c2"],
                    profile["poynting_power_erg_s"] / profile["mdot_g_s"] / C_CGS**2,
                )

    def test_bernoulli_ratio_is_undefined_at_zero_mass_flux(self):
        for bphi in (0.0, -0.03):
            with self.subTest(bphi=bphi), np.errstate(divide="raise", invalid="raise"):
                profile = derive_rmhd_profiles(
                    snapshot(beta_r=0.0, beta_phi=0.1, bphi=bphi),
                    self.radius, UNITS, PARAMETERS,
                )
                self.assertTrue(np.all(np.isfinite(profile["fluid_bernoulli_c2"])))
                self.assertTrue(np.all(np.isnan(profile["electromagnetic_bernoulli_c2"])))
                self.assertTrue(np.all(np.isnan(profile["bernoulli_c2"])))

    def test_rotating_toroidal_case_has_finite_derived_fluxes(self):
        profile = derive_rmhd_profiles(
            snapshot(beta_phi=0.1, bphi=-0.03), self.radius, UNITS, PARAMETERS
        )
        for key in (
            "poynting_power_erg_s", "angular_momentum_fluid_erg",
            "angular_momentum_electromagnetic_erg", "angular_momentum_total_erg",
            "magnetization", "alfven_speed_c",
        ):
            self.assertTrue(np.all(np.isfinite(profile[key])), key)
        np.testing.assert_allclose(profile["field_phi_over_r"], -0.3)

    def test_total_power_includes_fluid_enthalpy_and_em(self):
        for beta_r in (0.2, -0.2):
            with self.subTest(beta_r=beta_r):
                profile = derive_rmhd_profiles(
                    snapshot(beta_r=beta_r, beta_phi=0.1, bphi=-0.03),
                    self.radius, UNITS, PARAMETERS,
                )
                gamma = 1.0 / np.sqrt(1.0 - beta_r**2 - 0.1**2)
                h = 1.0 + PARAMETERS["GAMMA"] / (PARAMETERS["GAMMA"] - 1.0) * 0.2/10.0
                expected = (
                    profile["mdot_g_s"] * (h*gamma - 1.0) * C_CGS**2
                    + profile["poynting_power_erg_s"]
                )
                np.testing.assert_allclose(profile["total_power_erg_s"], expected)
                np.testing.assert_allclose(
                    profile["thermal_enthalpy_power_erg_s"],
                    profile["mdot_g_s"] * gamma * (h - 1.0) * C_CGS**2,
                )
                np.testing.assert_allclose(
                    profile["total_power_erg_s"],
                    profile["kinetic_power_erg_s"]
                    + profile["thermal_enthalpy_power_erg_s"]
                    + profile["poynting_power_erg_s"],
                )
                np.testing.assert_allclose(
                    profile["kinetic_plus_poynting_erg_s"],
                    profile["kinetic_power_erg_s"] + profile["poynting_power_erg_s"],
                )

    def test_total_power_is_defined_at_zero_mass_flux(self):
        profile = derive_rmhd_profiles(
            snapshot(beta_r=0.0, beta_phi=0.1, bphi=-0.03),
            self.radius, UNITS, PARAMETERS,
        )
        self.assertTrue(np.all(np.isfinite(profile["total_power_erg_s"])))
        np.testing.assert_allclose(profile["thermal_enthalpy_power_erg_s"], 0.0)
        np.testing.assert_allclose(
            profile["total_power_erg_s"], profile["poynting_power_erg_s"]
        )

    def test_cold_total_power_reduces_to_kinetic_plus_em(self):
        cold = snapshot(beta_phi=0.1, bphi=-0.03)
        cold["prs"][:] = 0.0
        profile = derive_rmhd_profiles(cold, self.radius, UNITS, PARAMETERS)
        np.testing.assert_allclose(profile["thermal_enthalpy_power_erg_s"], 0.0)
        np.testing.assert_allclose(
            profile["total_power_erg_s"], profile["kinetic_plus_poynting_erg_s"]
        )

    def test_interpolation_is_exact_radius_not_nearest_cell(self):
        values = self.radius**2
        self.assertEqual(interpolate_profile(self.radius, values, 30.0), 1000.0)
        self.assertTrue(np.isnan(interpolate_profile(self.radius, values, 5.0)))

    def test_all_outward_speed_crossings_are_retained(self):
        profile = {
            "velocity_c": np.array([0.0, 2.0, 0.0, 2.0]),
            "sound_speed_c": np.ones(4),
            "alfven_speed_c": np.ones(4) * 0.5,
        }
        roots = outward_crossing_radii(
            self.radius, profile, np.array([False, True, True, True])
        )
        self.assertEqual(len(roots["sonic"]), 1)
        self.assertAlmostEqual(roots["sonic"][0], 60.0)


class ReductionTests(unittest.TestCase):
    def test_robust_relative_span(self):
        values = np.array([9.0, 10.0, 11.0])
        expected = (10.68 - 9.32) / 10.0
        self.assertAlmostEqual(_robust_relative_span(values), expected)

    def test_json_safe_replaces_nonfinite_values(self):
        result = _json_safe({"nan": np.nan, "inf": np.inf, "ok": np.float64(2.0)})
        self.assertIsNone(result["nan"])
        self.assertIsNone(result["inf"])
        self.assertEqual(result["ok"], 2.0)


if __name__ == "__main__":
    unittest.main()
