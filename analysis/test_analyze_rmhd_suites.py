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
