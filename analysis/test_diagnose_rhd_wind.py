"""Focused numerical/edge-case tests; no simulation or GIF generation."""

import unittest

import numpy as np

from wind_common import C_CGS, G_CGS, M_SUN, derive_profiles, first_outward_zero_crossing
from diagnose_rhd_wind import radial_diagnostics, scalar_diagnostics, zero_crossings


UNITS = {"UNIT_DENSITY": 1e7, "UNIT_LENGTH": 1e5, "UNIT_VELOCITY": C_CGS}
PARAMETERS = {"GAMMA": 4.0 / 3.0, "M_NS": 1.4}


def primitives(rho, velocity, pressure):
    return {"rho": np.asarray(rho) / UNITS["UNIT_DENSITY"],
            "vx1": np.asarray(velocity) / C_CGS,
            "prs": np.asarray(pressure) / (UNITS["UNIT_DENSITY"] * C_CGS**2)}


class CrossingTests(unittest.TestCase):
    def roots(self, y, mask=None):
        return zero_crossings(np.arange(len(y), dtype=float) + 1, y,
                              np.ones(len(y), dtype=bool) if mask is None else mask)

    def test_multiple_directions(self):
        roots = self.roots([-1, 1, -3, 1])
        np.testing.assert_allclose([root.radius_km for root in roots], [1.5, 2.25, 3.75])
        self.assertEqual([root.direction for root in roots], [1, -1, 1])

    def test_exact_zero_and_plateau(self):
        self.assertEqual(self.roots([-1, 0, 1])[0].radius_km, 2)
        roots = self.roots([-1, 0, 0, 1])
        self.assertEqual(len(roots), 1)
        self.assertEqual(roots[0].radius_km, 2.5)

    def test_no_crossings(self):
        for y in ([0, 0, 0], [-1, 0, -1], [0, 1, 2], [-2, -1, 0], [1, 2, 3]):
            self.assertEqual(self.roots(y), [])

    def test_never_bridge_nan_or_boundary(self):
        self.assertEqual(self.roots([-1, np.nan, 1]), [])
        self.assertEqual(self.roots([-1, 0, 1], [True, False, True]), [])
        self.assertEqual(self.roots([-1, 1, 2], [False, True, True]), [])

    def test_plot_crossing_uses_same_gap_and_direction_rules(self):
        radius = np.arange(1., 6.)
        self.assertIsNone(first_outward_zero_crossing(radius, [-1, np.nan, 1, 1, 1], np.ones(5, bool)))
        self.assertIsNone(first_outward_zero_crossing(radius, [-1, 0, 1, 1, 1], [True, False, True, True, True]))
        self.assertEqual(first_outward_zero_crossing(radius, [1, -1, 0, 0, 1], np.ones(5, bool)), 3.5)


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.r = np.array([11., 20., 100., 1000., 2000.])
        self.active = self.r > 12
        self.rho = np.full(5, 1e10)
        self.v = C_CGS * np.array([0.1, 0., -0.2, 0.2, 0.4])
        self.P = self.rho * C_CGS**2 * 0.02
        self.snapshot = primitives(self.rho, self.v, self.P)

    def test_eos_and_flux(self):
        profile = radial_diagnostics(self.snapshot, self.r, UNITS, PARAMETERS, self.active)
        gamma = 1 / np.sqrt(1 - (self.v / C_CGS)**2)
        h = 1 + 4 * self.P / (self.rho * C_CGS**2)
        np.testing.assert_allclose(profile["mdot_g_s"], 4*np.pi*(self.r*1e5)**2*self.rho*gamma*self.v)
        np.testing.assert_allclose(profile["sound_speed_cm_s"], np.sqrt(PARAMETERS["GAMMA"]*self.P/(self.rho*h)))
        phi = -G_CGS*PARAMETERS["M_NS"]*M_SUN/(self.r*1e5*C_CGS**2)
        np.testing.assert_allclose(profile["bernoulli_sr_potential_c2"], h*gamma-1+phi)
        np.testing.assert_allclose(profile["mach"], self.v / profile["sound_speed_cm_s"])
        np.testing.assert_allclose(profile["K_cgs"], self.P / self.rho**PARAMETERS["GAMMA"])
        self.assertTrue(np.isnan(profile["bernoulli_balance_c2"][0]))

    def test_rmhd_uses_total_speed_for_lorentz_but_radial_mass_flux(self):
        snapshot = dict(self.snapshot, vx2=np.full(5, 0.3), vx3=np.full(5, 0.4))
        profile = derive_profiles(snapshot, self.r, "RMHD", UNITS, PARAMETERS)
        gamma = 1 / np.sqrt(1 - (self.v/C_CGS)**2 - 0.3**2 - 0.4**2)
        expected_mdot = 4*np.pi*(self.r*1e5)**2*self.rho*gamma*self.v
        np.testing.assert_allclose(profile["mdot_g_s"], expected_mdot)
        np.testing.assert_allclose(profile["edot_kin_erg_s"], expected_mdot*(gamma-1)*C_CGS**2)
        with self.assertRaises(ValueError):
            derive_profiles(dict(snapshot, vx3=np.ones(5)), self.r, "RMHD", UNITS, PARAMETERS)

    def test_signed_and_zero_kinetic_flux(self):
        profile = radial_diagnostics(self.snapshot, self.r, UNITS, PARAMETERS, self.active)
        self.assertEqual(profile["mdot_g_s"][1], 0)
        self.assertEqual(profile["edot_kin_erg_s"][1], 0)
        self.assertLess(profile["edot_kin_erg_s"][2], 0)
        self.assertGreater(profile["edot_kin_erg_s"][3], 0)

    def test_gravity_and_both_numerators(self):
        p = radial_diagnostics(self.snapshot, self.r, UNITS, PARAMETERS, self.active)
        radius_cm = self.r * 1e5
        g = -G_CGS * PARAMETERS["M_NS"] * M_SUN / radius_cm**2
        np.testing.assert_allclose(p["N_newtonian_cm_s2"], 2*p["sound_speed_cm_s"]**2/radius_cm + g)
        Q = 1 - self.P / (self.rho * p["enthalpy_c2"] * C_CGS**2 * p["lorentz_gamma"]**2)
        np.testing.assert_allclose(p["N_impl_cm_s2"], 2*p["sound_speed_cm_s"]**2/radius_cm + Q*g)
        self.assertTrue(np.all(p["N_impl_cm_s2"] > p["N_newtonian_cm_s2"]))

    def test_length_unit_conversion(self):
        p = radial_diagnostics(self.snapshot, self.r, UNITS, PARAMETERS, self.active)
        other = dict(UNITS, UNIT_LENGTH=2e5)
        q = radial_diagnostics(self.snapshot, self.r/2, other, PARAMETERS, self.active)
        for key in p:
            np.testing.assert_allclose(p[key], q[key], equal_nan=True)

    def test_invalid_primitives_fail(self):
        for field, value in (("rho", 0.), ("prs", -1.), ("vx1", 1.), ("rho", np.nan)):
            bad = {key: array.copy() for key, array in self.snapshot.items()}
            bad[field][2] = value
            with self.assertRaises(ValueError):
                radial_diagnostics(bad, self.r, UNITS, PARAMETERS, self.active)
        with self.assertRaises(ValueError):
            radial_diagnostics(dict(self.snapshot, vx2=np.ones(5)), self.r, UNITS, PARAMETERS, self.active)

    def test_cold_limit_and_small_velocity_precision(self):
        v = np.full(5, 1e-9*C_CGS)
        pressure = self.rho * C_CGS**2 * 1e-8
        p = radial_diagnostics(primitives(self.rho, v, pressure), self.r, UNITS, PARAMETERS, self.active)
        np.testing.assert_allclose(p["edot_kin_erg_s"], 0.5*p["mdot_g_s"]*v**2, rtol=1e-12)
        np.testing.assert_allclose(p["gravity_factor_impl"], 1., atol=1e-8)
        classical = (0.5*v[1]**2 + 4*pressure[1]/self.rho[1])/C_CGS**2 + p["potential_c2"][1]
        self.assertAlmostEqual(p["bernoulli_balance_c2"][1], classical, places=13)

    def test_analytic_isentropic_hydrostatic_balance(self):
        # For v=0, momentum with S_m=E*g gives
        # h+Gamma-1 = const * exp[-Phi/(Gamma*c^2)].
        r = np.geomspace(30., 200., 5001)
        Gamma = PARAMETERS["GAMMA"]
        phi = -G_CGS*PARAMETERS["M_NS"]*M_SUN/(r*1e5*C_CGS**2)
        h0 = 1.4
        h = (h0 + Gamma - 1)*np.exp(-(phi-phi[0])/Gamma) - (Gamma-1)
        theta = (h-1)*(Gamma-1)/Gamma
        rho = 1e10*(theta/theta[0])**(1/(Gamma-1))
        p = radial_diagnostics(primitives(rho, np.zeros_like(r), theta*rho*C_CGS**2), r, UNITS, PARAMETERS, np.ones(r.size, bool))
        self.assertLess(np.ptp(p["bernoulli_balance_c2"]), 1e-8)
        self.assertLess(np.ptp(p["K_cgs"])/np.mean(p["K_cgs"]), 1e-13)
        self.assertGreater(np.ptp(p["bernoulli_sr_potential_c2"]), 1e-4)


class ScalarTests(unittest.TestCase):
    def profile(self):
        r = np.array([10., 20., 100., 1000., 2000.])
        active = r > 12
        rho = np.ones(5)*1e10
        p = radial_diagnostics(primitives(rho, np.ones(5)*0.1*C_CGS, rho*C_CGS**2*0.02), r, UNITS, PARAMETERS, active)
        return r, active, p

    def test_no_sonic_and_no_stagnation_are_nan(self):
        r, active, p = self.profile()
        row, roots = scalar_diagnostics(p, r, active)
        self.assertTrue(np.isnan(row["r_sonic_km"]))
        self.assertTrue(np.isnan(row["r_stagnation_km"]))
        self.assertEqual(row["critical_candidate_impl"], 0)

    def test_exact_flux_interpolation_not_primitives(self):
        r, active, p = self.profile()
        r[2] = 120.
        row, _ = scalar_diagnostics(p, r, active)
        self.assertAlmostEqual(row["mdot_100_g_s"] / np.interp(100, r[active], p["mdot_g_s"][active]), 1.)
        self.assertEqual(row["mdot_out_g_s"], p["mdot_g_s"][-1])
        row, _ = scalar_diagnostics(p, r+100, active)
        self.assertTrue(np.isnan(row["mdot_20_g_s"]))
        self.assertTrue(np.isnan(row["mdot_100_g_s"]))

    def test_coincident_roots_and_missing_numerator(self):
        r, active, p = self.profile()
        p["mach"] = np.array([0., 0.5, 1.5, 2., 3.])
        p["N_impl_cm_s2"] = np.array([-1., -1., 1., 2., 3.])
        p["N_newtonian_cm_s2"] = np.ones(5)
        row, _ = scalar_diagnostics(p, r, active)
        self.assertEqual(row["r_sonic_km"], 60.)
        self.assertEqual(row["critical_impl_separation_cells"], 0.)
        self.assertEqual(row["critical_candidate_impl"], 1)
        self.assertEqual(row["critical_candidate_newtonian"], 0)
        self.assertTrue(np.isnan(row["r_critical_newtonian_nearest_km"]))

    def test_no_inward_sonic_mislabel_and_multiple_stagnation(self):
        r, active, p = self.profile()
        p["mach"] = np.array([-0.2, -0.5, -1.5, -2., -3.])
        p["velocity_cm_s"] = np.array([-1., -1., 1., -1., 1.])
        row, roots = scalar_diagnostics(p, r, active)
        self.assertTrue(np.isnan(row["r_sonic_km"]))
        self.assertEqual(row["n_stagnation_crossings"], 3)
        self.assertEqual(row["r_stagnation_km"], 60.)


if __name__ == "__main__":
    unittest.main()
