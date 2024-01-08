import unittest
import numpy as np
from swacmod import compile_model
import swacmod.nitrate as nitrate

class Test_Nitrate_Mass_Balance(unittest.TestCase):
	def test_calculate_Pro(self):
		her_array_mm_per_day = np.array([1.0, 1.0, 1.0,     1.0, 0.0, -1.0, 1.0])
		p_non = np.array(               [0.0, 0.0, 2.0/3.0, 0.2, 0.2, 0.2, 0.0])
		Psmd = np.array(                [0.0, 0.0, 0.0,     0.0, 0.0, 0.0, 0.8])
		pherperc = np.array(            [0.0, 0.5, 0.0,     0.3, 0.3, 0.3, 0.0])
		expected = np.array(            [1.0, 0.5, 1.0/3.0, 0.5, 0.0, 0.0, 0.2 ])
		actual = nitrate._calculate_Pro(her_array_mm_per_day, p_non, pherperc, Psmd)
		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_m3_array_kg_per_day(self):
		m0_array_kg_per_day = np.array([2.0, 6.0, 15.0, 14.0, 14.0, 14.0, 55.0])
		Pro = np.array([1.0, 0.5, 1.0/3.0, 0.5, 0.0, 0.0, 0.2 ])
		expected = np.array([2.0, 3.0, 5.0, 7.0, 0.0, 0.0, 11.0])
		actual = nitrate._calculate_m3_array_kg_per_day(m0_array_kg_per_day, Pro)
		np.testing.assert_array_almost_equal(expected, actual)
	
	def test_calculate_dSMD_array_mm_per_day_for_zero_days(self):
		self.assert_dSMD_array_mm_per_day([], [], [])

	def test_calculate_dSMD_array_mm_per_day_for_one_day(self):
		self.assert_dSMD_array_mm_per_day([7.0], [7.0], [0.0])
		self.assert_dSMD_array_mm_per_day([4.0], [7.0], [3.0])
		self.assert_dSMD_array_mm_per_day([7.0], [7.0], [-3.0])

	def test_calculate_dSMD_array_mm_per_day_for_three_days(self):
		self.assert_dSMD_array_mm_per_day([90.0, 9.0, 1.0], [100.0, 10.0, 1.0], [10.0, 1.0, 0.0])
		self.assert_dSMD_array_mm_per_day([7.0, 100.0, 1000.0], [10.0, 100.0, 1000.0], [3.0, 0.0, -20.0])

	def assert_dSMD_array_mm_per_day(self, expected, input_smd, input_potential_smd):
		data = None
		output = {
			"smd" : np.array(input_smd),
			"p_smd" : np.array(input_potential_smd),
		}
		node = None
		actual = nitrate._calculate_dSMD_array_mm_per_day(data, output, node)
		expected_numpy = np.array(expected)
		np.testing.assert_array_almost_equal(expected_numpy, actual)

	def test_calculate_Psoilperc_for_zero_days(self):
		input_perc_through_root_mm_per_day = np.array([])
		input_TAW_array_mm = np.array([])
		expected = []
		self.assert_Psoilperc(expected, input_perc_through_root_mm_per_day, input_TAW_array_mm)

	def test_calculate_Psoilperc_when_percolation_greater_than_zero(self):
		input_perc_through_root_mm_per_day = np.array([2.0, 30.0, 500.0])
		input_TAW_array_mm = np.array([8.0, 70.0, 500.0])
		expected = [0.2, 0.3, 0.5]
		self.assert_Psoilperc(expected, input_perc_through_root_mm_per_day, input_TAW_array_mm)

	def test_calculate_Psoilperc_when_percolation_equal_to_zero(self):
		input_perc_through_root_mm_per_day = np.array([0.0])
		input_TAW_array_mm = np.array([1.0])
		expected = [0.0]
		self.assert_Psoilperc(expected, input_perc_through_root_mm_per_day, input_TAW_array_mm)

	def test_calculate_Psoilperc_when_denominator_equal_to_zero(self):
		input_perc_through_root_mm_per_day = np.array([0.0, 2.0])
		input_TAW_array_mm = np.array([0.0, -2.0])
		expected = [0.0, 0.0]
		self.assert_Psoilperc(expected, input_perc_through_root_mm_per_day, input_TAW_array_mm)

	def test_calculate_Psoilperc_when_percolation_less_than_zero(self):
		input_perc_through_root_mm_per_day = np.array([-1.0])
		input_TAW_array_mm = np.array([2.0])
		expected = [0.0]
		self.assert_Psoilperc(expected, input_perc_through_root_mm_per_day, input_TAW_array_mm)

	def assert_Psoilperc(self, expected, input_perc_through_root_mm_per_day, input_TAW_array_mm):
		data = None
		output = {
			"perc_through_root": input_perc_through_root_mm_per_day,
			"tawtew": input_TAW_array_mm,
		}
		node = None
		actual = nitrate._calculate_Psoilperc(data, output, node)
		expected_numpy = np.array(expected)
		np.testing.assert_array_almost_equal(expected_numpy, actual)

	def test_calculate_Pherperc_for_zero_days(self):
		input_perc_through_root_mm_per_day = np.array([])
		input_her_array_mm_per_day = np.array([])
		expected = []
		self.assert_Pherperc(expected, input_perc_through_root_mm_per_day, input_her_array_mm_per_day)

	def test_calculate_Pherperc_when_percolation_greater_than_zero(self):
		input_perc_through_root_mm_per_day = np.array([14.0, 33.0, 65.0])
		input_her_array_mm_per_day = np.array([2.0, 3.0, 5.0])
		expected = [7.0, 11.0, 13.0]
		self.assert_Pherperc(expected, input_perc_through_root_mm_per_day, input_her_array_mm_per_day)

	def test_calculate_Pherperc_when_percolation_equal_to_zero(self):
		input_perc_through_root_mm_per_day = np.array([0.0])
		input_her_array_mm_per_day = np.array([2.0])
		expected = [0.0]
		self.assert_Pherperc(expected, input_perc_through_root_mm_per_day, input_her_array_mm_per_day)

	def test_calculate_Pherperc_when_percolation_less_than_zero(self):
		input_perc_through_root_mm_per_day = np.array([-1.0])
		input_her_array_mm_per_day = np.array([2.0])
		expected = [0.0]
		self.assert_Pherperc(expected, input_perc_through_root_mm_per_day, input_her_array_mm_per_day)

	def test_calculate_Pherperc_when_HER_equals_zero(self):
		input_perc_through_root_mm_per_day = np.array([-1.0, 0.0, 1.0])
		input_her_array_mm_per_day = np.array([0.0, 0.0, 0.0])
		expected = [0.0, 0.0, 0.0]
		self.assert_Pherperc(expected, input_perc_through_root_mm_per_day, input_her_array_mm_per_day)

	def assert_Pherperc(self, expected, input_perc_through_root_mm_per_day, input_her_array_mm_per_day):
		data = None
		output = {
			"perc_through_root": input_perc_through_root_mm_per_day,
		}
		node = None
		actual = nitrate._calculate_Pherperc(data, output, node, input_her_array_mm_per_day)
		expected_numpy = np.array(expected)
		np.testing.assert_array_almost_equal(expected_numpy, actual)

	def test_M_soil_in_kg(self):
		# (Psmd+Pherperc)*M0
		m0_array_kg_per_day = np.array([0.0, 2.0, 2.0, 2.0, 12.0])
		Psmd = np.array([0.0, 1.0, 0.5, 0.0, 0.25])
		Pherperc = np.array([0.0, 0.0, 0.0, 0.5, 0.5])
		expected = np.array([0.0, 2.0, 1.0, 1.0, 9.0])
		actual = nitrate._calculate_M_soil_in_kg(m0_array_kg_per_day, Psmd, Pherperc)
		np.testing.assert_array_almost_equal(expected, actual)

	def test_M_soil_tot_kg(self):
		# From Calcs sheet:
		# Msoil_tot_initial+Msoil_in-M1

		# From Node_12719_WB sheet:
		# AX3=AI3+AR3-AS3
		# AI3 = Msoil_tot_initial (kg) = AX2
		# AR3 = Msoil_in (kg)
		# AS3 = M1
		# M_soil_tot_kg[day] = M_soil_tot_kg[day - 1] + Msoil_in - M1
		pass

	def test_calculate_M4_array_mm_per_day(self):
		M_soil_tot_kg = []
		m1_array_kg_per_day = []
		expected = []
		self.M4_array_mm_per_day(expected, M_soil_tot_kg, m1_array_kg_per_day)

		# From Calcs sheet:
		# M4 = Msoil_tot - Msoil_tot_initial

		# Therefore:
		# M4 = Msoil_tot - Msoil_tot_initial
		#    = Msoil_tot_initial + Msoil_in - M1 - Msoil_tot_initial
		#    = Msoil_in - M1
		pass

	def M4_array_mm_per_day(self, expected, M_soil_tot_kg, m1_array_kg_per_day):
		pass

	def test_calculate_M4_array_mm_per_day_for_zero_days(self):
		dSMD_array_mm_per_day = np.array([])
		her_array_mm_per_day = np.array([])
		m0_array_kg_per_day = np.array([])
		actual = self._calculate_M4_array_mm_per_day(dSMD_array_mm_per_day, her_array_mm_per_day, m0_array_kg_per_day)
		expected = np.array([])
		np.testing.assert_array_almost_equal(expected, actual)

	def _calculate_M4_array_mm_per_day(self, dSMD_array_mm_per_day, her_array_mm_per_day, m0_array_kg_per_day):
		Psmd = nitrate._calculate_Psmd(her_array_mm_per_day, dSMD_array_mm_per_day)
		return nitrate._calculate_M4_array_mm_per_day(dSMD_array_mm_per_day, her_array_mm_per_day, m0_array_kg_per_day, Psmd)

	def test_calculate_M4_array_mm_per_day_for_one_day_zero_dSMD(self):
		dSMD_array_mm_per_day = np.array([0.0])
		her_array_mm_per_day = np.array([2.0])
		m0_array_kg_per_day = np.array([5.0])
		actual = self._calculate_M4_array_mm_per_day(dSMD_array_mm_per_day, her_array_mm_per_day, m0_array_kg_per_day)
		expected = np.array([0.0])
		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_M4_array_mm_per_day_for_one_day_positive_dSMD(self):
		dSMD_array_mm_per_day = np.array([22.0])
		her_array_mm_per_day = np.array([2.0])
		m0_array_kg_per_day = np.array([5.0])
		actual = self._calculate_M4_array_mm_per_day(dSMD_array_mm_per_day, her_array_mm_per_day, m0_array_kg_per_day)
		expected = np.array([55.0])
		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_M4_array_mm_per_day_for_one_day_zero_her(self):
		dSMD_array_mm_per_day = np.array([22.0])
		her_array_mm_per_day = np.array([0.0])
		m0_array_kg_per_day = np.array([5.0])
		actual = self._calculate_M4_array_mm_per_day(dSMD_array_mm_per_day, her_array_mm_per_day, m0_array_kg_per_day)
		expected = np.array([0.0])
		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_M4_array_mm_per_day_for_one_day_negative_dSMD(self):
		dSMD_array_mm_per_day = np.array([-22.0])
		her_array_mm_per_day = np.array([2.0])
		m0_array_kg_per_day = np.array([5.0])
		actual = self._calculate_M4_array_mm_per_day(dSMD_array_mm_per_day, her_array_mm_per_day, m0_array_kg_per_day)
		expected = np.array([0.0])
		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_M4_array_mm_per_day_for_three_days_positive_dSMD(self):
		dSMD_array_mm_per_day = np.array([34.0, 57.0, 115.0])
		her_array_mm_per_day = np.array([2.0, 3.0, 5.0])
		m0_array_kg_per_day = np.array([7.0, 11.0, 13.0])
		actual = self._calculate_M4_array_mm_per_day(dSMD_array_mm_per_day, her_array_mm_per_day, m0_array_kg_per_day)
		expected = np.array([119.0, 209.0, 299.0])
		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_M4_array_mm_per_day_for_three_days_positive_dSMD_followed_by_three_days_negative_dSMD(self):
		dSMD_array_mm_per_day = np.array([34.0, 57.0, 115.0, -27.0, -29.0, -31])
		her_array_mm_per_day = np.array([2.0, 3.0, 5.0, 1.0, 1.0, 1.0])
		m0_array_kg_per_day = np.array([7.0, 11.0, 13.0, 1.0, 1.0, 1.0])
		actual = self._calculate_M4_array_mm_per_day(dSMD_array_mm_per_day, her_array_mm_per_day, m0_array_kg_per_day)
		expected = np.array([119.0, 209.0, 299.0, 0.0, 0.0, 0.0])
		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_M4_array_mm_per_day_for_three_days_positive_dSMD_followed_by_three_days_positive_dSMD_and_zero_HER(self):
		dSMD_array_mm_per_day = np.array([34.0, 57.0, 115.0, 27.0, 29.0, 31])
		her_array_mm_per_day = np.array([2.0, 3.0, 5.0, 0.0, 0.0, 0.0])
		m0_array_kg_per_day = np.array([7.0, 11.0, 13.0, 1.0, 1.0, 1.0])
		actual = self._calculate_M4_array_mm_per_day(dSMD_array_mm_per_day, her_array_mm_per_day, m0_array_kg_per_day)
		expected = np.array([119.0, 209.0, 299.0, 0.0, 0.0, 0.0])
		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_M4out_array_mm_per_day_for_zero_days(self):
		expected = []
		input_smd = []
		input_tawtew = []
		input_dSMD_array_mm_per_day = []
		input_M4_array_kg = []
		self.assert_M4out_array_mm_per_day(expected, input_smd, input_tawtew, input_dSMD_array_mm_per_day, input_M4_array_kg)

	def assert_M4out_array_mm_per_day(self, expected, input_smd, input_tawtew, input_dSMD_array_mm_per_day, input_M4_array_kg):
		data = None
		output = {
			"smd" : np.array(input_smd),
			"tawtew": np.array(input_tawtew),
		}
		node = None
		dSMD_array_mm_per_day = np.array(input_dSMD_array_mm_per_day)
		M4_array_kg = np.array(input_M4_array_kg)
		actual = nitrate._calculate_M4out_array_mm_per_day(data, output, node, dSMD_array_mm_per_day, M4_array_kg)
		expected_numpy = np.array(expected)
		np.testing.assert_array_almost_equal(expected_numpy, actual)

	def test_calculate_M4out_array_mm_per_day_for_one_day_zero_dSMD(self):
		input_smd = [1.0]
		input_tawtew = [8.0]
		input_dSMD_array_mm_per_day = [0.0]
		input_M4_array_kg = [0.0]
		expected = [0.0]
		self.assert_M4out_array_mm_per_day(expected, input_smd, input_tawtew, input_dSMD_array_mm_per_day, input_M4_array_kg)

	def test_calculate_M4out_array_mm_per_day_for_one_day_positive_dSMD(self):
		input_smd = [1.0]
		input_tawtew = [8.0]
		input_dSMD_array_mm_per_day = [22.0]
		input_M4_array_kg = [55.0]
		expected = [0.0]
		self.assert_M4out_array_mm_per_day(expected, input_smd, input_tawtew, input_dSMD_array_mm_per_day, input_M4_array_kg)

	def test_calculate_M4out_array_mm_per_day_for_one_day_negative_dSMD(self):
		input_smd = [1.0]
		input_tawtew = [8.0]
		input_dSMD_array_mm_per_day = [-22.0]
		input_M4_array_kg = [0.0]
		expected = [0.0]
		self.assert_M4out_array_mm_per_day(expected, input_smd, input_tawtew, input_dSMD_array_mm_per_day, input_M4_array_kg)

	def test_calculate_M4out_array_mm_per_day_for_three_days_positive_dSMD(self):
		input_smd = [1.0]
		input_tawtew = [8.0]
		input_dSMD_array_mm_per_day = [34.0, 57.0, 115.0]
		input_M4_array_kg = [119.0, 209.0, 299.0]
		expected = [0.0, 0.0, 0.0]
		self.assert_M4out_array_mm_per_day(expected, input_smd, input_tawtew, input_dSMD_array_mm_per_day, input_M4_array_kg)

	def test_calculate_M4out_array_mm_per_day_for_three_days_positive_dSMD_followed_by_three_days_negative_dSMD(self):
		input_smd = [1.0, 1.0, 1.0, 87.0, 13.0, 59.0]
		input_tawtew = [3.0, 3.0, 3.0, 600.0, 100.0, 400.0]
		input_dSMD_array_mm_per_day = [34.0, 57.0, 115.0, -27.0, -29.0, -31]
		input_M4_array_kg = [119.0, 209.0, 299.0, 0.0, 0.0, 0.0]
		expected = [0.0, 0.0, 0.0, 33.0, 198.0, 36.0]
		self.assert_M4out_array_mm_per_day(expected, input_smd, input_tawtew, input_dSMD_array_mm_per_day, input_M4_array_kg)

	def test_is_mass_balanced_for_empty_arrays(self):
		self.assert_masses_balanced([], [])
		self.assert_masses_balanced([1.234], [1.234])
		self.assert_masses_not_balanced([1.234], [5.678])
		self.assert_masses_balanced([1.234, 5.678, 9.012], [1.234, 5.678, 9.012])
		self.assert_masses_not_balanced([1.234, 5.678, 9.012], [1.234, 7.890, 9.012])
		self.assert_masses_balanced([0], [1e-4])
		self.assert_masses_not_balanced([0], [1e-3])

	def assert_masses_balanced(self, m1, m2):
		m1_np = np.array(m1)
		m2_np = np.array(m2)
		actual = nitrate._is_mass_balanced(m1_np, m2_np)
		self.assertTrue(actual)

	def assert_masses_not_balanced(self, m1, m2):
		m1_np = np.array(m1)
		m2_np = np.array(m2)
		actual = nitrate._is_mass_balanced(m1_np, m2_np)
		self.assertFalse(actual)

	def test_find_unbalanced_day_to_report(self):
		self.assert_unbalanced_day_to_report(0, [1.0], [2.0])
		self.assert_unbalanced_day_to_report(0, [10.0, 2.0], [5.0, 1.0])
		self.assert_unbalanced_day_to_report(1, [2.0, 10.0], [1.0, 5.0])
		self.assert_unbalanced_day_to_report(0, [5.0, 1.0], [10.0, 2.0])
		self.assert_unbalanced_day_to_report(1, [1.0, 5.0], [2.0, 10.0])
	
	def assert_unbalanced_day_to_report(self, expected, m1, m2):
		m1_np = np.array(m1)
		m2_np = np.array(m2)
		actual = nitrate._find_unbalanced_day_to_report(m1_np, m2_np)
		self.assertEqual(expected, actual)
