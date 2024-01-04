import unittest
import numpy as np
from swacmod import compile_model
import swacmod.nitrate as nitrate

class Test_Nitrate_Mass_Balance(unittest.TestCase):
	def test_calculate_m3_array_kg_per_day(self):
		data = None
		output = {
			"rapid_runoff" : np.array([55.0, 55.0]),
		}
		node = None
		her_array_mm_per_day = np.array([11.0, 0.0])
		m0_array_kg_per_day = np.array([7.0, 7.0])
		
		actual = nitrate._calculate_m3_array_kg_per_day(data, output, node, her_array_mm_per_day, m0_array_kg_per_day)
		expected = np.array([35.0, 0.0])

		np.testing.assert_array_almost_equal(expected, actual)
	
	def test_calculate_dSMD_array_mm_per_day_for_zero_days(self):
		data = None
		output = {
			"smd" : np.array([]),
		}
		node = None
		actual = nitrate._calculate_dSMD_array_mm_per_day(data, output, node)
		expected = np.array([])
		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_dSMD_array_mm_per_day_for_one_day(self):
		data = None
		output = {
			"smd" : np.array([7.0]),
		}
		node = None
		actual = nitrate._calculate_dSMD_array_mm_per_day(data, output, node)
		expected = np.array([7.0])
		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_dSMD_array_mm_per_day_for_three_days(self):
		data = None
		output = {
			"smd" : np.array([100.0, 10.0, 1.0]),
		}
		node = None
		actual = nitrate._calculate_dSMD_array_mm_per_day(data, output, node)
		expected = np.array([90.0, 9.0, 1.0])
		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_M4_array_mm_per_day_for_zero_days(self):
		dSMD_array_mm_per_day = np.array([])
		her_array_mm_per_day = np.array([])
		m0_array_kg_per_day = np.array([])
		actual = nitrate._calculate_M4_array_mm_per_day(dSMD_array_mm_per_day, her_array_mm_per_day, m0_array_kg_per_day)
		expected = np.array([])
		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_M4_array_mm_per_day_for_one_day_zero_dSMD(self):
		dSMD_array_mm_per_day = np.array([0.0])
		her_array_mm_per_day = np.array([2.0])
		m0_array_kg_per_day = np.array([5.0])
		actual = nitrate._calculate_M4_array_mm_per_day(dSMD_array_mm_per_day, her_array_mm_per_day, m0_array_kg_per_day)
		expected = np.array([0.0])
		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_M4_array_mm_per_day_for_one_day_positive_dSMD(self):
		dSMD_array_mm_per_day = np.array([22.0])
		her_array_mm_per_day = np.array([2.0])
		m0_array_kg_per_day = np.array([5.0])
		actual = nitrate._calculate_M4_array_mm_per_day(dSMD_array_mm_per_day, her_array_mm_per_day, m0_array_kg_per_day)
		expected = np.array([55.0])
		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_M4_array_mm_per_day_for_one_day_zero_her(self):
		dSMD_array_mm_per_day = np.array([22.0])
		her_array_mm_per_day = np.array([0.0])
		m0_array_kg_per_day = np.array([5.0])
		actual = nitrate._calculate_M4_array_mm_per_day(dSMD_array_mm_per_day, her_array_mm_per_day, m0_array_kg_per_day)
		expected = np.array([0.0])
		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_M4_array_mm_per_day_for_one_day_negative_dSMD(self):
		dSMD_array_mm_per_day = np.array([-22.0])
		her_array_mm_per_day = np.array([2.0])
		m0_array_kg_per_day = np.array([5.0])
		actual = nitrate._calculate_M4_array_mm_per_day(dSMD_array_mm_per_day, her_array_mm_per_day, m0_array_kg_per_day)
		expected = np.array([0.0])
		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_M4_array_mm_per_day_for_three_days_positive_dSMD(self):
		dSMD_array_mm_per_day = np.array([34.0, 57.0, 115.0])
		her_array_mm_per_day = np.array([2.0, 3.0, 5.0])
		m0_array_kg_per_day = np.array([7.0, 11.0, 13.0])
		actual = nitrate._calculate_M4_array_mm_per_day(dSMD_array_mm_per_day, her_array_mm_per_day, m0_array_kg_per_day)
		expected = np.array([119.0, 209.0, 299.0])
		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_M4_array_mm_per_day_for_three_days_positive_dSMD_followed_by_three_days_negative_dSMD(self):
		dSMD_array_mm_per_day = np.array([34.0, 57.0, 115.0, -27.0, -29.0, -31])
		her_array_mm_per_day = np.array([2.0, 3.0, 5.0, 1.0, 1.0, 1.0])
		m0_array_kg_per_day = np.array([7.0, 11.0, 13.0, 1.0, 1.0, 1.0])
		actual = nitrate._calculate_M4_array_mm_per_day(dSMD_array_mm_per_day, her_array_mm_per_day, m0_array_kg_per_day)
		expected = np.array([119.0, 209.0, 299.0, 0.0, 0.0, 0.0])
		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_M4_array_mm_per_day_for_three_days_positive_dSMD_followed_by_three_days_positive_dSMD_and_zero_HER(self):
		dSMD_array_mm_per_day = np.array([34.0, 57.0, 115.0, 27.0, 29.0, 31])
		her_array_mm_per_day = np.array([2.0, 3.0, 5.0, 0.0, 0.0, 0.0])
		m0_array_kg_per_day = np.array([7.0, 11.0, 13.0, 1.0, 1.0, 1.0])
		actual = nitrate._calculate_M4_array_mm_per_day(dSMD_array_mm_per_day, her_array_mm_per_day, m0_array_kg_per_day)
		expected = np.array([119.0, 209.0, 299.0, 0.0, 0.0, 0.0])
		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_M4out_array_mm_per_day_for_zero_days(self):
		data = None
		output = {
			"smd" : np.array([]),
			"tawtew": np.array([]),
		}
		node = None
		dSMD_array_mm_per_day = np.array([])
		M4_array_kg = np.array([])
		actual = nitrate._calculate_M4out_array_mm_per_day(data, output, node, dSMD_array_mm_per_day, M4_array_kg)
		expected = np.array([])
		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_M4out_array_mm_per_day_for_one_day_zero_dSMD(self):
		data = None
		output = {
			"smd" : np.array([1.0]),
			"tawtew": np.array([8.0]),
		}
		node = None
		dSMD_array_mm_per_day = np.array([0.0])
		M4_array_kg = np.array([0.0])
		actual = nitrate._calculate_M4out_array_mm_per_day(data, output, node, dSMD_array_mm_per_day, M4_array_kg)
		expected = np.array([0.0])
		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_M4out_array_mm_per_day_for_one_day_positive_dSMD(self):
		data = None
		output = {
			"smd" : np.array([1.0]),
			"tawtew": np.array([8.0]),
		}
		node = None
		dSMD_array_mm_per_day = np.array([22.0])
		M4_array_kg = np.array([55.0])
		actual = nitrate._calculate_M4out_array_mm_per_day(data, output, node, dSMD_array_mm_per_day, M4_array_kg)
		expected = np.array([0.0])
		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_M4out_array_mm_per_day_for_one_day_negative_dSMD(self):
		data = None
		output = {
			"smd" : np.array([1.0]),
			"tawtew": np.array([8.0]),
		}
		node = None
		dSMD_array_mm_per_day = np.array([-22.0])
		M4_array_kg = np.array([0.0])
		actual = nitrate._calculate_M4out_array_mm_per_day(data, output, node, dSMD_array_mm_per_day, M4_array_kg)
		expected = np.array([0.0])
		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_M4out_array_mm_per_day_for_three_days_positive_dSMD(self):
		data = None
		output = {
			"smd" : np.array([1.0]),
			"tawtew": np.array([8.0]),
		}
		node = None
		dSMD_array_mm_per_day = np.array([34.0, 57.0, 115.0])
		M4_array_kg = np.array([119.0, 209.0, 299.0])
		actual = nitrate._calculate_M4out_array_mm_per_day(data, output, node, dSMD_array_mm_per_day, M4_array_kg)
		expected = np.array([0.0, 0.0, 0.0])
		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_M4out_array_mm_per_day_for_three_days_positive_dSMD_followed_by_three_days_negative_dSMD(self):
		data = None
		output = {
			"smd" : np.array([1.0, 1.0, 1.0, 87.0, 13.0, 59.0]),
			"tawtew": np.array([3.0, 3.0, 3.0, 600.0, 100.0, 400.0]),
		}
		node = None
		dSMD_array_mm_per_day = np.array([34.0, 57.0, 115.0, -27.0, -29.0, -31])
		M4_array_kg = np.array([119.0, 209.0, 299.0, 0.0, 0.0, 0.0])
		actual = nitrate._calculate_M4out_array_mm_per_day(data, output, node, dSMD_array_mm_per_day, M4_array_kg)
		expected = np.array([0.0, 0.0, 0.0, 33.0, 198.0, 36.0])
		np.testing.assert_array_almost_equal(expected, actual)

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
