import swacmod.nitrate as nitrate
import numpy as np
import unittest

class Test_Make_Repeated_Array_Offset(unittest.TestCase):
	def test_making_repeated_array_offset_from_empty_array(self):
		input_array = np.empty(shape=(0))
		expected = np.empty(shape=(0, 0))
		actual = nitrate._make_repeated_array_offset(input_array)
		np.testing.assert_array_equal(expected, actual)

	def test_making_repeated_array_offset_from_length_1(self):
		input_array = np.array([2])
		expected = np.array([[2]])
		actual = nitrate._make_repeated_array_offset(input_array)
		np.testing.assert_array_equal(expected, actual)

	def test_making_repeated_array_offset_from_length_2(self):
		input_array = np.array([2, 3])
		expected = np.array([[2, 3, 0], [0, 2, 3]])
		actual = nitrate._make_repeated_array_offset(input_array)
		np.testing.assert_array_equal(expected, actual)

	def test_making_repeated_array_offset_from_length_3(self):
		input_array = np.array([2, 3, 5])
		expected = np.array([[2, 3, 5, 0, 0], [0, 2, 3, 5, 0], [0, 0, 2, 3, 5]])
		actual = nitrate._make_repeated_array_offset(input_array)
		np.testing.assert_array_equal(expected, actual)

	def test_making_repeated_array_offset_from_length_4(self):
		input_array = np.array([2, 3, 5, 7])
		expected = np.array([[2, 3, 5, 7, 0, 0, 0], [0, 2, 3, 5, 7, 0, 0], [0, 0, 2, 3, 5, 7, 0], [0, 0, 0, 2, 3, 5, 7]])
		actual = nitrate._make_repeated_array_offset(input_array)
		np.testing.assert_array_equal(expected, actual)

	def test_multiply_repeated_array_along_an_axis(self):
		repeating_array = np.array([[2, 3, 5, 0, 0], [0, 2, 3, 5, 0], [0, 0, 2, 3, 5]])
		multiply_by = np.array([7, 11, 13])
		expected = np.array([[14, 21, 35, 0, 0], [0, 22, 33, 55, 0], [0, 0, 26, 39, 65]])
		actual = convert_repeating_proportions_to_mass_reaching_water_table_2d_array_kg(repeating_array, multiply_by)
		np.testing.assert_array_equal(expected, actual)

def convert_repeating_proportions_to_mass_reaching_water_table_2d_array_kg(
		sliding_proportion_reaching_water_table_array_per_day,
		mass_reaching_water_table_array_kg):
	return sliding_proportion_reaching_water_table_array_per_day * mass_reaching_water_table_array_kg[:, np.newaxis]
