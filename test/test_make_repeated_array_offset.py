import numpy as np
import unittest

class Test_Make_Repeated_Array_Offset(unittest.TestCase):
	def test_making_repeated_array_offset_from_empty_array(self):
		input_array = np.empty(shape=(0))
		expected = np.empty(shape=(0, 0))
		actual = make_repeated_array_offset(input_array)
		np.testing.assert_array_equal(expected, actual)
	
	def test_making_repeated_array_offset_from_length_1(self):
		input_array = np.array([2])
		expected = np.array([[2]])
		actual = make_repeated_array_offset(input_array)
		np.testing.assert_array_equal(expected, actual)
	
	def test_making_repeated_array_offset_from_length_2(self):
		input_array = np.array([2, 3])
		expected = np.array([[2, 3, 0], [0, 2, 3]])
		actual = make_repeated_array_offset(input_array)
		np.testing.assert_array_equal(expected, actual)
	
def make_repeated_array_offset(array):
	length = len(array)
	if length == 2:
		padded_length = 3
		padded_array = np.zeros(padded_length)
		padded_array[0:length] = array
	else:
		padded_array = array
		padded_length = length
	result = np.broadcast_to(padded_array, shape=(length, padded_length))
	r, c = np.ogrid[:result.shape[0], :result.shape[1]]
	result = result[r, c - r]
	return result
