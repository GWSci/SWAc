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
	
def make_repeated_array_offset(array):
	length = len(array)
	result = np.broadcast_to(array, shape=(length, length))
	return result
