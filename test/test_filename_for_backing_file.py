import unittest
import swacmod.time_series_data as ts

class Test_Filename_For_Backing_File(unittest.TestCase):
	def test_x(self):
		base_path = "aardvark/bat/"
		filename = "cat.csv"
		shape = "dog"
		actual = ts.calculate_filename_for_backing_file(base_path, filename, shape)
		expected = "aardvark/bat/cat.csv.dog.numpydumpy"
		self.assertEqual(expected, actual)
