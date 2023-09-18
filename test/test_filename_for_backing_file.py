import swacmod.time_series_data as ts
import unittest
import os

sep = os.sep

class Test_Filename_For_Backing_File(unittest.TestCase):
	def test_backing_filename_combines_base_directory_filename_and_meta_data(self):
		base_path = "aardvark/"
		filename = "cat.csv"
		shape = "dog"
		actual = ts.calculate_filename_for_backing_file(base_path, filename, shape)
		expected = f"aardvark{sep}cat.csv.dog.swacmod_array"
		self.assertEqual(expected, actual)

	def test_backing_filename_converts_forward_slashes_to_current_operating_system(self):
		base_path = "aardvark/bat/"
		filename = "cat.csv"
		shape = "dog"
		actual = ts.calculate_filename_for_backing_file(base_path, filename, shape)
		expected = f"aardvark{sep}bat{sep}cat.csv.dog.swacmod_array"
		self.assertEqual(expected, actual)
