import unittest
from pathlib import Path
import swacmod_run
from swacmod import utils as u
import os
import swacmod.feature_flags as ff

class TestFixture():
	def __init__(self, test_instance, reference_output_folder, output_folder, input_file):
		self.test_instance = test_instance
		self.reference_output_folder = reference_output_folder
		self.output_folder = output_folder
		self.input_file = input_file
		test_instance.maxDiff = None
	
	def clear_output_directory(self):
		for f in Path(self.output_folder).iterdir():
			f.unlink()		

	def run_swacmod(self):
		default_input_file = u.CONSTANTS["INPUT_FILE"]
		default_input_dir = u.CONSTANTS["INPUT_DIR"]
		default_output_dir = u.CONSTANTS["OUTPUT_DIR"]
		try:
			u.CONSTANTS["INPUT_FILE"] = self.input_file
			u.CONSTANTS["INPUT_DIR"] = os.path.dirname(self.input_file)
			u.CONSTANTS["OUTPUT_DIR"] = self.output_folder
			swacmod_run.run(file_format="csv")
		finally:
			u.CONSTANTS["INPUT_FILE"] = default_input_file
			u.CONSTANTS["INPUT_DIR"] = default_input_dir
			u.CONSTANTS["OUTPUT_DIR"] = default_output_dir

	def assert_all_but_first_line_identical(self, filename):
		expected, actual = self.read_reference_and_actual_removing_first_line(filename)
		self.test_instance.assertEqual(expected, actual)

	def read_reference_and_actual_removing_first_line(self, filename):
		expected, actual = self.read_reference_and_actual(filename)
		expected = self.remove_first_line(expected)
		actual = self.remove_first_line(actual)
		return expected, actual

	def remove_first_line(self, s):
		first_line_end_index = s.find('\n') + 1
		return s[first_line_end_index:]

	def assert_file_is_identical(self, filename):
		expected, actual = self.read_reference_and_actual(filename)
		self.test_instance.assertEqual(expected, actual)
	
	def read_reference_and_actual(self, filename):
		expected_contents = self.read_file(self.reference_output_folder, filename)
		actual_contents = self.read_file(self.output_folder, filename)
		return expected_contents, actual_contents

	def read_file(self, folder, filename):
		path = Path(folder) / filename
		return path.read_text()

class Test_Demo_Models(unittest.TestCase):
	def test_demo_model(self):
		fixture = TestFixture(self, "test/reference_output/", "output_files/", "input_files/input.yml")
		fixture.clear_output_directory()

		default_use_natproc = ff.use_natproc
		try:
			ff.use_natproc = False
			fixture.run_swacmod()
		finally:
			ff.use_natproc = default_use_natproc
		fixture.assert_all_but_first_line_identical("my_run.evt")
		fixture.assert_all_but_first_line_identical("my_run.rch")
		fixture.assert_all_but_first_line_identical("my_run.sfr")
		fixture.assert_file_is_identical("my_runSpatial1980-01-01.csv")
		fixture.assert_file_is_identical("my_run_z_1.csv")
		fixture.assert_file_is_identical("my_run_z_2.csv")
