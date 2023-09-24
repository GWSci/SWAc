import unittest
from pathlib import Path
import swacmod_run
from swacmod import utils as u
import os

class TestFixture():
	def __init__(self, test_instance, reference_output_folder, output_folder):
		self.test_instance = test_instance
		self.reference_output_folder = reference_output_folder
		self.output_folder = output_folder
	
	def assert_file_is_identical(self, filename):
		expected_path = Path(self.reference_output_folder) / filename
		expected_contents = expected_path.read_text()
		actual_path = Path(self.output_folder) / filename
		actual_contents = actual_path.read_text()
		self.test_instance.assertEqual(expected_contents, actual_contents)


class Test_Demo_Models(unittest.TestCase):
	def test_demo_model(self):
		reference_output_folder = "test/reference_output/"
		output_folder = "output_files/"
		input_file = "input_files/input.yml"

		fixture = TestFixture(self, reference_output_folder, output_folder)

		for f in Path(output_folder).iterdir():
			f.unlink()

		default_input_file = u.CONSTANTS["INPUT_FILE"]
		default_input_dir = u.CONSTANTS["INPUT_DIR"]
		default_output_dir = u.CONSTANTS["OUTPUT_DIR"]
		try:
			u.CONSTANTS["INPUT_FILE"] = input_file
			u.CONSTANTS["INPUT_DIR"] = os.path.dirname(input_file)
			u.CONSTANTS["OUTPUT_DIR"] = output_folder
			swacmod_run.run()
		finally:
			u.CONSTANTS["INPUT_FILE"] = default_input_file
			u.CONSTANTS["INPUT_DIR"] = default_input_dir
			u.CONSTANTS["OUTPUT_DIR"] = default_output_dir

		fixture.assert_file_is_identical("my_runSpatial1980-01-01.csv")
