import unittest
import numpy as np
import swacmod.utils as utils
import swacmod.nitrate as nitrate
import tempfile
import pathlib

class Test_Write_Csv_File(unittest.TestCase):
	def setUp(self):
		self.original_output_dir = utils.CONSTANTS['OUTPUT_DIR']
		self.tempdir = tempfile.mkdtemp()
		self.files_to_delete = []
		utils.CONSTANTS['OUTPUT_DIR'] = self.tempdir

	def tearDown(self):
		utils.CONSTANTS['OUTPUT_DIR'] = self.original_output_dir
		for f in self.files_to_delete:
			pathlib.Path(f).unlink()
		pathlib.Path(self.tempdir).rmdir()

	def test_writing_a_nitrate_csv_file_produces_a_file_with_the_expected_contents(self):
		filename = self.write_csv_return_filename()
		actual = slurp(filename)
		expected = '"Stress Period","Node","Recharge Concentration (metric tons/m3)"\r\n1,1,2.1\r\n1,2,3\r\n1,3,5\r\n2,1,7\r\n2,2,11\r\n2,3,13\r\n'
		self.assertEqual(expected, actual)

	def write_csv_return_filename(self):
		nitrate_aggregation = np.array([
			[2.1, 3.0, 5.0],
			[7.0, 11.0, 13.0],
		])

		data = {
			"params" : {
				"run_name" : "aardvark"
			}
		}

		filename = nitrate.write_nitrate_csv(data, nitrate_aggregation)
		self.files_to_delete.append(filename)
		return filename

	def test_writing_a_stream_nitrate_csv_file_produces_a_file_with_the_expected_contents(self):
		filename = self.write_stream_nitrate_csv_return_filename()
		actual = slurp(filename)
		expected = '"Stress Period","Reach","Stream Concentration (metric tons/m3)"\r\n1,1,2.1\r\n1,2,3\r\n1,3,5\r\n2,1,7\r\n2,2,11\r\n2,3,13\r\n'
		self.assertEqual(expected, actual)

	def write_stream_nitrate_csv_return_filename(self):
		nitrate_aggregation = np.array([
			[2.1, 3.0, 5.0],
			[7.0, 11.0, 13.0],
		])

		data = {
			"params" : {
				"run_name" : "aardvark"
			}
		}

		filename = nitrate.write_stream_nitrate_csv(data, nitrate_aggregation)
		self.files_to_delete.append(filename)
		return filename

def slurp(filename):
	with open(filename, "r", newline="") as f:
		return f.read()
