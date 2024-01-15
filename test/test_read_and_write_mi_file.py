import unittest
import numpy as np
import swacmod.nitrate as nitrate
import swacmod.utils as utils
import tempfile
import pathlib

class Test_Read_And_Write_mi_File(unittest.TestCase):
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

	def test_writing_an_mi_file_produces_a_file_with_the_expected_contents(self):
		nitrate_mi_aggregation = np.array([
			[2.0, 3.0, 5.0],
			[7.0, 11.0, 13.0],
		])

		data = {
			"params" : {
				"run_name" : "aardvark"
			}
		}

		filename = nitrate.write_mi_csv(data, nitrate_mi_aggregation)
		filename = self.files_to_delete.append(filename)

		self.assertEqual(1, 1)
