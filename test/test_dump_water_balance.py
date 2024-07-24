import unittest
import swacmod.input_output as input_output
import sys
import tempfile
import os

class Test_Dump_water_Balance(unittest.TestCase):
	def test_x(self):
		class_name = self.__class__.__name__
		test_name = sys._getframe().f_code.co_name
		run_name = f"run-{class_name}-{test_name}"

		data = {
			"params": {
				"num_nodes" : 1,
				"run_name": run_name,
				"node_areas": {1 : 100},
				"time_periods": [],
				"output_fac": 2,
			}
		}
		output = None
		file_format = "csv"
		output_dir = tempfile.gettempdir()
		node = 1
		zone = None
		reduced = True
		input_output.dump_water_balance(data, output, file_format, output_dir, node, zone, reduced)

		expected_path = os.path.join(output_dir, f"{run_name}_n_1.{file_format}")
		with open(expected_path) as file:
			actual = file.read()

		self.assertEqual(expected_csv, actual)

expected_csv = """DATE,nDays,Area,CombinedRecharge,CombinedSW,CombinedAE,UnitilisedPE
"""
