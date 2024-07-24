import unittest
import swacmod.input_output as input_output
import sys
import tempfile
import os
import datetime
import swacmod.utils as utils

class Test_Dump_water_Balance(unittest.TestCase):
	def test_dump_water_balance_csv(self):
		class_name = self.__class__.__name__
		test_name = sys._getframe().f_code.co_name
		run_name = f"run-{class_name}-{test_name}"

		date_series = [datetime.datetime(2024, 1, 1), datetime.datetime(2024, 1, 2), datetime.datetime(2024, 1, 3)]

		data = {
			"params": {
				"num_nodes" : 1,
				"run_name": run_name,
				"node_areas": {1 : 100},
				"time_periods": [[1, 3]],
				"output_fac": 2,
			},
			"series": {
				"date": date_series
			},
		}
		output = {}
		for column in utils.col_order():
			output[column] = [1, 2, 3]
		output.pop('date')
		output["combined_recharge"] = [10, 20, 30]
		output["combined_str"] = [100, 200, 300]
		output["combined_ae"] = [1000, 2000, 3000]
		output["evt"] = [10000, 20000, 30000]
		
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
02/01/2024,2,100,6.0,60.0,600.0,6000.0
"""
