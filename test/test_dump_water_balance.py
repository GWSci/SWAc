import unittest
import swacmod.input_output as input_output
import sys
import tempfile
import os
import datetime

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
		output = {
			'rainfall_ts': [1, 2, 3],
			'pe_ts': [1, 2, 3],
			'pefac': [1, 2, 3],
			'canopy_storage': [1, 2, 3],
			'net_pefac': [1, 2, 3],
			'precip_to_ground': [1, 2, 3],
			'snowfall_o': [1, 2, 3],
			'rainfall_o': [1, 2, 3],
			'snowpack': [1, 2, 3],
			'snowmelt': [1, 2, 3],
			'net_rainfall': [1, 2, 3],
			'rapid_runoff': [1, 2, 3],
			'runoff_recharge': [1, 2, 3],
			'macropore_att': [1, 2, 3],
			'macropore_dir': [1, 2, 3],
			'percol_in_root': [1, 2, 3],
			'rawrew': [1, 2, 3],
			'tawtew': [1, 2, 3],
			'p_smd': [1, 2, 3],
			'smd': [1, 2, 3],
			'ae': [1, 2, 3],
			'rejected_recharge': [1, 2, 3],
			'perc_through_root': [1, 2, 3],
			'subroot_leak': [1, 2, 3],
			'interflow_bypass': [1, 2, 3],
			'interflow_store_input': [1, 2, 3],
			'interflow_volume': [1, 2, 3],
			'infiltration_recharge': [1, 2, 3],
			'interflow_to_rivers': [1, 2, 3],
			'recharge_store_input': [1, 2, 3],
			'recharge_store': [1, 2, 3],
			'combined_recharge': [1, 2, 3],
			'atten_input': [1, 2, 3],
			'sw_attenuation': [1, 2, 3],
			'pond_direct': [1, 2, 3],
			'pond_atten': [1, 2, 3],
			'open_water_ae': [1, 2, 3],
			'atten_input_actual': [1, 2, 3],
			'pond_over': [1, 2, 3],
			'sw_other': [1, 2, 3],
			'open_water_evap': [1, 2, 3],
			'swabs_ts': [1, 2, 3],
			'swdis_ts': [1, 2, 3],
			'combined_str': [1, 2, 3],
			'combined_ae': [1, 2, 3],
			'evt': [1, 2, 3],
			'average_in': [1, 2, 3],
			'average_out': [1, 2, 3],
			'total_storage_change': [1, 2, 3],
			'balance': [1, 2, 3],
		}
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
02/01/2024,2,100,0.6000000000000001,0.6000000000000001,0.6000000000000001,0.6000000000000001
"""
