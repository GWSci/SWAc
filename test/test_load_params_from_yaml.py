import numpy as np
import unittest
import swacmod.input_output as input_output
import swacmod.time_series_data

class Test_Load_Params_From_Yaml(unittest.TestCase):
	def test_historical_mi_array_kg_per_time_period_is_read_from_input_file(self):
		expected = np.array([
			[2.1, 3.0, 5.0],
			[7.0, 11.0, 13.0],
		])
		actual = load_params()["historical_mi_array_kg_per_time_period"]
		np.testing.assert_allclose(expected, actual)

	def test_historical_mi_array_kg_per_time_period_is_read_using_time_series_data(self):
		expected = swacmod.time_series_data.CsvTimeSeriesData
		actual = type(load_params()["historical_mi_array_kg_per_time_period"])
		self.assertEqual(expected, actual)

	def test_historical_nitrate_process_is_read_from_input_file(self):
		self.assertEqual("enabled", load_params()["historical_nitrate_process"])

	def test_historical_time_periods_is_read_from_input_file(self):
		expected = [[1, 4], [4, 8]]
		self.assertEqual(expected, load_params()["historical_time_periods"])

	def test_nitrate_process_is_read_from_input_file(self):
		self.assertEqual("enabled", load_params()["nitrate_process"])

	def test_historical_start_date_is_read_from_input_file(self):
		self.assertEqual("2024-01-18", load_params()["historical_start_date"])

	def test_nitrate_depth_to_water_is_read_from_input_file(self):
		expected = {
			1: [110],
			2: [120],
			3: [130],
		}
		self.assertEqual(expected, load_params()["nitrate_depth_to_water"])

	def test_nitrate_loading_is_read_from_input_file(self):
		expected = {
			1: [101, 201.1, 301.1, 401.1, 501.1, 601.1, 701.1, 801.1, 901.1, 101.1],
			2: [102, 202.2, 302.2, 402.2, 502.2, 602.2, 702.2, 802.2, 902.2, 102.2],
			3: [103, 203.3, 303.3, 403.3, 503.3, 603.3, 703.3, 803.3, 903.3, 103.3],
		}
		self.assertEqual(expected, load_params()["nitrate_loading"])

	def test_nitrate_calibration_a_is_read_from_input_file(self):
		self.assertEqual(1.38, load_params()["nitrate_calibration_a"])

	def test_nitrate_calibration_mu_is_read_from_input_file(self):
		self.assertEqual(1.58, load_params()["nitrate_calibration_mu"])

	def test_nitrate_calibration_sigma_is_read_from_input_file(self):
		self.assertEqual(3.96, load_params()["nitrate_calibration_sigma"])

	def test_nitrate_calibration_alpha_is_read_from_input_file(self):
		self.assertEqual(1.7, load_params()["nitrate_calibration_alpha"])

	def test_nitrate_calibration_effective_porosity_is_read_from_input_file(self):
		self.assertEqual(0.0029, load_params()["nitrate_calibration_effective_porosity"])

	def test_start_date_is_read_from_input_file(self):
		self.assertEqual("2024-01-17", load_params()["start_date"])

	def test_time_period_is_read_from_input_file(self):
		expected = [[1, 3], [3, 5]]
		self.assertEqual(expected, load_params()["time_periods"])

	def test_output_sfr_is_read_from_input_file(self):
		expected = True
		self.assertEqual(expected, load_params()["output_sfr"])

	def test_attenuate_sfr_flows_is_read_from_input_file(self):
		expected = True
		self.assertEqual(expected, load_params()["attenuate_sfr_flows"])

def load_params():
	input = input_output.load_params_from_yaml(
		input_file="./test/resources/loading_params/input.yml", 
		input_dir="./test/resources/loading_params/",
		tqdm=lambda x, desc: x)
	return input["params"]
