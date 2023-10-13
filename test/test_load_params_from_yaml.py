import unittest
import swacmod.input_output as input_output

class Test_Load_Params_From_Yaml(unittest.TestCase):
	def test_nitrate_process_is_read_from_input_file(self):
		self.assertEqual("enabled", load_params()["nitrate_process"])

	def test_nitrate_depth_to_water_is_read_from_input_file(self):
		expected = {
			1: [110],
			2: [120],
			3: [130],
		}
		self.assertEqual(expected, load_params()["nitrate_depth_to_water"])

	def test_nitrate_leaching_is_read_from_input_file(self):
		expected = {
			1: [101, 201.1, 301.1, 401.1, 501.1, 601.1, 701.1, 801.1, 901.1, 101.1],
			2: [102, 202.2, 302.2, 402.2, 502.2, 602.2, 702.2, 802.2, 902.2, 102.2],
			3: [103, 203.3, 303.3, 403.3, 503.3, 603.3, 703.3, 803.3, 903.3, 103.3],
		}
		self.assertEqual(expected, load_params()["nitrate_loading"])

def load_params():
	input = input_output.load_params_from_yaml(
		input_file="./test/resources/loading_params/input.yml", 
		input_dir="./test/resources/loading_params/",
		tqdm=lambda x, desc: x)
	return input["params"]
