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

def load_params():
	input = input_output.load_params_from_yaml(
		input_file="./test/resources/loading_params/input.yml", 
		input_dir="./test/resources/loading_params/")
	return input["params"]
