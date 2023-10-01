import unittest
import swacmod.input_output as input_output

class Test_Load_Params_From_Yaml(unittest.TestCase):
	def test_nitrate_process_is_read_from_input_file(self):
		input = input_output.load_params_from_yaml(
			input_file="./test/resources/loading_params/input.yml", 
			input_dir="./test/resources/loading_params/")
		params = input["params"]
		actual = params["nitrate_process"]
		expected = "enabled"
		self.assertEquals(expected, actual)
