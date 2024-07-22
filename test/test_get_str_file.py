import unittest
import swacmod.model as m
import warnings
import swacmod.input_output as input_output
import test.file_test_helpers as file_test_helpers

class Test_Get_Str_File(unittest.TestCase):
	def test_get_str_file(self):
		data = {
			"params" : {
				"node_areas" : [-1, 100, 200, 300],
				"run_name" : "str-aardvark",
				"time_periods" : [1, 2],
				"num_nodes" : 3,
				"mf96_lrc" : [1, 1, 3],
				"routing_topology" : {
					1 : [1, 1, 1, 40, 50, 60, 70, 80, 90, 100],
					2 : [2, 1, 2, 40, 50, 60, 70, 80, 90, 100],
					3 : [3, 1, 3, 40, 50, 60, 70, 80, 90, 100],
				},
				"istcb1" : 0,
				"istcb2" : 0,
			}
		}
		runoff = [-1, 5, 7, 11, 13, 17, 19]
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=DeprecationWarning)
			warnings.filterwarnings("ignore", category=UserWarning)
			str = m.get_str_file(data, runoff)
			str.write_file()
		actual = file_test_helpers.slurp("output_files/str-aardvark.str")
		self.assertEqual(expected_sfr_aardvark, actual)

expected_sfr_aardvark = """# DELETE ME
         3         3         8         0         0     86400         0         0
         3         2         1  # stress period 1
    0    0    1    1    1              0  150.0000 4571.4287  -10.0000   60.0000
    0    0    2    2    1    1.399999976  150.0000 4571.4287  -10.0000   60.0000
    0    0    3    3    1    3.299999952  150.0000 4571.4287  -10.0000   60.0000
    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0
         3         2         1  # stress period 2
    0    0    1    1    1              0  150.0000 4571.4287  -10.0000   60.0000
    0    0    2    2    1    3.400000095  150.0000 4571.4287  -10.0000   60.0000
    0    0    3    3    1    5.699999809  150.0000 4571.4287  -10.0000   60.0000
    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0
"""