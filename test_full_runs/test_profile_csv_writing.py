import cProfile
import unittest
import swacmod.nitrate as nitrate

class Test_Profile_Csv_Writing(unittest.TestCase):
	
	@unittest.skip("performance test")
	def test_csv_writing_bytes_cython(self):
		data, nitrate_aggregation = make_nitrate_aggregation()
		profile("nitrate.write_nitrate_csv_bytes_cython(data, nitrate_aggregation)", data, nitrate_aggregation)
		self.assertEqual(1, 2)

def profile(command, data, nitrate_aggregation):
	globals = {"nitrate" : nitrate}
	locals = {"data": data, "nitrate_aggregation": nitrate_aggregation}
	cProfile.runctx(command, globals, locals)
	

def make_nitrate_aggregation():
	time_period_count = 10000
	node_count = 1000
	data = {
		"params": {
			"run_name": "aardvark",
			"time_periods": [0] * time_period_count,
			"node_areas" : [0] * node_count,
		}
	}
	nitrate_aggregation = nitrate.make_aggregation_array(data)
	i = 0.0
	for t in range(time_period_count):
		for n in range(node_count):
			nitrate_aggregation[t, n] = i / 1000
			i += 1
	return data, nitrate_aggregation
