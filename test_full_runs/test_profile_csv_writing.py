import cProfile
import unittest
import swacmod.solute as solute

class Test_Profile_Csv_Writing(unittest.TestCase):
	
	@unittest.skip("performance test")
	def test_csv_writing_bytes_cython(self):
		data, solute_aggregation = make_solute_aggregation()
		profile("solute.write_solute_csv(data, solute_aggregation)", data, solute_aggregation)
		self.assertEqual(1, 2)

def profile(command, data, solute_aggregation):
	globals = {"solute" : solute}
	locals = {"data": data, "solute_aggregation": solute_aggregation}
	cProfile.runctx(command, globals, locals)
	

def make_solute_aggregation():
	time_period_count = 10000
	node_count = 1000
	data = {
		"params": {
			"run_name": "aardvark",
			"time_periods": [0] * time_period_count,
			"node_areas" : [0] * node_count,
		}
	}
	solute_aggregation = solute.make_aggregation_array(data)
	i = 0.0
	for t in range(time_period_count):
		for n in range(node_count):
			solute_aggregation[t, n] = i / 1000
			i += 1
	return data, solute_aggregation
