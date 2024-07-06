import unittest
import datetime

class Test_Release_Proportion(unittest.TestCase):
	def test_release_proportion_with_no_zones(self):
		data = {
			"params": {
				"sfr_flow_zones" : {}
			},
			"sfr_flow_monthly_proportions" : {
				1: [],
				2: [],
				3: [],
				4: [],
				5: [],
				6: [],
				7: [],
				8: [],
				9: [],
				10: [],
				11: [],
				12: [],
			},
		}
		date = datetime.datetime(2024, 1, 1)
		stream_ca_order = []
		time_period = [1, 2]
		actual = extract_release_proportion(data, stream_ca_order, time_period)
		expected = []
		self.assertEqual(expected, actual)

def extract_release_proportion(data, stream_ca_order, time_period):
	return []