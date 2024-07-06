import unittest
import datetime

class Test_Release_Proportion(unittest.TestCase):
	def test_release_proportion_with_no_zones(self):
		data = {
			"params": {
				"sfr_flow_zones" : {},
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
			},
		}
		stream_ca_order = []
		time_period = [1, 2]
		actual = extract_release_proportion(data, stream_ca_order, time_period)
		expected = []
		self.assertEqual(expected, actual)

	def test_release_proportion_with_one_zone_and_one_node(self):
		data = {
			"params": {
				"sfr_flow_zones" : {
					1: [1]
				},
				"sfr_flow_monthly_proportions" : {
					1: [0.01],
					2: [0.02],
					3: [0.03],
					4: [0.04],
					5: [0.05],
					6: [0.06],
					7: [0.07],
					8: [0.08],
					9: [0.09],
					10: [0.1],
					11: [0.11],
					12: [0.12],
				},
			},
		}
		stream_ca_order = [(0, 0, 0)]
		self.assertEqual([0.01], extract_release_proportion(data, stream_ca_order, [1, 2]))
		self.assertEqual([0.02], extract_release_proportion(data, stream_ca_order, [32, 2]))

def extract_release_proportion(data, stream_ca_order, time_period):
	result = []
	month_key = 1
	if (time_period[0] == 32):
		month_key = 2
	for node_index, stream_index, downstream_stream_index in stream_ca_order:
		zone = 0
		result.append(data["params"]["sfr_flow_monthly_proportions"][month_key][zone])
	return result