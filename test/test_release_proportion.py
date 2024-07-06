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
			"series": {
				"date" : make_date_series()
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
			"series": {
				"date" : make_date_series()
			},
		}
		stream_ca_order = [(0, 0, 0)]
		self.assertEqual([0.01], extract_release_proportion(data, stream_ca_order, [1, 2]))
		self.assertEqual([0.02], extract_release_proportion(data, stream_ca_order, [32, 2]))
		self.assertEqual([0.03], extract_release_proportion(data, stream_ca_order, [60, 2]))

	def test_release_proportion_with_several_zones_and_nodes(self):
		data = {
			"params": {
				"sfr_flow_zones" : {
					2: [1],
					3: [1],
					5: [2],
					7: [2],
					11: [3],
				},
				"sfr_flow_monthly_proportions" : {
					1: [0.01, 0.21, 0.41],
					2: [0.02, 0.22, 0.42],
					3: [0.03, 0.23, 0.43],
					4: [0.04, 0.24, 0.44],
					5: [0.05, 0.25, 0.45],
					6: [0.06, 0.26, 0.46],
					7: [0.07, 0.27, 0.47],
					8: [0.08, 0.28, 0.48],
					9: [0.09, 0.29, 0.49],
					10: [0.1, 0.3, 0.5],
					11: [0.11, 0.31, 0.51],
					12: [0.12, 0.32, 0.52],
				},
			},
			"series": {
				"date" : make_date_series()
			},
		}
		stream_ca_order = [
			(1, 0, 0),
			(10, 1, 0),
			(2, 2, 0),
			(6, 3, 0),
			(4, 4, 0),
		]
		self.assertEqual([0.01, 0.41, 0.01, 0.21, 0.21], extract_release_proportion(data, stream_ca_order, [1, 2]))

def make_date_series():
	max_time = 365 * 10
	start_date = datetime.datetime(1980, 1, 1)
	day = datetime.timedelta(1)
	return [start_date + day * num for num in range(max_time)]

def extract_release_proportion(data, stream_ca_order, time_period):
	result = []
	month_key = data["series"]["date"][time_period[0]].month
	zone_index_to_proportion = data["params"]["sfr_flow_monthly_proportions"][month_key]
	for node_index, stream_index, downstream_stream_index in stream_ca_order:
		node_number = node_index + 1
		zone = data["params"]["sfr_flow_zones"][node_number][0] - 1
		result.append(zone_index_to_proportion[zone])
	return result