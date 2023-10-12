import numpy as np
import unittest

class Test_Make_Nitrate_Csv_Output(unittest.TestCase):
	def test_make_nitrate_csv_output_for_empty_aggregation(self):
		nitrate_aggregation = np.array([])
		expected = [
			["Stress Period", "Node", "Recharge Concentration (metric tons/m³)"],
		]
		actual = make_nitrate_csv_output(nitrate_aggregation)
		self.assertEqual(expected, actual)

	def test_make_nitrate_csv_output_for_one_entry(self):
		nitrate_aggregation = np.array([[2.0]])
		expected = [
			["Stress Period", "Node", "Recharge Concentration (metric tons/m³)"],
			[1, 1, 2.0],
		]
		actual = make_nitrate_csv_output(nitrate_aggregation)
		self.assertEqual(expected, actual)

	def test_make_nitrate_csv_output_for_three_nodes(self):
		nitrate_aggregation = np.array([[2.0, 3.0, 5.0]])
		expected = [
			["Stress Period", "Node", "Recharge Concentration (metric tons/m³)"],
			[1, 1, 2.0],
			[1, 2, 3.0],
			[1, 3, 5.0],
		]
		actual = make_nitrate_csv_output(nitrate_aggregation)
		self.assertEqual(expected, actual)

	def test_make_nitrate_csv_output_for_three_stress_periods(self):
		nitrate_aggregation = np.array([
			[2.0],
			[3.0],
			[5.0],
		])
		expected = [
			["Stress Period", "Node", "Recharge Concentration (metric tons/m³)"],
			[1, 1, 2.0],
			[2, 1, 2.0],
			[3, 1, 2.0],
		]
		actual = make_nitrate_csv_output(nitrate_aggregation)
		self.assertEqual(expected, actual)

def make_nitrate_csv_output(nitrate_aggregation):
	result = []
	result.append(["Stress Period", "Node", "Recharge Concentration (metric tons/m³)"])
	for stress_period_index, node_index in np.ndindex(nitrate_aggregation.shape):
		stress_period = stress_period_index + 1
		node = node_index + 1
		recharge_concentration = nitrate_aggregation[0, node_index]
		result.append([stress_period, node, recharge_concentration])
	return result