import swacmod.nitrate as nitrate
import numpy as np
import unittest

class Test_Make_Nitrate_Csv_Output(unittest.TestCase):
	def setUp(self):
		self.expected_header_row = ["Stress Period", "Node", "Recharge Concentration (metric tons/m3)"]

	def test_make_nitrate_csv_output_for_empty_aggregation(self):
		nitrate_aggregation = np.array([])
		expected = [
			self.expected_header_row,
		]
		actual = nitrate.make_nitrate_csv_output(nitrate_aggregation)
		self.assertEqual(expected, actual)

	def test_make_nitrate_csv_output_for_one_entry(self):
		nitrate_aggregation = np.array([[2.0]])
		expected = [
			self.expected_header_row,
			[1, 1, 2.0],
		]
		actual = nitrate.make_nitrate_csv_output(nitrate_aggregation)
		self.assertEqual(expected, actual)

	def test_make_nitrate_csv_output_for_three_nodes(self):
		nitrate_aggregation = np.array([[2.0, 3.0, 5.0]])
		expected = [
			self.expected_header_row,
			[1, 1, 2.0],
			[1, 2, 3.0],
			[1, 3, 5.0],
		]
		actual = nitrate.make_nitrate_csv_output(nitrate_aggregation)
		self.assertEqual(expected, actual)

	def test_make_nitrate_csv_output_for_three_stress_periods(self):
		nitrate_aggregation = np.array([
			[2.0],
			[3.0],
			[5.0],
		])
		expected = [
			self.expected_header_row,
			[1, 1, 2.0],
			[2, 1, 3.0],
			[3, 1, 5.0],
		]
		actual = nitrate.make_nitrate_csv_output(nitrate_aggregation)
		self.assertEqual(expected, actual)

	def test_make_nitrate_csv_output_for_multiple_nodes_and_stress_periods(self):
		nitrate_aggregation = np.array([
			[ 2.0,  3.0],
			[ 5.0,  7.0],
			[11.0, 13.0],
		])
		expected = [
			self.expected_header_row,
			[1, 1,  2.0],
			[1, 2,  3.0],
			[2, 1,  5.0],
			[2, 2,  7.0],
			[3, 1, 11.0],
			[3, 2, 13.0],
		]
		actual = nitrate.make_nitrate_csv_output(nitrate_aggregation)
		self.assertEqual(expected, actual)
