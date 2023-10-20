import io
import pathlib
import swacmod.nitrate as nitrate
import numpy as np
import unittest

class Test_Make_Nitrate_Csv_Output(unittest.TestCase):
	def setUp(self):
		self.expected_header_row = ["Stress Period", "Node", "Recharge Concentration (metric tons/m3)"]
		self.expected_header_row_str = '"Stress Period","Node","Recharge Concentration (metric tons/m3)"\n'

	def test_make_nitrate_csv_output_for_empty_aggregation(self):
		nitrate_aggregation = np.array([])
		expected = self.expected_header_row_str
		actual = make_nitrate_csv_adapter(nitrate_aggregation)
		self.assertEqual(expected, actual)

	def test_make_nitrate_csv_output_for_one_entry(self):
		nitrate_aggregation = np.array([[2.0]])
		expected = (
			self.expected_header_row_str +
			"1,1,2.0\n"
		)
		actual = make_nitrate_csv_adapter(nitrate_aggregation)
		self.assertEqual(expected, actual)

	def test_make_nitrate_csv_output_for_three_nodes(self):
		nitrate_aggregation = np.array([[2.0, 3.0, 5.0]])
		expected = (
			self.expected_header_row_str +
			"1,1,2.0\n" +
			"1,2,3.0\n" +
			"1,3,5.0\n"
		)
		actual = make_nitrate_csv_adapter(nitrate_aggregation)
		self.assertEqual(expected, actual)

	def test_make_nitrate_csv_output_for_three_stress_periods(self):
		nitrate_aggregation = np.array([
			[2.0],
			[3.0],
			[5.0],
		])
		expected = (
			self.expected_header_row_str +
			"1,1,2.0\n" +
			"2,1,3.0\n" +
			"3,1,5.0\n"
		)
		actual = make_nitrate_csv_adapter(nitrate_aggregation)
		self.assertEqual(expected, actual)

	def test_make_nitrate_csv_output_for_multiple_nodes_and_stress_periods(self):
		nitrate_aggregation = np.array([
			[ 2.0,  3.0],
			[ 5.0,  7.0],
			[11.0, 13.0],
		])
		expected = (
			self.expected_header_row_str +
			"1,1,2.0\n" +
			"1,2,3.0\n" +
			"2,1,5.0\n" +
			"2,2,7.0\n" +
			"3,1,11.0\n" +
			"3,2,13.0\n"
		)
		actual = make_nitrate_csv_adapter(nitrate_aggregation)
		self.assertEqual(expected, actual)

def make_nitrate_csv_adapter(nitrate_aggregation):
	data = {
		"params" : {
			"run_name" : "aardvark"
		}
	}
	spy = OpenSpy()
	nitrate.write_nitrate_csv(data, nitrate_aggregation)
	filename = nitrate.make_output_filename(data)
	return pathlib.Path(filename).read_text()

class OpenSpy:
	def __init__(self):
		self.string_capture = io.StringIO()
		self.string_capture.close = lambda: None

	def open(self, filename, mode, newline):
		return self.string_capture
