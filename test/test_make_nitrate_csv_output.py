import io
import swacmod.nitrate as nitrate
import numpy as np
import unittest

class Test_Make_Nitrate_Csv_Output(unittest.TestCase):
	def setUp(self):
		self.expected_header_row = ["Stress Period", "Node", "Recharge Concentration (metric tons/m3)"]
		self.expected_header_row_str = '"Stress Period","Node","Recharge Concentration (metric tons/m3)"\r\n'

	def test_make_nitrate_csv_output_for_empty_aggregation(self):
		nitrate_aggregation = np.array([])
		expected = self.expected_header_row_str
		actual = make_nitrate_csv_adapter(nitrate_aggregation)
		self.assertEqual(expected, actual)

	def test_make_nitrate_csv_output_for_one_entry(self):
		nitrate_aggregation = np.array([[2.0]])
		expected = (
			self.expected_header_row_str +
			"1,1,2.0\r\n"
		)
		actual = make_nitrate_csv_adapter(nitrate_aggregation)
		self.assertEqual(expected, actual)

	def test_make_nitrate_csv_output_for_three_nodes(self):
		nitrate_aggregation = np.array([[2.0, 3.0, 5.0]])
		expected = (
			self.expected_header_row_str +
			"1,1,2.0\r\n" +
			"1,2,3.0\r\n" +
			"1,3,5.0\r\n"
		)
		actual = make_nitrate_csv_adapter(nitrate_aggregation)
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

def make_nitrate_csv_adapter(nitrate_aggregation):
	data = {
		"params" : {
			"run_name" : "aardvark"
		}
	}
	spy = OpenSpy()
	filename = nitrate.make_output_filename(data)
	nitrate_csv_rows = nitrate.make_nitrate_csv_output(nitrate_aggregation)
	nitrate.write_nitrate_csv_file(filename, nitrate_csv_rows, spy.open)
	spy.string_capture.seek(0)
	return spy.string_capture.read()

class OpenSpy:
	def __init__(self):
		self.string_capture = None

	def open(self, filename, mode, newline):
		self.string_capture = io.StringIO()
		self.string_capture.close = lambda: None
		return self.string_capture
