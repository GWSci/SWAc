import io
import csv
import unittest

class Test_Assert_Csv_Similar(unittest.TestCase):
	def test_empty_csv_files_are_equal(self):
		self.assert_passes(self.get_assertion_result("", ""))

	def test_csv_files_with_different_text_are_not_equal(self):
		self.assert_failure_message("Difference in row=0, col=0. Expected: a Actual: b", self.get_assertion_result("a", "b"))
		self.assert_failure_message("Difference in row=0, col=0. Expected: a Actual: c", self.get_assertion_result("a", "c"))
		self.assert_failure_message("Difference in row=0, col=0. Expected: x Actual: b", self.get_assertion_result("x", "b"))

	def test_csv_files_reports_the_column_number(self):
		self.assert_failure_message("Difference in row=0, col=0. Expected: a Actual: x", self.get_assertion_result("a,b,c\n", "x,b,c\n"))
		self.assert_failure_message("Difference in row=0, col=1. Expected: b Actual: y", self.get_assertion_result("a,b,c\n", "a,y,c\n"))
		self.assert_failure_message("Difference in row=0, col=2. Expected: c Actual: z", self.get_assertion_result("a,b,c\n", "a,b,z\n"))

	def test_csv_files_reports_the_row_number(self):
		self.assert_failure_message("Difference in row=0, col=0. Expected: a Actual: x", self.get_assertion_result("a\nb\nc\n", "x\nb\nc\n"))
		self.assert_failure_message("Difference in row=1, col=0. Expected: b Actual: x", self.get_assertion_result("a\nb\nc\n", "a\nx\nc\n"))
		self.assert_failure_message("Difference in row=2, col=0. Expected: c Actual: x", self.get_assertion_result("a\nb\nc\n", "a\nb\nx\n"))

	def test_identical_csv_files_are_equal(self):
		self.assert_passes(self.get_assertion_result("a", "a"))

	def assert_passes(self, actual_assertion_result):
		self.assertTrue(actual_assertion_result.is_pass)

	def assert_failure_message(self, expected_message, actual_assertion_result):
		self.assertFalse(actual_assertion_result.is_pass)
		self.assertIn(expected_message, actual_assertion_result.message)

	def get_assertion_result(self, expected, actual):
		try:
			assert_csv_equal(expected, actual)
			return AssertionResult(True, "")
		except AssertionError as e:
			return AssertionResult(False, e.args[0])

class AssertionResult:
	def __init__(self, is_pass, message):
		self.is_pass = is_pass
		self.message = message

def assert_csv_equal(expected, actual):
	expected_grid = _read_csv(expected)
	actual_grid = _read_csv(actual)
	error_messages = []
	for row_index in range(len(expected_grid)):
		for col_index in range(len(expected_grid[0])):
			expected_cell = expected_grid[row_index][col_index]
			actual_cell = actual_grid[row_index][col_index]
			if (expected_cell != actual_cell):
				message = f"Difference in row={row_index}, col={col_index}. Expected: {expected_cell} Actual: {actual_cell}"
				error_messages.append(message)
	if (len(error_messages) > 0):
		raise AssertionError(error_messages)

def _read_csv(file_contents):
	file = io.StringIO(file_contents)
	csv_reader = csv.reader(file)
	result = []
	for row in csv_reader:
		result.append(row)
	return result