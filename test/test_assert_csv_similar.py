import test.csv_assertions
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

	def test_csv_files_with_different_row_counts_report_the_difference(self):
		self.assert_failure_message("Difference in row counts. Expected: 2 Actual: 3", self.get_assertion_result("a\nb\n", "a\nb\nc\n"))
		self.assert_failure_message("Difference in row counts. Expected: 1 Actual: 3", self.get_assertion_result("a\n", "a\nb\nc\n"))
		self.assert_failure_message("Difference in row counts. Expected: 1 Actual: 2", self.get_assertion_result("a\n", "a\nb\n"))
		self.assert_failure_message("Difference in row counts. Expected: 2 Actual: 1", self.get_assertion_result("a\nb\n", "a\n"))

	def test_csv_files_with_different_column_counts_reports_the_difference(self):
		self.assert_failure_message("Difference in column count for row=0. Expected: 2 Actual: 3", self.get_assertion_result("a,b\n", "a,b,c\n"))
		self.assert_failure_message("Difference in column count for row=0. Expected: 1 Actual: 3", self.get_assertion_result("a\n", "a,b,c\n"))
		self.assert_failure_message("Difference in column count for row=0. Expected: 1 Actual: 2", self.get_assertion_result("a\n", "a,b\n"))
		self.assert_failure_message("Difference in column count for row=0. Expected: 2 Actual: 1", self.get_assertion_result("a,b\n", "a\n"))
		self.assert_failure_message("Difference in column count for row=1. Expected: 2 Actual: 1", self.get_assertion_result("a\nb,c\n", "a\nb\n"))

	def test_csv_floats_that_have_the_same_text_are_equal(self):
		self.assert_passes(self.get_assertion_result("1.234", "1.234"))

	def test_csv_floats_that_are_very_different_are_different(self):
		self.assert_failure_message("Difference in row=0, col=0. Expected: 1.234 Actual: 1.235", self.get_assertion_result("1.234\n", "1.235\n"))
		self.assert_failure_message("Difference in row=0, col=0. Expected: 1.235 Actual: 1.234", self.get_assertion_result("1.235\n", "1.234\n"))

	def test_csv_floats_that_are_very_close_are_equal(self):
		self.assert_passes(self.get_assertion_result("1.0000001", "1.0000002"))
		self.assert_passes(self.get_assertion_result("1.0000002", "1.0000001"))

	def test_csv_dates_are_treated_like_strings(self):
		self.assert_passes(self.get_assertion_result("30/09/2014", "30/09/2014"))
		self.assert_failure_message("Difference in row=0, col=0. Expected: 30/09/2014 Actual: 20/09/2014", self.get_assertion_result("30/09/2014", "20/09/2014"))
		self.assert_failure_message("Difference in row=0, col=0. Expected: 30/09/2014 Actual: 30/08/2014", self.get_assertion_result("30/09/2014", "30/08/2014"))
		self.assert_failure_message("Difference in row=0, col=0. Expected: 30/09/2014 Actual: 30/09/2013", self.get_assertion_result("30/09/2014", "30/09/2013"))

	def test_identical_csv_files_are_equal(self):
		self.assert_passes(self.get_assertion_result("a", "a"))
		self.assert_passes(self.get_assertion_result("a,b,c\n", "a,b,c\n"))
		self.assert_passes(self.get_assertion_result("a\nb\nc\n", "a\nb\nc\n"))

	def assert_passes(self, actual_assertion_result):
		self.assertTrue(actual_assertion_result.is_pass, actual_assertion_result.message)

	def assert_failure_message(self, expected_message, actual_assertion_result):
		self.assertFalse(actual_assertion_result.is_pass)
		self.assertIn(expected_message, actual_assertion_result.message)

	def get_assertion_result(self, expected, actual):
		try:
			test.csv_assertions.assert_csv_similar(expected, actual)
			return AssertionResult(True, "")
		except AssertionError as e:
			return AssertionResult(False, e.args[0])

class AssertionResult:
	def __init__(self, is_pass, message):
		self.is_pass = is_pass
		self.message = message
