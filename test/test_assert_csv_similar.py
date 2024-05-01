import unittest

class Test_Assert_Csv_Similar(unittest.TestCase):
	def test_empty_csv_files_are_equal(self):
		self.assert_testee_passes("", "")

	def test_csv_files_with_different_text_are_not_equal(self):
		actual = self.assert_testee_fails("a", "b")
		self.assert_failure_message("Difference in row=0, col=0. Expected: a Actual: b", actual)

	def assert_failure_message(self, expected_message, actual_assertion_result):
		self.assertFalse(actual_assertion_result.is_pass)

	def assert_testee_passes(self, expected, actual):
		assert_csv_equal(self, expected, actual)

	def assert_testee_fails(self, expected, actual):
		try:
			assert_csv_equal(self, expected, actual)
		except AssertionError as e:
			return AssertionResult(False, "")
		self.fail()

class AssertionResult:
	def __init__(self, is_pass, message):
		self.is_pass = is_pass
		self.message = message

def assert_csv_equal(test_case, expected, actual):
	test_case.assertEqual(expected, actual)
