import unittest

class Test_Assert_Csv_Similar(unittest.TestCase):
	def test_empty_csv_files_are_equal(self):
		self.assert_testee_passes("", "")

	def test_csv_files_with_different_text_are_not_equal(self):
		self.assert_testee_fails("a", "b")

	def assert_testee_passes(self, expected, actual):
		assert_csv_equal(self, expected, actual)

	def assert_testee_fails(self, expected, actual):
		try:
			assert_csv_equal(self, expected, actual)
		except AssertionError as e:
			return
		self.fail()

def assert_csv_equal(test_case, expected, actual):
	test_case.assertEqual(expected, actual)
