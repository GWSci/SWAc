import unittest

class Test_Assert_Csv_Similar(unittest.TestCase):
	def test_empty_csv_files_are_equal(self):
		assert_testee_passes("", "")
	
def assert_testee_passes(expected, actual):
	assert_csv_equal(expected, actual)

def assert_csv_equal(expected, actual):
	pass
