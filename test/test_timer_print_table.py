import unittest
import swacmod.timer as timer

class test_timer_print_table(unittest.TestCase):
	def test_timer_print_table_for_an_empty_set_of_input_tokens(self):
		input_tokens = []
		actual = timer._make_time_table(input_tokens)
		expected = []
		self.assertEqual(expected, actual)

	def test_timer_print_table_for_one_input_token(self):
		input_tokens = [{"message": "aardvark", "elapsed_seconds": 1.23}]
		actual = timer._make_time_table(input_tokens)
		expected = ["aardvark: 1.23"]
		self.assertEqual(expected, actual)

	def test_timer_print_table_aligns_colon_for_multiple_rows(self):
		input_tokens = [
			{"message": "aardvark", "elapsed_seconds": 1.23},
			{"message": "bat", "elapsed_seconds": 2.34},
		]
		actual = timer._make_time_table(input_tokens)
		expected = [
			"aardvark: 1.23",
			"bat     : 2.34",
		]
		self.assertEqual(expected, actual)

	def test_timer_print_table_aligns_colon_for_multiple_rows_when_the_longest_row_is_not_the_first(self):
		input_tokens = [
			{"message": "aardvark", "elapsed_seconds": 1.23},
			{"message": "bat", "elapsed_seconds": 2.34},
			{"message": "hippopotamus", "elapsed_seconds": 3.45},
		]
		actual = timer._make_time_table(input_tokens)
		expected = [
			"aardvark    : 1.23",
			"bat         : 2.34",
			"hippopotamus: 3.45",
		]
		self.assertEqual(expected, actual)

	def test_timer_print_table_aligns_decimal_point_in_times(self):
		input_tokens = [
			{"message": "bat", "elapsed_seconds": 12.34},
			{"message": "cat", "elapsed_seconds": 5.6},
		]
		actual = timer._make_time_table(input_tokens)
		expected = [
			"bat: 12.34",
			"cat:  5.6",
		]
		self.assertEqual(expected, actual)
