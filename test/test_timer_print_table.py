import unittest
import swacmod.timer as timer

class test_timer_print_table(unittest.TestCase):
	def test_timer_print_table_for_an_empty_set_of_input_tokens(self):
		input_tokens = []
		actual = timer.make_time_table(input_tokens)
		expected = []
		self.assertEqual(expected, actual)

	def test_timer_print_table_for_one_input_token(self):
		input_tokens = [{"message": "aardvark", "elapsed_seconds": 1.23}]
		actual = timer.make_time_table(input_tokens)
		expected = ["aardvark: 1.23"]
		self.assertEqual(expected, actual)
